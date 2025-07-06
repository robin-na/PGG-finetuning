# baseline_gru_set.py
"""Baseline GRU Set model for Public Goods Games.

This script loads behavioural logs and environment parameters,
constructs a dataset with player exits handled, trains the model
for 10 epochs and exposes a `simulate_game` function. When run
directly it also saves the trained model to ``baseline_gru_set.pt``
so later sessions can call ``simulate_game`` without retraining.

Feature layout per round `t`
----------------------------
```
SET = { ENV_TOKEN , Player_1 , ... , Player_Nlive , FUND }
ENV_TOKEN : length 18 vector (slots 2-17 hold env scalars)
Player_i  : [ last_contribution_i , wallet_i_before , zeros(16) ]
FUND      : [ public_fund_t , mean_wallet_t , zeros(16) ]
```
Rows in `player-rounds.csv` where all behavioural fields are
`NaN`/`{}` indicate the player already left; that player is
removed from modelling after that round.
The loss forces all edge predictions to zero whenever both
punishment and reward mechanisms are disabled.
"""

import json
from dataclasses import dataclass
from typing import List

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

ENV_VARS = [
    "CONFIG_playerCount", "CONFIG_numRounds", "CONFIG_showNRounds",
    "CONFIG_allOrNothing", "CONFIG_chat", "CONFIG_defaultContribProp",
    "CONFIG_punishmentCost", "CONFIG_punishmentMagnitude",
    "CONFIG_rewardExists", "CONFIG_rewardCost", "CONFIG_rewardMagnitude",
    "CONFIG_showOtherSummaries", "CONFIG_showPunishmentId",
    "CONFIG_showRewardId", "CONFIG_MPCR", "CONFIG_endowment"
]

@dataclass
class RoundSample:
    env: torch.Tensor
    player_indices: List[int]
    last_contrib: List[float]
    wallets: List[float]
    public_fund: float
    mean_wallet: float
    target_contrib: List[float]
    target_edges: torch.Tensor
    disable_edges: bool

class SimpleSetEncoder(nn.Module):
    def __init__(self, input_dim=18, hidden_dim=128):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
        self.ln = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x)
        attn, _ = self.attn(h, h, h)
        h = self.ln(h + attn)
        h = self.ff(h) + h
        return h

class GRUSetModel(nn.Module):
    def __init__(self, num_players: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = SimpleSetEncoder(18, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.head_c = nn.Linear(hidden_dim, 1)
        self.head_edge = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
        self.hidden_dim = hidden_dim
        self.register_buffer("memory", torch.zeros(num_players, hidden_dim))

    def forward(self, sample: RoundSample):
        env_vec = torch.zeros(18)
        env_vec[2:18] = sample.env
        fund_vec = torch.zeros(18)
        fund_vec[0] = sample.public_fund
        fund_vec[1] = sample.mean_wallet
        player_vecs = []
        for c, w in zip(sample.last_contrib, sample.wallets):
            v = torch.zeros(18)
            v[0] = c
            v[1] = w
            player_vecs.append(v)
        x = torch.stack([env_vec] + player_vecs + [fund_vec], dim=0)
        z = self.encoder(x)
        player_z = z[1:-1]
        m_list = []
        c_hat = []
        for idx, emb in zip(sample.player_indices, player_z):
            prev = self.memory[idx]
            new = self.gru(emb, prev)
            self.memory[idx] = new.detach()
            m_list.append(new)
            c_hat.append(self.head_c(new).squeeze(-1))
        mem = torch.stack(m_list)
        c_hat = torch.stack(c_hat)
        N = len(sample.player_indices)
        pair_inputs = []
        for i in range(N):
            for j in range(N):
                pair_inputs.append(torch.cat([mem[i], mem[j]], dim=-1))
        pair_inputs = torch.stack(pair_inputs)
        out = self.head_edge(pair_inputs)
        p_hat = torch.sigmoid(out[:, 0]).reshape(N, N)
        u_hat = F.relu(out[:, 1]).reshape(N, N)
        return c_hat, p_hat, u_hat

def build_dataset(rounds_path: str, env_path: str):
    df = pd.read_csv(rounds_path)
    env_df = pd.read_csv(env_path)
    env_map = env_df.set_index('gameId')
    player_list = df['playerId'].unique().tolist()
    p2i = {p: i for i, p in enumerate(player_list)}
    samples: List[RoundSample] = []
    for gid, gdf in df.groupby('gameId'):
        gdf = gdf.sort_values('roundId')
        env_row = env_map.loc[gid]
        players = gdf['playerId'].unique().tolist()
        active = players.copy()
        wallets = {p: 0.0 for p in players}
        last_c = {p: 0.0 for p in players}
        prev_fund = 0.0
        env_tensor = torch.tensor(env_row[ENV_VARS].astype(float).values, dtype=torch.float32)
        for rid in sorted(gdf['roundId'].unique()):
            round_rows = gdf[gdf['roundId'] == rid]
            player_indices = [p2i[p] for p in active]
            last_contrib = [last_c[p] for p in active]
            w_before = [wallets[p] for p in active]
            target_c = []
            edges = torch.zeros(len(active), len(active))
            for i, p in enumerate(active):
                row = round_rows[round_rows['playerId'] == p]
                if row.empty or row[['data.contribution', 'data.punished', 'data.rewarded']].isna().all(axis=1).any():
                    contrib = float('nan')
                    target_c.append(contrib)
                else:
                    row = row.iloc[0]
                    contrib = row['data.contribution']
                    target_c.append(contrib)
                    pun = row['data.punished']
                    rew = row['data.rewarded']
                    if isinstance(pun, str) and pun != '{}':
                        for t, u in json.loads(pun).items():
                            if t in active:
                                j = active.index(t)
                                edges[i, j] -= float(u)
                    if isinstance(rew, str) and rew != '{}':
                        for t, u in json.loads(rew).items():
                            if t in active:
                                j = active.index(t)
                                edges[i, j] += float(u)
            samples.append(
                RoundSample(
                    env=env_tensor,
                    player_indices=player_indices,
                    last_contrib=last_contrib,
                    wallets=w_before,
                    public_fund=prev_fund,
                    mean_wallet=float(sum(w_before) / len(w_before)) if w_before else 0.0,
                    target_contrib=target_c,
                    target_edges=edges,
                    disable_edges=env_row['CONFIG_rewardExists'] == 0 and env_row['CONFIG_punishmentCost'] == 0,
                )
            )
            total_c = sum(c for c in target_c if pd.notna(c))
            share = total_c * env_row['CONFIG_MPCR'] / len(active)
            for i, p in enumerate(active):
                c = target_c[i]
                if pd.isna(c):
                    continue
                wallets[p] += env_row['CONFIG_endowment'] - c + share
                for j, t in enumerate(active):
                    u = edges[i, j]
                    if u < 0:
                        wallets[p] -= env_row['CONFIG_punishmentCost'] * abs(u)
                        wallets[t] += env_row['CONFIG_punishmentMagnitude'] * u
                    elif u > 0 and env_row['CONFIG_rewardExists'] == 1:
                        wallets[p] -= env_row['CONFIG_rewardCost'] * u
                        wallets[t] += env_row['CONFIG_rewardMagnitude'] * u
            for i, p in enumerate(active):
                last_c[p] = target_c[i] if pd.notna(target_c[i]) else 0.0
            leaving = [p for i, p in enumerate(active) if pd.isna(target_c[i])]
            for p in leaving:
                active.remove(p)
            prev_fund = total_c * env_row['CONFIG_MPCR']
    return samples, p2i

def train(model: GRUSetModel, samples: List[RoundSample], epochs: int = 10):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(epochs):
        total = 0.0
        count = 0
        model.memory.zero_()
        for s in tqdm(samples, desc=f"epoch{ep}"):
            c_hat, p_hat, u_hat = model(s)
            t_c = torch.tensor([0.0 if pd.isna(c) else c for c in s.target_contrib])
            mask = torch.tensor([not pd.isna(c) for c in s.target_contrib])
            loss = F.smooth_l1_loss(c_hat[mask], t_c[mask])
            exist = (s.target_edges.abs() > 0).float()
            loss += F.binary_cross_entropy(p_hat, exist)
            loss += F.smooth_l1_loss(u_hat, s.target_edges.abs())
            if s.disable_edges:
                loss += (p_hat.abs().sum() + u_hat.sum())
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            count += 1
        print(f"epoch {ep} mean loss {total / count:.4f}")

def save_trained_model(model: GRUSetModel, mapping: dict, path: str) -> None:
    """Persist model parameters and player mapping to disk."""
    torch.save({
        "state_dict": model.state_dict(),
        "player_mapping": mapping,
        "hidden_dim": model.hidden_dim,
    }, path)

def load_trained_model(path: str) -> (GRUSetModel, dict):
    """Load model parameters and mapping from ``save_trained_model``."""
    ckpt = torch.load(path, map_location="cpu")
    mapping = ckpt["player_mapping"]
    model = GRUSetModel(num_players=len(mapping), hidden_dim=ckpt.get("hidden_dim", 128))
    model.load_state_dict(ckpt["state_dict"])
    return model, mapping

def simulate_game(env_row: pd.Series, model: GRUSetModel, seed: int = 42):
    torch.manual_seed(seed)
    num_players = int(env_row['CONFIG_playerCount'])
    players = list(range(num_players))
    wallets = {i: 0.0 for i in players}
    last_c = {i: 0.0 for i in players}
    public_fund = 0.0
    history = []
    model.memory.zero_()
    for _ in range(int(env_row['CONFIG_numRounds'])):
        env_tensor = torch.tensor(env_row[ENV_VARS].values, dtype=torch.float32)
        sample = RoundSample(
            env=env_tensor,
            player_indices=players,
            last_contrib=[last_c[i] for i in players],
            wallets=[wallets[i] for i in players],
            public_fund=public_fund,
            mean_wallet=float(sum(wallets.values()) / len(players)),
            target_contrib=[0.0] * len(players),
            target_edges=torch.zeros(len(players), len(players)),
            disable_edges=env_row['CONFIG_rewardExists'] == 0 and env_row['CONFIG_punishmentCost'] == 0,
        )
        with torch.no_grad():
            c_hat, p_hat, u_hat = model(sample)
        contribs = c_hat.clamp(0, env_row['CONFIG_endowment']).tolist()
        edges = (u_hat * (p_hat > 0.5).float()).tolist()
        history.append((contribs, edges))
        total_c = sum(contribs)
        share = total_c * env_row['CONFIG_MPCR'] / len(players)
        for i in players:
            wallets[i] += env_row['CONFIG_endowment'] - contribs[i] + share
            for j in players:
                u = edges[i][j]
                if u > 0 and env_row['CONFIG_rewardExists'] == 1:
                    wallets[i] -= env_row['CONFIG_rewardCost'] * u
                    wallets[j] += env_row['CONFIG_rewardMagnitude'] * u
                if u > 0 and env_row['CONFIG_punishmentCost'] > 0:
                    wallets[i] -= env_row['CONFIG_punishmentCost'] * u
                    wallets[j] -= env_row['CONFIG_punishmentMagnitude'] * u
        for i in players:
            last_c[i] = contribs[i]
        public_fund = total_c * env_row['CONFIG_MPCR']
    return history

if __name__ == '__main__':
    rounds = 'data/raw_data/learning_wave/player-rounds.csv'
    env = 'data/processed_data/df_analysis_learn.csv'
    dataset, mapping = build_dataset(rounds, env)
    model = GRUSetModel(num_players=len(mapping))
    train(model, dataset, epochs=10)
    save_trained_model(model, mapping, "baseline_gru_set.pt")
    print("model saved to baseline_gru_set.pt")
    # After this script runs you can load the weights and simulate with:
    #   model, mapping = load_trained_model('baseline_gru_set.pt')
    #   env_row = pd.read_csv(env).iloc[0]
    #   history = simulate_game(env_row, model)
