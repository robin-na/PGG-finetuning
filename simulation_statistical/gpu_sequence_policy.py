from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from simulation_statistical.archetype_distribution_embedding.models.env_distribution_dirichlet import (
    DirichletEnvRegressor,
)
from simulation_statistical.archetype_distribution_embedding.utils.constants import REQUIRED_CONFIG_COLUMNS
from simulation_statistical.common import as_bool
from simulation_statistical.history_conditioned_policy import (
    DEFAULT_ARTIFACTS_ROOT,
    DEFAULT_LEARN_ANALYSIS_CSV,
    DEFAULT_LEARN_CLUSTER_WEIGHTS_PATH,
    DEFAULT_LEARN_ROUNDS_CSV,
    _action_dicts_by_player,
    _build_residual_store,
    _contrib_by_player,
    _round_payoff_by_player,
    _sample_residual,
)
from simulation_statistical.structured_sequence_policy import (
    ACTION_CLASSES,
    ACTION_LABEL_NONE,
    ACTION_LABEL_PUNISH,
    ACTION_LABEL_REWARD,
    ExactSequenceGameState,
    _action_edge_feature_row,
    _available_action_budget_coins,
    _build_round_state,
    _contribution_feature_row,
    _iter_training_batches,
    _masked_action_probabilities,
    _relative_peer_ids,
    _rows_to_numeric_frame,
)


DEFAULT_GPU_SEQUENCE_POLICY_MODEL_PATH = DEFAULT_ARTIFACTS_ROOT / "models" / "gpu_sequence_policy.pt"
DEFAULT_GPU_SEQUENCE_TRAIN_SUMMARY_PATH = (
    DEFAULT_ARTIFACTS_ROOT / "outputs" / "gpu_sequence_policy_train_summary.csv"
)
DEFAULT_GPU_ENV_MODEL_PATH = DEFAULT_ARTIFACTS_ROOT / "models" / "dirichlet_env_model.pkl"


@dataclass(frozen=True)
class GpuSequenceArchetypePolicyConfig:
    artifacts_root: str | None = None
    rebuild_model: bool = False
    device: str | None = None


class ContributionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.aon_head = nn.Linear(hidden_dim, 1)
        self.cont_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(features)
        return self.aon_head(hidden).squeeze(-1), self.cont_head(hidden).squeeze(-1)


class ActionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out = nn.Linear(hidden_dim, len(ACTION_CLASSES))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(features)
        return self.out(hidden)


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f"{path.stem}_",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = handle.name
        torch.save(dict(payload), tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _resolve_device(device: str | None) -> torch.device:
    requested = (device or os.environ.get("PGG_GPU_DEVICE") or "").strip()
    if requested:
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"Requested device '{requested}' but CUDA is not available.")
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _configure_torch(device: torch.device) -> None:
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def _frame_to_tensor(frame: pd.DataFrame, device: torch.device) -> torch.Tensor:
    array = np.asarray(frame.to_numpy(copy=False), dtype=np.float32)
    return torch.from_numpy(array).to(device=device, dtype=torch.float32, non_blocking=False)


def _compute_action_class_weights(
    none_count: int,
    punish_count: int,
    reward_count: int,
) -> torch.Tensor:
    counts = np.asarray(
        [
            max(int(none_count), 1),
            max(int(punish_count), 1),
            max(int(reward_count), 1),
        ],
        dtype=np.float32,
    )
    weights = float(np.sum(counts)) / (3.0 * counts)
    weights = weights / float(np.mean(weights))
    return torch.tensor(weights, dtype=torch.float32)


def _print_progress(prefix: str, message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {prefix} {message}", flush=True)


def train_gpu_sequence_policy(
    *,
    output_model_path: Path = DEFAULT_GPU_SEQUENCE_POLICY_MODEL_PATH,
    summary_output_path: Path = DEFAULT_GPU_SEQUENCE_TRAIN_SUMMARY_PATH,
    learn_cluster_weights_path: Path = DEFAULT_LEARN_CLUSTER_WEIGHTS_PATH,
    learn_analysis_csv: Path = DEFAULT_LEARN_ANALYSIS_CSV,
    learn_rounds_csv: Path = DEFAULT_LEARN_ROUNDS_CSV,
    device: str | None = None,
    epochs: int = 6,
    batch_size: int = 512,
    hidden_dim: int = 256,
    dropout: float = 0.10,
    lr: float = 5e-4,
    weight_decay: float = 1e-5,
    progress_every: int = 25,
    max_batches_per_epoch: int | None = None,
    seed: int = 0,
) -> Dict[str, Any]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    resolved_device = _resolve_device(device)
    _configure_torch(resolved_device)
    _print_progress("gpu-train", f"start device={resolved_device} epochs={epochs} batch_size={batch_size}")

    contribution_feature_names: set[str] = set()
    action_feature_names: set[str] = set()
    total_batches = 0
    contrib_rows_count = 0
    contrib_aon_count = 0
    contrib_cont_count = 0
    contrib_target_sum = 0.0
    contrib_aon_positive = 0
    action_rows_count = 0
    action_none_count = 0
    action_punish_count = 0
    action_reward_count = 0

    for batch_index, batch in enumerate(
        _iter_training_batches(
            learn_cluster_weights_path=learn_cluster_weights_path,
            learn_analysis_csv=learn_analysis_csv,
            learn_rounds_csv=learn_rounds_csv,
            batch_size=batch_size,
        ),
        start=1,
    ):
        total_batches += 1
        for row in batch["contribution_rows"]:
            contribution_feature_names.update(row.keys())
        for row in batch["action_rows"]:
            action_feature_names.update(row.keys())

        contrib_targets = np.asarray(batch["contribution_targets"], dtype=float)
        aon_mask = np.asarray(batch["contribution_aon"], dtype=int) == 1
        cont_mask = ~aon_mask
        contrib_rows_count += int(len(contrib_targets))
        contrib_aon_count += int(np.sum(aon_mask))
        contrib_cont_count += int(np.sum(cont_mask))
        contrib_target_sum += float(np.sum(contrib_targets))
        contrib_aon_positive += int(np.sum(contrib_targets[aon_mask] >= 0.5))

        action_labels = np.asarray(batch["action_labels"], dtype=int)
        action_rows_count += int(len(action_labels))
        action_none_count += int(np.sum(action_labels == ACTION_LABEL_NONE))
        action_punish_count += int(np.sum(action_labels == ACTION_LABEL_PUNISH))
        action_reward_count += int(np.sum(action_labels == ACTION_LABEL_REWARD))

        if max_batches_per_epoch is not None and batch_index >= int(max_batches_per_epoch):
            break
        if progress_every > 0 and batch_index % progress_every == 0:
            _print_progress("gpu-train", f"discover batches={batch_index}")

    if contrib_rows_count == 0:
        raise ValueError("GPU sequence contribution training data is empty.")
    if action_rows_count == 0:
        raise ValueError("GPU sequence action training data is empty.")

    contribution_feature_columns = sorted(contribution_feature_names)
    action_feature_columns = sorted(action_feature_names)
    _print_progress(
        "gpu-train",
        (
            f"discovered batches={total_batches} contrib_cols={len(contribution_feature_columns)} "
            f"action_cols={len(action_feature_columns)} contrib_rows={contrib_rows_count} "
            f"action_rows={action_rows_count}"
        ),
    )

    contrib_model = ContributionHead(
        input_dim=len(contribution_feature_columns),
        hidden_dim=int(hidden_dim),
        dropout=float(dropout),
    ).to(resolved_device)
    action_model = ActionHead(
        input_dim=len(action_feature_columns),
        hidden_dim=int(hidden_dim),
        dropout=float(dropout),
    ).to(resolved_device)

    contrib_optimizer = torch.optim.AdamW(
        contrib_model.parameters(),
        lr=float(lr),
        weight_decay=float(weight_decay),
    )
    action_optimizer = torch.optim.AdamW(
        action_model.parameters(),
        lr=float(lr),
        weight_decay=float(weight_decay),
    )

    negative_aon = max(int(contrib_aon_count) - int(contrib_aon_positive), 0)
    if contrib_aon_positive > 0:
        aon_pos_weight = torch.tensor(
            [float(max(negative_aon, 1) / max(int(contrib_aon_positive), 1))],
            dtype=torch.float32,
            device=resolved_device,
        )
    else:
        aon_pos_weight = torch.tensor([1.0], dtype=torch.float32, device=resolved_device)
    action_class_weights = _compute_action_class_weights(
        none_count=action_none_count,
        punish_count=action_punish_count,
        reward_count=action_reward_count,
    ).to(resolved_device)
    aon_loss_fn = nn.BCEWithLogitsLoss(pos_weight=aon_pos_weight)
    action_loss_fn = nn.CrossEntropyLoss(weight=action_class_weights)

    epoch_rows: List[Dict[str, Any]] = []
    for epoch_index in range(1, int(epochs) + 1):
        contrib_model.train()
        action_model.train()
        epoch_contrib_loss = 0.0
        epoch_action_loss = 0.0
        epoch_contrib_steps = 0
        epoch_action_steps = 0

        for batch_index, batch in enumerate(
            _iter_training_batches(
                learn_cluster_weights_path=learn_cluster_weights_path,
                learn_analysis_csv=learn_analysis_csv,
                learn_rounds_csv=learn_rounds_csv,
                batch_size=batch_size,
            ),
            start=1,
        ):
            if batch["contribution_rows"]:
                contrib_frame = _rows_to_numeric_frame(
                    batch["contribution_rows"],
                    contribution_feature_columns,
                )
                contrib_x = _frame_to_tensor(contrib_frame, resolved_device)
                contrib_targets = torch.tensor(
                    np.asarray(batch["contribution_targets"], dtype=np.float32),
                    dtype=torch.float32,
                    device=resolved_device,
                )
                aon_mask = torch.tensor(
                    np.asarray(batch["contribution_aon"], dtype=np.int64) == 1,
                    dtype=torch.bool,
                    device=resolved_device,
                )
                cont_mask = ~aon_mask
                aon_logits, cont_logits = contrib_model(contrib_x)
                losses: List[torch.Tensor] = []
                if int(aon_mask.sum().item()) > 0:
                    aon_targets = (contrib_targets[aon_mask] >= 0.5).to(dtype=torch.float32)
                    losses.append(aon_loss_fn(aon_logits[aon_mask], aon_targets))
                if int(cont_mask.sum().item()) > 0:
                    cont_pred = torch.sigmoid(cont_logits[cont_mask])
                    losses.append(F.smooth_l1_loss(cont_pred, contrib_targets[cont_mask]))
                if losses:
                    contrib_loss = torch.stack(losses).sum()
                    contrib_optimizer.zero_grad(set_to_none=True)
                    contrib_loss.backward()
                    contrib_optimizer.step()
                    epoch_contrib_loss += float(contrib_loss.detach().cpu().item())
                    epoch_contrib_steps += 1

            if batch["action_rows"]:
                action_frame = _rows_to_numeric_frame(
                    batch["action_rows"],
                    action_feature_columns,
                )
                action_x = _frame_to_tensor(action_frame, resolved_device)
                action_targets = torch.tensor(
                    np.asarray(batch["action_labels"], dtype=np.int64),
                    dtype=torch.long,
                    device=resolved_device,
                )
                action_logits = action_model(action_x)
                action_loss = action_loss_fn(action_logits, action_targets)
                action_optimizer.zero_grad(set_to_none=True)
                action_loss.backward()
                action_optimizer.step()
                epoch_action_loss += float(action_loss.detach().cpu().item())
                epoch_action_steps += 1

            if progress_every > 0 and batch_index % progress_every == 0:
                _print_progress(
                    "gpu-train",
                    (
                        f"epoch={epoch_index}/{epochs} batch={batch_index}/{total_batches} "
                        f"contrib_loss={(epoch_contrib_loss / max(epoch_contrib_steps, 1)):.4f} "
                        f"action_loss={(epoch_action_loss / max(epoch_action_steps, 1)):.4f}"
                    ),
                )
            if max_batches_per_epoch is not None and batch_index >= int(max_batches_per_epoch):
                break

        epoch_row = {
            "epoch": int(epoch_index),
            "mean_contribution_loss": float(epoch_contrib_loss / max(epoch_contrib_steps, 1)),
            "mean_action_loss": float(epoch_action_loss / max(epoch_action_steps, 1)),
            "contribution_steps": int(epoch_contrib_steps),
            "action_steps": int(epoch_action_steps),
            "device": str(resolved_device),
            "hidden_dim": int(hidden_dim),
            "dropout": float(dropout),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "total_batches": int(total_batches),
            "contribution_rows": int(contrib_rows_count),
            "action_rows": int(action_rows_count),
        }
        epoch_rows.append(epoch_row)
        _print_progress(
            "gpu-train",
            (
                f"epoch_done {epoch_index}/{epochs} "
                f"contrib_loss={epoch_row['mean_contribution_loss']:.4f} "
                f"action_loss={epoch_row['mean_action_loss']:.4f}"
            ),
        )

    contrib_model.eval()
    continuous_actual: List[float] = []
    continuous_pred: List[float] = []
    continuous_cluster_ids: List[str] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(
            _iter_training_batches(
                learn_cluster_weights_path=learn_cluster_weights_path,
                learn_analysis_csv=learn_analysis_csv,
                learn_rounds_csv=learn_rounds_csv,
                batch_size=batch_size,
            ),
            start=1,
        ):
            if not batch["contribution_rows"]:
                if max_batches_per_epoch is not None and batch_index >= int(max_batches_per_epoch):
                    break
                continue
            contrib_frame = _rows_to_numeric_frame(
                batch["contribution_rows"],
                contribution_feature_columns,
            )
            contrib_x = _frame_to_tensor(contrib_frame, resolved_device)
            contrib_targets = np.asarray(batch["contribution_targets"], dtype=np.float32)
            cont_mask = np.asarray(batch["contribution_aon"], dtype=np.int64) == 0
            if int(np.sum(cont_mask)) > 0:
                _, cont_logits = contrib_model(contrib_x)
                cont_pred = torch.sigmoid(cont_logits).detach().cpu().numpy()
                continuous_actual.extend(contrib_targets[cont_mask].tolist())
                continuous_pred.extend(cont_pred[cont_mask].tolist())
                continuous_cluster_ids.extend(
                    np.asarray(batch["contribution_cluster_ids"], dtype=object)[cont_mask].tolist()
                )
            if max_batches_per_epoch is not None and batch_index >= int(max_batches_per_epoch):
                break

    continuous_residuals = (
        _build_residual_store(
            actual=pd.Series(continuous_actual, dtype=float),
            predicted=np.asarray(continuous_pred, dtype=float),
            cluster_ids=pd.Series(continuous_cluster_ids, dtype=object),
        )
        if continuous_actual
        else {"global": [0.0]}
    )

    payload = {
        "version": 1,
        "contribution_feature_columns": contribution_feature_columns,
        "action_feature_columns": action_feature_columns,
        "hidden_dim": int(hidden_dim),
        "dropout": float(dropout),
        "contribution_model_state_dict": contrib_model.state_dict(),
        "action_model_state_dict": action_model.state_dict(),
        "continuous_contribution_residuals": continuous_residuals,
        "learn_cluster_weights_path": str(learn_cluster_weights_path),
        "learn_analysis_csv": str(learn_analysis_csv),
        "learn_rounds_csv": str(learn_rounds_csv),
        "class_probabilities": [
            float(action_none_count / max(action_rows_count, 1)),
            float(action_punish_count / max(action_rows_count, 1)),
            float(action_reward_count / max(action_rows_count, 1)),
        ],
    }
    _atomic_torch_save(payload, output_model_path)

    summary_df = pd.DataFrame(epoch_rows)
    summary_df["contribution_feature_count"] = int(len(contribution_feature_columns))
    summary_df["action_feature_count"] = int(len(action_feature_columns))
    summary_df["all_or_nothing_rows"] = int(contrib_aon_count)
    summary_df["continuous_rows"] = int(contrib_cont_count)
    summary_df["mean_target_contribution_rate"] = float(contrib_target_sum / max(contrib_rows_count, 1))
    summary_df["action_none_rate"] = float(action_none_count / max(action_rows_count, 1))
    summary_df["action_punish_rate"] = float(action_punish_count / max(action_rows_count, 1))
    summary_df["action_reward_rate"] = float(action_reward_count / max(action_rows_count, 1))
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_output_path, index=False)

    _print_progress("gpu-train", f"saved model -> {output_model_path}")
    _print_progress("gpu-train", f"saved summary -> {summary_output_path}")
    return payload


class GpuSequenceArchetypePolicyRuntime:
    def __init__(
        self,
        *,
        env_model: DirichletEnvRegressor,
        model_payload: Mapping[str, Any],
        device: torch.device,
    ) -> None:
        self.env_model = env_model
        self.model_payload = dict(model_payload)
        self.device = device
        _configure_torch(self.device)
        self.contribution_model = ContributionHead(
            input_dim=len(self.model_payload["contribution_feature_columns"]),
            hidden_dim=int(self.model_payload["hidden_dim"]),
            dropout=float(self.model_payload["dropout"]),
        ).to(self.device)
        self.contribution_model.load_state_dict(self.model_payload["contribution_model_state_dict"])
        self.contribution_model.eval()
        self.action_model = ActionHead(
            input_dim=len(self.model_payload["action_feature_columns"]),
            hidden_dim=int(self.model_payload["hidden_dim"]),
            dropout=float(self.model_payload["dropout"]),
        ).to(self.device)
        self.action_model.load_state_dict(self.model_payload["action_model_state_dict"])
        self.action_model.eval()

    @classmethod
    def from_config(
        cls,
        config: GpuSequenceArchetypePolicyConfig,
    ) -> "GpuSequenceArchetypePolicyRuntime":
        artifacts_root = Path(config.artifacts_root) if config.artifacts_root else DEFAULT_ARTIFACTS_ROOT
        env_model_path = artifacts_root / "models" / "dirichlet_env_model.pkl"
        model_path = artifacts_root / "models" / "gpu_sequence_policy.pt"
        if not env_model_path.exists():
            raise FileNotFoundError(
                f"Trained env model not found at {env_model_path}. Run the archetype distribution pipeline first."
            )
        rebuild_model = bool(config.rebuild_model) or not model_path.exists()
        if rebuild_model:
            train_gpu_sequence_policy(
                output_model_path=model_path,
                summary_output_path=artifacts_root / "outputs" / "gpu_sequence_policy_train_summary.csv",
                device=config.device,
            )
        model_payload = torch.load(model_path, map_location="cpu", weights_only=False)
        if int(model_payload.get("version", 0) or 0) < 1:
            train_gpu_sequence_policy(
                output_model_path=model_path,
                summary_output_path=artifacts_root / "outputs" / "gpu_sequence_policy_train_summary.csv",
                device=config.device,
            )
            model_payload = torch.load(model_path, map_location="cpu", weights_only=False)
        env_model = DirichletEnvRegressor.load(env_model_path)
        return cls(
            env_model=env_model,
            model_payload=model_payload,
            device=_resolve_device(config.device),
        )

    def predict_cluster_distribution(self, env: Mapping[str, Any]) -> List[float]:
        row = {column: env.get(column) for column in REQUIRED_CONFIG_COLUMNS}
        frame = pd.DataFrame([row])
        predicted = self.env_model.predict(frame)[0]
        predicted = np.clip(np.asarray(predicted, dtype=float), 1e-8, None)
        predicted = predicted / predicted.sum()
        return predicted.tolist()

    def start_game(
        self,
        *,
        env: Mapping[str, Any],
        player_ids: Sequence[str],
        avatar_by_player: Mapping[str, str],
        rng: np.random.Generator | np.random.RandomState | Any,
    ) -> ExactSequenceGameState:
        distribution = self.predict_cluster_distribution(env)
        cluster_ids = np.arange(1, len(distribution) + 1, dtype=int)
        sampled = list(rng.choice(cluster_ids, size=len(player_ids), p=distribution))
        return ExactSequenceGameState(
            env=dict(env),
            player_ids=[str(player_id) for player_id in player_ids],
            avatar_by_player={str(player_id): str(avatar_by_player[player_id]) for player_id in player_ids},
            player_by_avatar={str(avatar): str(player_id) for player_id, avatar in avatar_by_player.items()},
            cluster_by_player={str(player_id): int(cluster_id) for player_id, cluster_id in zip(player_ids, sampled)},
            history_rounds=[],
        )

    def sample_contributions_for_round(
        self,
        *,
        game_state: ExactSequenceGameState,
        round_idx: int,
        rng: np.random.Generator,
    ) -> Dict[str, int]:
        rows = [
            _contribution_feature_row(
                env=game_state.env,
                cluster_id=int(game_state.cluster_by_player[player_id]),
                player_ids=game_state.player_ids,
                focal_player_id=player_id,
                history_rounds=game_state.history_rounds,
                round_idx=int(round_idx),
            )
            for player_id in game_state.player_ids
        ]
        features = _frame_to_tensor(
            _rows_to_numeric_frame(rows, self.model_payload["contribution_feature_columns"]),
            self.device,
        )
        endowment = int(game_state.env.get("CONFIG_endowment", 20) or 20)
        with torch.no_grad():
            aon_logits, cont_logits = self.contribution_model(features)
            aon_probs = torch.sigmoid(aon_logits).detach().cpu().numpy()
            cont_rates = torch.sigmoid(cont_logits).detach().cpu().numpy()
        out: Dict[str, int] = {}
        if as_bool(game_state.env.get("CONFIG_allOrNothing", False)):
            for player_id, prob in zip(game_state.player_ids, aon_probs):
                out[player_id] = endowment if float(rng.random()) < float(prob) else 0
            return out
        for player_id, predicted_rate in zip(game_state.player_ids, cont_rates):
            cluster_id = int(game_state.cluster_by_player[player_id])
            residual = _sample_residual(
                rng,
                self.model_payload.get("continuous_contribution_residuals", {"global": [0.0]}),
                cluster_id,
            )
            sampled_rate = float(max(0.0, min(1.0, float(predicted_rate) + residual)))
            out[player_id] = int(max(0, min(endowment, round(sampled_rate * endowment))))
        return out

    def _predict_action_probabilities(
        self,
        rows: Sequence[Mapping[str, Any]],
    ) -> np.ndarray:
        if not rows:
            return np.zeros((0, len(ACTION_CLASSES)), dtype=float)
        features = _frame_to_tensor(
            _rows_to_numeric_frame(rows, self.model_payload["action_feature_columns"]),
            self.device,
        )
        with torch.no_grad():
            logits = self.action_model(features)
            probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        return np.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)

    def _sample_joint_actions_for_player(
        self,
        *,
        game_state: ExactSequenceGameState,
        player_id: str,
        contributions_by_player: Mapping[str, int],
        round_idx: int,
        rng: np.random.Generator,
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        peer_ids = _relative_peer_ids(game_state.player_ids, player_id)
        if not peer_ids:
            return {}, {}
        base_history_row = _contribution_feature_row(
            env=game_state.env,
            cluster_id=int(game_state.cluster_by_player[player_id]),
            player_ids=game_state.player_ids,
            focal_player_id=player_id,
            history_rounds=game_state.history_rounds,
            round_idx=int(round_idx),
        )
        rows = [
            _action_edge_feature_row(
                env=game_state.env,
                cluster_id=int(game_state.cluster_by_player[player_id]),
                player_ids=game_state.player_ids,
                focal_player_id=player_id,
                target_player_id=target_player_id,
                history_rounds=game_state.history_rounds,
                round_idx=int(round_idx),
                current_contributions_by_player=contributions_by_player,
                base_history_row=base_history_row,
            )
            for target_player_id in peer_ids
        ]
        punish_enabled = as_bool(game_state.env.get("CONFIG_punishmentExists", False))
        reward_enabled = as_bool(game_state.env.get("CONFIG_rewardExists", False))
        probabilities = _masked_action_probabilities(
            self._predict_action_probabilities(rows),
            punish_enabled=punish_enabled,
            reward_enabled=reward_enabled,
        )
        available_coins = _available_action_budget_coins(
            env=game_state.env,
            focal_player_id=player_id,
            contributions_by_player=contributions_by_player,
        )
        punish_cost = float(game_state.env.get("CONFIG_punishmentCost", 1) or 1)
        reward_cost = float(game_state.env.get("CONFIG_rewardCost", 1) or 1)
        sampled_actions: List[Tuple[str, int, float, float]] = []
        total_cost = 0.0
        for row_index, target_player_id in enumerate(peer_ids):
            probs = probabilities[row_index]
            probs = probs / max(float(np.sum(probs)), 1e-12)
            label = int(rng.choice(ACTION_CLASSES, p=probs))
            if label == ACTION_LABEL_NONE:
                continue
            action_cost = punish_cost if label == ACTION_LABEL_PUNISH else reward_cost
            confidence = float(probs[label])
            sampled_actions.append((str(target_player_id), label, action_cost, confidence))
            total_cost += float(action_cost)
        if total_cost > float(available_coins) + 1e-9:
            sampled_actions.sort(key=lambda item: item[3])
            running_cost = float(total_cost)
            kept_actions: List[Tuple[str, int, float, float]] = []
            for action in sampled_actions:
                if running_cost <= float(available_coins) + 1e-9:
                    kept_actions.append(action)
                    continue
                running_cost -= float(action[2])
            if running_cost > float(available_coins) + 1e-9:
                kept_actions = []
                running_cost = 0.0
                for action in sorted(sampled_actions, key=lambda item: item[3], reverse=True):
                    next_cost = running_cost + float(action[2])
                    if next_cost <= float(available_coins) + 1e-9:
                        kept_actions.append(action)
                        running_cost = next_cost
            sampled_actions = kept_actions
        punish_allocations: Dict[str, int] = {}
        reward_allocations: Dict[str, int] = {}
        for target_player_id, label, _, _ in sampled_actions:
            if int(label) == ACTION_LABEL_PUNISH:
                punish_allocations[str(target_player_id)] = 1
            elif int(label) == ACTION_LABEL_REWARD:
                reward_allocations[str(target_player_id)] = 1
        return punish_allocations, reward_allocations

    def sample_actions_for_round(
        self,
        *,
        game_state: ExactSequenceGameState,
        contributions_by_player: Mapping[str, int],
        round_idx: int,
        rng: np.random.Generator,
    ) -> Dict[str, Dict[str, Dict[str, int]]]:
        punish_out: Dict[str, Dict[str, int]] = {}
        reward_out: Dict[str, Dict[str, int]] = {}
        for player_id in game_state.player_ids:
            punish_allocations, reward_allocations = self._sample_joint_actions_for_player(
                game_state=game_state,
                player_id=player_id,
                contributions_by_player=contributions_by_player,
                round_idx=int(round_idx),
                rng=rng,
            )
            punish_out[player_id] = punish_allocations
            reward_out[player_id] = reward_allocations
        return {"punish": punish_out, "reward": reward_out}

    def record_round(
        self,
        *,
        game_state: ExactSequenceGameState,
        contributions_by_player: Mapping[str, int],
        punish_by_player: Mapping[str, Mapping[str, int]],
        reward_by_player: Mapping[str, Mapping[str, int]],
        payoff_by_player: Mapping[str, float],
        round_idx: Optional[int] = None,
    ) -> None:
        if round_idx is None:
            round_idx = int(len(game_state.history_rounds) + 1)
        game_state.history_rounds.append(
            _build_round_state(
                round_idx=int(round_idx),
                contributions_by_player=contributions_by_player,
                punish_by_player=punish_by_player,
                reward_by_player=reward_by_player,
                payoff_by_player=payoff_by_player,
            )
        )

    def record_actual_round(
        self,
        *,
        game_state: ExactSequenceGameState,
        round_rows: pd.DataFrame,
    ) -> None:
        if round_rows is None or round_rows.empty:
            return
        round_idx = int(pd.to_numeric(round_rows["roundIndex"], errors="coerce").dropna().iloc[0])
        contributions_by_player = _contrib_by_player(round_rows)
        punish_by_player = _action_dicts_by_player(round_rows, "data.punished")
        reward_by_player = _action_dicts_by_player(round_rows, "data.rewarded")
        payoff_by_player = _round_payoff_by_player(round_rows)
        self.record_round(
            game_state=game_state,
            contributions_by_player=contributions_by_player,
            punish_by_player=punish_by_player,
            reward_by_player=reward_by_player,
            payoff_by_player=payoff_by_player,
            round_idx=round_idx,
        )
