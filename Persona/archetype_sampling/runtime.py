from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CONFIG_BANK_MODE = "config_bank_archetype"

SUPPORTED_ARCHETYPE_MODES = {
    "matched_summary",
    "random_summary",
    CONFIG_BANK_MODE,
}

DEFAULT_VAL_SUMMARY_POOL = "Persona/archetype_oracle_gpt51_val.jsonl"
DEFAULT_LEARN_SUMMARY_POOL = "Persona/archetype_oracle_gpt51_learn.jsonl"
DEFAULT_LEARN_PLAYER_TABLE = (
    "simulation_statistical/archetype_distribution_embedding/artifacts/intermediate/"
    "player_game_table_learn_clean.parquet"
)
DEFAULT_LEARN_EMBEDDINGS = (
    "simulation_statistical/archetype_distribution_embedding/artifacts/intermediate/"
    "embedding_matrix_learn.parquet"
)
DEFAULT_VAL_PLAYER_TABLE = (
    "simulation_statistical/archetype_distribution_embedding/artifacts/intermediate/"
    "player_game_table_val_clean.parquet"
)

SOFT_BANK_BOOL_COLUMNS = [
    "CONFIG_showNRounds",
    "CONFIG_allOrNothing",
    "CONFIG_chat",
    "CONFIG_punishmentExists",
    "CONFIG_rewardExists",
    "CONFIG_showOtherSummaries",
    "CONFIG_showPunishmentId",
    "CONFIG_showRewardId",
]

SOFT_BANK_CATEGORICAL_COLUMNS = [
    "CONFIG_punishmentTech",
    "CONFIG_rewardTech",
]

SOFT_BANK_NUMERIC_COLUMNS = [
    "CONFIG_playerCount",
    "CONFIG_numRounds",
    "CONFIG_defaultContribProp",
    "CONFIG_endowment",
    "CONFIG_multiplier",
    "CONFIG_MPCR",
    "CONFIG_punishmentCost",
    "CONFIG_punishmentMagnitude",
    "CONFIG_rewardCost",
    "CONFIG_rewardMagnitude",
]

SOFT_BANK_FEATURE_COLUMNS = (
    SOFT_BANK_BOOL_COLUMNS
    + SOFT_BANK_CATEGORICAL_COLUMNS
    + SOFT_BANK_NUMERIC_COLUMNS
)


@dataclass
class ArchetypeSummaryPool:
    all_records: List[Dict[str, Any]]
    by_game_player: Dict[str, Dict[str, Dict[str, Any]]]


@dataclass
class AssignmentBatch:
    assignments_by_player: Dict[str, Dict[str, Any]]
    manifest_rows: List[Dict[str, Any]]
    summary: Dict[str, Any]


@dataclass
class PrecomputedAssignmentIndex:
    source_path: str
    assignments_by_game_player: Dict[str, Dict[str, Dict[str, Any]]]
    manifest_rows_by_game_player: Dict[str, Dict[str, Dict[str, Any]]]
    modes_by_game: Dict[str, str]


def canonicalize_archetype_mode(mode: Any) -> str:
    return str(mode or "").strip()


def default_summary_pool_for_mode(mode: str) -> str:
    mode_name = canonicalize_archetype_mode(mode)
    if mode_name == CONFIG_BANK_MODE:
        return DEFAULT_LEARN_SUMMARY_POOL
    return DEFAULT_VAL_SUMMARY_POOL


def load_finished_summary_pool(path: str) -> ArchetypeSummaryPool:
    if not path:
        raise ValueError("archetype_summary_pool path must be set when archetype mode is enabled")
    pool_path = Path(path)
    if not pool_path.exists():
        raise FileNotFoundError(f"archetype summary pool file not found: {path}")

    all_records: List[Dict[str, Any]] = []
    by_game_player: Dict[str, Dict[str, Dict[str, Any]]] = {}
    with pool_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if rec.get("game_finished") is not True:
                continue
            text = rec.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            participant = str(rec.get("participant") or "").strip()
            experiment = str(rec.get("experiment") or "").strip()
            entry = {
                "participant": participant,
                "experiment": experiment,
                "text": text.strip(),
            }
            all_records.append(entry)
            if participant and experiment:
                by_game_player.setdefault(experiment, {}).setdefault(participant, entry)
    if not all_records:
        raise ValueError(f"No finished-game archetype summary records found in {path}")
    return ArchetypeSummaryPool(all_records=all_records, by_game_player=by_game_player)


def load_precomputed_assignment_index(path: str) -> PrecomputedAssignmentIndex:
    if not path:
        raise ValueError("precomputed assignment manifest path must be set when using fixed assignments")
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"precomputed assignment manifest file not found: {path}")

    assignments_by_game_player: Dict[str, Dict[str, Dict[str, Any]]] = {}
    manifest_rows_by_game_player: Dict[str, Dict[str, Dict[str, Any]]] = {}
    modes_by_game: Dict[str, str] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            game_id = str(rec.get("target_gameId") or "").strip()
            player_id = str(rec.get("target_playerId") or "").strip()
            if not game_id or not player_id:
                continue
            mode_name = canonicalize_archetype_mode(rec.get("archetype_mode"))
            if game_id not in modes_by_game:
                modes_by_game[game_id] = mode_name
            elif mode_name and modes_by_game[game_id] and mode_name != modes_by_game[game_id]:
                raise ValueError(
                    f"Inconsistent archetype modes for target_gameId={game_id} in {path}: "
                    f"{modes_by_game[game_id]} vs {mode_name}"
                )
            assignment_record = {
                "participant": str(rec.get("source_playerId") or ""),
                "experiment": str(rec.get("source_gameId") or ""),
                "text": str(rec.get("source_text") or ""),
                "source_score": rec.get("source_score"),
                "source_weight": rec.get("source_weight"),
                "source_rank": rec.get("source_rank"),
                "bank_size": rec.get("bank_size"),
                "temperature": rec.get("temperature"),
            }
            assignments_by_game_player.setdefault(game_id, {})[player_id] = assignment_record
            copied_row = dict(rec)
            copied_row["archetype_mode"] = mode_name or str(rec.get("archetype_mode") or "")
            manifest_rows_by_game_player.setdefault(game_id, {})[player_id] = copied_row
    if not assignments_by_game_player:
        raise ValueError(f"No valid target assignment rows found in {path}")
    return PrecomputedAssignmentIndex(
        source_path=str(manifest_path),
        assignments_by_game_player=assignments_by_game_player,
        manifest_rows_by_game_player=manifest_rows_by_game_player,
        modes_by_game=modes_by_game,
    )


def assignment_batch_from_precomputed(
    *,
    index: PrecomputedAssignmentIndex,
    game_id: str,
    player_ids: Sequence[str],
    requested_mode: Optional[str] = None,
) -> AssignmentBatch:
    game_key = str(game_id)
    assignments_by_player = index.assignments_by_game_player.get(game_key)
    manifest_rows_by_player = index.manifest_rows_by_game_player.get(game_key)
    if not assignments_by_player or not manifest_rows_by_player:
        raise KeyError(f"No precomputed assignments found for target_gameId={game_key} in {index.source_path}")

    requested_mode_name = canonicalize_archetype_mode(requested_mode)
    stored_mode = canonicalize_archetype_mode(index.modes_by_game.get(game_key))
    if requested_mode_name and stored_mode and requested_mode_name != stored_mode:
        raise ValueError(
            f"Precomputed assignments for target_gameId={game_key} were generated with mode "
            f"{stored_mode}, but the run requested {requested_mode_name}."
        )

    missing = [str(pid) for pid in player_ids if str(pid) not in assignments_by_player]
    if missing:
        sample_missing = ", ".join(missing[:5])
        raise KeyError(
            f"Precomputed assignments for target_gameId={game_key} are missing playerIds "
            f"{sample_missing} (total_missing={len(missing)}) in {index.source_path}"
        )

    ordered_assignments = {
        str(pid): dict(assignments_by_player[str(pid)])
        for pid in player_ids
    }
    ordered_manifest_rows = [
        dict(manifest_rows_by_player[str(pid)])
        for pid in player_ids
    ]
    summary = {
        "mode": stored_mode or requested_mode_name or "",
        "target_gameId": game_key,
        "source": "precomputed_assignment_manifest",
        "source_path": index.source_path,
        "assigned_players": int(len(ordered_assignments)),
    }
    return AssignmentBatch(
        assignments_by_player=ordered_assignments,
        manifest_rows=ordered_manifest_rows,
        summary=summary,
    )


def _parse_boolish(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if pd.isna(value):
            return None
        if float(value) == 1.0:
            return True
        if float(value) == 0.0:
            return False
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _stable_random(seed: Any, game_id: str, mode: str) -> random.Random:
    return random.Random(f"{seed}|{game_id}|{mode}")


def _coerce_feature_frame(
    frame: pd.DataFrame,
    *,
    numeric_columns: Sequence[str] = SOFT_BANK_NUMERIC_COLUMNS,
    bool_columns: Sequence[str] = SOFT_BANK_BOOL_COLUMNS,
    categorical_columns: Sequence[str] = SOFT_BANK_CATEGORICAL_COLUMNS,
) -> pd.DataFrame:
    out = frame.copy()
    for col in numeric_columns:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in bool_columns:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = out[col].map(_parse_boolish).astype("float64")
    for col in categorical_columns:
        if col not in out.columns:
            out[col] = np.nan
        values = out[col]
        mask = values.isna()
        out[col] = values.astype(str)
        out.loc[mask, col] = np.nan
    selected = list(numeric_columns) + list(bool_columns) + list(categorical_columns)
    return out[selected]


def _weighted_sample_without_replacement(
    indices: Sequence[int],
    weights: Sequence[float],
    k: int,
    rng: random.Random,
) -> List[int]:
    remaining_indices = list(indices)
    remaining_weights = [max(0.0, float(weight)) for weight in weights]
    chosen: List[int] = []
    draws = min(int(k), len(remaining_indices))
    for _ in range(draws):
        total = sum(remaining_weights)
        if total <= 0.0:
            break
        threshold = rng.random() * total
        cumulative = 0.0
        picked_pos = len(remaining_weights) - 1
        for pos, weight in enumerate(remaining_weights):
            cumulative += weight
            if cumulative >= threshold:
                picked_pos = pos
                break
        chosen.append(remaining_indices.pop(picked_pos))
        remaining_weights.pop(picked_pos)
    return chosen


def _build_assignment_manifest_rows(
    *,
    mode: str,
    target_game_id: str,
    target_player_ids: Sequence[str],
    target_env: Mapping[str, Any],
    assignments_by_player: Mapping[str, Mapping[str, Any]],
    summary_pool_path: str,
    assignment_seed: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    treatment_name = str(target_env.get("CONFIG_treatmentName", "") or "")
    for slot_idx, player_id in enumerate(target_player_ids, start=1):
        record = dict(assignments_by_player.get(player_id) or {})
        rows.append(
            {
                "target_gameId": target_game_id,
                "target_treatmentName": treatment_name,
                "target_playerId": str(player_id),
                "target_player_slot": int(slot_idx),
                "archetype_mode": mode,
                "summary_pool_path": summary_pool_path,
                "assignment_seed": int(assignment_seed),
                "source_gameId": str(record.get("experiment") or ""),
                "source_playerId": str(record.get("participant") or ""),
                "source_text": str(record.get("text") or ""),
                "source_score": record.get("source_score"),
                "source_weight": record.get("source_weight"),
                "source_rank": record.get("source_rank"),
                "bank_size": record.get("bank_size"),
                "temperature": record.get("temperature"),
            }
        )
    return rows


class SoftBankSummarySampler:
    """CONFIG-conditioned bank sampling over finished learn-wave archetype summaries."""

    def __init__(
        self,
        *,
        summary_pool_path: str,
        learn_player_table_path: str = DEFAULT_LEARN_PLAYER_TABLE,
        learn_embedding_path: str = DEFAULT_LEARN_EMBEDDINGS,
        temperature: float = 0.07,
    ) -> None:
        if float(temperature) <= 0.0:
            raise ValueError(f"temperature must be > 0 for {CONFIG_BANK_MODE}")
        self.summary_pool_path = str(summary_pool_path)
        self.learn_player_table_path = str(learn_player_table_path)
        self.learn_embedding_path = str(learn_embedding_path)
        self.temperature = float(temperature)

        self.summary_pool = load_finished_summary_pool(self.summary_pool_path)
        self.bank_df = self._build_bank_frame()
        self.embed_columns = [col for col in self.bank_df.columns if col.startswith("embed_")]
        if not self.embed_columns:
            raise ValueError(
                f"No embedding columns found in {self.learn_embedding_path}; expected columns starting with 'embed_'."
            )
        self.numeric_columns = [
            col for col in SOFT_BANK_NUMERIC_COLUMNS if col in self.bank_df.columns and self.bank_df[col].notna().any()
        ]
        self.bool_columns = [
            col for col in SOFT_BANK_BOOL_COLUMNS if col in self.bank_df.columns and self.bank_df[col].notna().any()
        ]
        self.categorical_columns = [
            col for col in SOFT_BANK_CATEGORICAL_COLUMNS if col in self.bank_df.columns and self.bank_df[col].notna().any()
        ]
        self.feature_frame = _coerce_feature_frame(
            self.bank_df,
            numeric_columns=self.numeric_columns,
            bool_columns=self.bool_columns,
            categorical_columns=self.categorical_columns,
        )
        self.embedding_matrix = self.bank_df[self.embed_columns].to_numpy(dtype=float)
        self.embedding_matrix_normalized = _normalize_rows(self.embedding_matrix)
        self.model = self._fit_model()

    def _build_bank_frame(self) -> pd.DataFrame:
        summary_text_by_pair = {
            (str(rec["experiment"]), str(rec["participant"])): str(rec["text"])
            for rec in self.summary_pool.all_records
        }

        player_df = pd.read_parquet(self.learn_player_table_path).copy()
        if "game_finished" in player_df.columns:
            player_df = player_df[player_df["game_finished"] == True].copy()  # noqa: E712
        player_df["game_id"] = player_df["game_id"].astype(str)
        player_df["player_id"] = player_df["player_id"].astype(str)
        player_df["pair_key"] = list(zip(player_df["game_id"], player_df["player_id"]))
        player_df = player_df[player_df["pair_key"].isin(summary_text_by_pair)].copy()

        embedding_df = pd.read_parquet(self.learn_embedding_path).copy()
        embedding_df["game_id"] = embedding_df["game_id"].astype(str)
        embedding_df["player_id"] = embedding_df["player_id"].astype(str)

        merged = player_df.merge(
            embedding_df,
            on=["wave", "game_id", "player_id"],
            how="inner",
            validate="one_to_one",
        )
        if merged.empty:
            raise ValueError(
                "Soft-bank training bank is empty after joining summaries to learn embeddings. "
                "Use a learn-wave finished-summary pool such as Persona/archetype_oracle_gpt51_learn.jsonl."
            )
        merged["text"] = merged["pair_key"].map(summary_text_by_pair)
        merged = merged[merged["text"].map(lambda value: isinstance(value, str) and bool(value.strip()))].copy()
        merged["participant"] = merged["player_id"]
        merged["experiment"] = merged["game_id"]
        merged = merged.sort_values(["game_id", "player_id"], kind="stable").reset_index(drop=True)
        return merged

    def _fit_model(self) -> Pipeline:
        transformers = []
        if self.numeric_columns:
            transformers.append(
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("impute", SimpleImputer(strategy="median")),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    self.numeric_columns,
                )
            )
        if self.bool_columns:
            transformers.append(
                (
                    "bool",
                    Pipeline(
                        steps=[
                            ("impute", SimpleImputer(strategy="most_frequent")),
                        ]
                    ),
                    self.bool_columns,
                )
            )
        if self.categorical_columns:
            transformers.append(
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    self.categorical_columns,
                )
            )
        if not transformers:
            raise ValueError("Soft-bank sampler found no usable CONFIG features in the learn-wave bank.")
        preprocessor = ColumnTransformer(transformers=transformers)
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", MultiOutputRegressor(Ridge(alpha=3.0))),
            ]
        )
        model.fit(self.feature_frame, self.embedding_matrix)
        return model

    def score_bank(self, env: Mapping[str, Any]) -> pd.DataFrame:
        env_frame = _coerce_feature_frame(
            pd.DataFrame([{col: env.get(col) for col in SOFT_BANK_FEATURE_COLUMNS}]),
            numeric_columns=self.numeric_columns,
            bool_columns=self.bool_columns,
            categorical_columns=self.categorical_columns,
        )
        predicted = self.model.predict(env_frame)
        predicted = np.asarray(predicted, dtype=float)
        if predicted.ndim == 1:
            predicted = predicted.reshape(1, -1)
        predicted_normalized = _normalize_rows(predicted)
        scores = (predicted_normalized @ self.embedding_matrix_normalized.T).reshape(-1)
        scaled = (scores - float(scores.max())) / self.temperature
        weights = np.exp(scaled)
        weights = weights / weights.sum()

        ranked = self.bank_df[["experiment", "participant", "text"]].copy()
        ranked["source_score"] = scores
        ranked["source_weight"] = weights
        ranked = ranked.sort_values(
            ["source_weight", "source_score", "experiment", "participant"],
            ascending=[False, False, True, True],
            kind="stable",
        ).reset_index(drop=True)
        ranked["source_rank"] = np.arange(1, len(ranked) + 1)
        ranked["bank_size"] = int(len(ranked))
        ranked["temperature"] = float(self.temperature)
        return ranked

    def assign_for_game(
        self,
        *,
        game_id: str,
        player_ids: Sequence[str],
        env: Mapping[str, Any],
        seed: int,
    ) -> AssignmentBatch:
        ranked = self.score_bank(env)
        rng = _stable_random(seed, game_id, CONFIG_BANK_MODE)
        chosen_ranks = _weighted_sample_without_replacement(
            indices=ranked.index.tolist(),
            weights=ranked["source_weight"].tolist(),
            k=len(player_ids),
            rng=rng,
        )

        assignments: Dict[str, Dict[str, Any]] = {}
        chosen_records = ranked.loc[chosen_ranks].reset_index(drop=True)
        for player_id, (_, rec) in zip(player_ids, chosen_records.iterrows()):
            assignments[str(player_id)] = {
                "participant": str(rec["participant"]),
                "experiment": str(rec["experiment"]),
                "text": str(rec["text"]),
                "source_score": float(rec["source_score"]),
                "source_weight": float(rec["source_weight"]),
                "source_rank": int(rec["source_rank"]),
                "bank_size": int(rec["bank_size"]),
                "temperature": float(rec["temperature"]),
            }

        manifest_rows = _build_assignment_manifest_rows(
            mode=CONFIG_BANK_MODE,
            target_game_id=game_id,
            target_player_ids=player_ids,
            target_env=env,
            assignments_by_player=assignments,
            summary_pool_path=self.summary_pool_path,
            assignment_seed=seed,
        )
        summary = {
            "mode": CONFIG_BANK_MODE,
            "target_gameId": str(game_id),
            "bank_size": int(len(ranked)),
            "temperature": float(self.temperature),
            "effective_support": float(1.0 / np.square(ranked["source_weight"]).sum()),
            "max_weight": float(ranked["source_weight"].max()),
        }
        return AssignmentBatch(
            assignments_by_player=assignments,
            manifest_rows=manifest_rows,
            summary=summary,
        )


def assign_archetypes_for_game(
    *,
    mode: str,
    game_id: str,
    player_ids: Sequence[str],
    env: Mapping[str, Any],
    seed: int,
    summary_pool: Optional[ArchetypeSummaryPool],
    summary_pool_path: str,
    soft_bank_sampler: Optional[SoftBankSummarySampler] = None,
    precomputed_assignment_index: Optional[PrecomputedAssignmentIndex] = None,
    log_fn: Optional[Callable[..., None]] = None,
) -> AssignmentBatch:
    mode_name = canonicalize_archetype_mode(mode)
    if mode_name not in SUPPORTED_ARCHETYPE_MODES:
        supported = ", ".join(sorted(SUPPORTED_ARCHETYPE_MODES))
        raise ValueError(f"Unsupported archetype mode '{mode_name}'. Allowed values: {supported}.")

    if precomputed_assignment_index is not None:
        return assignment_batch_from_precomputed(
            index=precomputed_assignment_index,
            game_id=str(game_id),
            player_ids=[str(pid) for pid in player_ids],
            requested_mode=mode_name,
        )

    if summary_pool is None:
        raise ValueError("summary_pool must be provided when not using precomputed assignments")

    if mode_name == "matched_summary":
        game_map = summary_pool.by_game_player.get(str(game_id), {})
        missing = [str(pid) for pid in player_ids if str(pid) not in game_map]
        if missing and log_fn is not None:
            sample_missing = ", ".join(missing[:5])
            log_fn(
                f"[warn] matched_summary missing archetypes for gameId={game_id}, "
                f"missing_playerIds={sample_missing}, total_missing={len(missing)}"
            )
        assignments = {str(pid): dict(game_map[str(pid)]) for pid in player_ids if str(pid) in game_map}
        manifest_rows = _build_assignment_manifest_rows(
            mode=mode_name,
            target_game_id=game_id,
            target_player_ids=player_ids,
            target_env=env,
            assignments_by_player=assignments,
            summary_pool_path=summary_pool_path,
            assignment_seed=seed,
        )
        summary = {
            "mode": mode_name,
            "target_gameId": str(game_id),
            "assigned_players": int(len(assignments)),
            "missing_players": int(len(missing)),
        }
        return AssignmentBatch(assignments_by_player=assignments, manifest_rows=manifest_rows, summary=summary)

    if mode_name == "random_summary":
        rng = _stable_random(seed, game_id, "random_summary")
        records = summary_pool.all_records
        assignments = {
            str(pid): dict(records[rng.randrange(len(records))])
            for pid in player_ids
        }
        manifest_rows = _build_assignment_manifest_rows(
            mode=mode_name,
            target_game_id=game_id,
            target_player_ids=player_ids,
            target_env=env,
            assignments_by_player=assignments,
            summary_pool_path=summary_pool_path,
            assignment_seed=seed,
        )
        summary = {
            "mode": mode_name,
            "target_gameId": str(game_id),
            "assigned_players": int(len(assignments)),
            "pool_size": int(len(records)),
        }
        return AssignmentBatch(assignments_by_player=assignments, manifest_rows=manifest_rows, summary=summary)

    if soft_bank_sampler is None:
        raise ValueError(f"{CONFIG_BANK_MODE} requires a SoftBankSummarySampler")
    return soft_bank_sampler.assign_for_game(
        game_id=str(game_id),
        player_ids=[str(pid) for pid in player_ids],
        env=env,
        seed=int(seed),
    )


def build_validation_treatment_contexts(val_player_table_path: str = DEFAULT_VAL_PLAYER_TABLE) -> pd.DataFrame:
    val_df = pd.read_parquet(val_player_table_path).copy()
    if "CONFIG_treatmentName" not in val_df.columns:
        raise ValueError(f"CONFIG_treatmentName missing from {val_player_table_path}")
    group_columns = ["CONFIG_treatmentName", "game_id", *SOFT_BANK_FEATURE_COLUMNS]
    available_columns = [col for col in group_columns if col in val_df.columns]
    grouped = (
        val_df[available_columns]
        .sort_values(["CONFIG_treatmentName", "game_id"], kind="stable")
        .groupby("CONFIG_treatmentName", as_index=False)
        .first()
    )
    grouped = grouped.rename(columns={"game_id": "reference_game_id"})
    return grouped
