from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from common import wasserstein_distance_1d
except ModuleNotFoundError:  # pragma: no cover - package import fallback
    from .common import wasserstein_distance_1d


GAME_ORDER = ["Sender", "Receiver", "PD", "SH", "C", "Proposer", "Responder"]
ROLE_SLICE_ORDER = ["both", "no", "yes"]
CASE_GROUP_ORDER = ["Benchmark", "AIR", "TD", "OD"]

GAME_TO_FIELD = {
    "C": "C_decision",
    "Sender": "TGSender_decision",
    "Receiver": "TGReceiver_decision",
    "Proposer": "UGProposer_decision",
    "Responder": "UGResponder_decision",
    "PD": "PD_decision",
    "SH": "SH_decision",
}

GAME_TO_DELEGATION_FIELD = {
    "C": "C_delegated",
    "Sender": "TGSender_delegated",
    "Receiver": "TGReceiver_delegated",
    "Proposer": "UGProposer_delegated",
    "Responder": "UGResponder_delegated",
    "PD": "PD_delegated",
    "SH": "SH_delegated",
}

PLANET_TO_CODE = {
    "Mercury": -2,
    "Venus": -1,
    "Earth": 0,
    "Mars": 1,
    "Saturn": 2,
}

PAYOFF_METRIC_FAMILY = "payoff_distribution_distance"
PAYOFF_METRIC_ORDER = {
    "sender_expected_payoff": 0,
    "receiver_expected_payoff": 1,
    "pd_expected_payoff": 2,
    "sh_expected_payoff": 3,
    "c_expected_payoff": 4,
    "proposer_expected_payoff": 5,
    "responder_expected_payoff": 6,
    "mean_expected_payoff": 7,
}


def _stable_hash_int(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _encode_game_value(game: str, value: Any) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if game == "C":
        return float(PLANET_TO_CODE.get(str(value), np.nan))
    if game == "Sender":
        text = str(value).strip().upper()
        if text == "YES":
            return 1.0
        if text == "NO":
            return 0.0
        return None
    if game == "Receiver":
        return float(pd.to_numeric(value, errors="coerce"))
    if game == "Proposer":
        return float(pd.to_numeric(value, errors="coerce"))
    if game == "Responder":
        return float(pd.to_numeric(value, errors="coerce"))
    if game == "PD":
        text = str(value).strip().upper()
        if text == "A":
            return 1.0
        if text == "B":
            return 0.0
        return None
    if game == "SH":
        text = str(value).strip().upper()
        if text == "X":
            return 1.0
        if text == "Y":
            return 0.0
        return None
    raise ValueError(f"Unsupported game: {game}")


def _load_subject_ai_flavour_map(repo_root: Path) -> dict[str, int]:
    raw_path = (
        repo_root
        / "non-PGG_generalization"
        / "data"
        / "multi_game_llm_fvk2c"
        / "Package"
        / "data"
        / "MainDataRawClean.csv"
    )
    columns = ["SubjectID", *[f"Personalization_{index}" for index in range(1, 8)]]
    raw = pd.read_csv(raw_path, usecols=columns)

    def first_valid(series: pd.Series) -> float:
        clean = pd.to_numeric(series, errors="coerce").dropna()
        return float(clean.iloc[0]) if not clean.empty else float("nan")

    grouped = raw.groupby("SubjectID", as_index=False).agg(
        {column: first_valid for column in columns if column != "SubjectID"}
    )
    flavour_map: dict[str, int] = {}
    for row in grouped.to_dict(orient="records"):
        bits: list[int] = []
        valid = True
        for index in range(1, 8):
            value = row.get(f"Personalization_{index}")
            if pd.isna(value):
                valid = False
                break
            bits.append(int(value) - 1)
        if not valid:
            continue
        ai_flavour = 64 * bits[0] + 32 * bits[1] + 16 * bits[2] + 8 * bits[3] + 4 * bits[4] + 2 * bits[5] + bits[6] + 1
        flavour_map[str(int(row["SubjectID"]))] = int(ai_flavour)
    return flavour_map


def _load_ai_library(repo_root: Path) -> tuple[pd.DataFrame, list[int]]:
    ai_path = (
        repo_root
        / "non-PGG_generalization"
        / "data"
        / "multi_game_llm_fvk2c"
        / "Package"
        / "data"
        / "AIData.csv"
    )
    ai = pd.read_csv(ai_path)
    ai = ai.reset_index(drop=True).copy()
    ai["ai_index"] = np.arange(1, len(ai) + 1)
    ai["Type"] = ai["Type"].astype(str).str.lower()

    ai["C"] = (
        ai["DEC.C"]
        .astype(str)
        .str.replace(r"\.[0-9]*$", "", regex=True)
        .str.strip()
        .str.capitalize()
        .map(PLANET_TO_CODE)
        .astype(float)
    )
    trust = ai["DEC.TG.trustor"].astype(str).str.replace(r"\.[0-9]*$", "", regex=True).str.strip().str.upper()
    ai["Sender"] = trust.map({"YES": 1.0, "NO": 0.0}).astype(float)
    ai["Receiver"] = pd.to_numeric(ai["DEC.TG.trustee"], errors="coerce").astype(float)
    ai["Proposer"] = pd.to_numeric(ai["DEC.UG.proposer"], errors="coerce").astype(float)
    ai["Responder"] = pd.to_numeric(ai["DEC.UG.responder"], errors="coerce").astype(float)
    ai["PD"] = ai["DEC.PD"].astype(str).str.strip().str.upper().map({"A": 1.0, "B": 0.0}).astype(float)
    ai["SH"] = ai["DEC.SH"].astype(str).str.strip().str.upper().map({"A": 1.0, "B": 0.0}).astype(float)

    library = ai.set_index("ai_index")[["Type", *GAME_ORDER]].copy()
    unpersonalized_indices = library.index[library["Type"] == "unpersonalized"].tolist()
    return library, unpersonalized_indices


def _prepare_session_context(
    *,
    sessions: pd.DataFrame,
    scenarios: pd.DataFrame,
    ai_flavour_map: dict[str, int],
) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str, str], dict[str, Any]]]:
    sessions_copy = sessions.copy()
    sessions_copy["record_id"] = sessions_copy["record_id"].astype(str)
    sessions_copy["ai_flavour"] = sessions_copy["record_id"].map(ai_flavour_map)
    session_lookup = {
        str(row["record_id"]): row
        for row in sessions_copy.to_dict(orient="records")
    }

    scenarios_copy = scenarios.copy()
    scenarios_copy["record_id"] = scenarios_copy["record_id"].astype(str)
    scenario_lookup = {
        (str(row["record_id"]), str(row["scenario"]), str(row["case"])): row
        for row in scenarios_copy.to_dict(orient="records")
    }
    return session_lookup, scenario_lookup


def _human_decision(
    *,
    scenario_lookup: dict[tuple[str, str, str], dict[str, Any]],
    subject_id: str,
    scenario: str,
    case: str,
    game: str,
) -> float | None:
    row = scenario_lookup.get((subject_id, scenario, case))
    if row is None:
        return None
    return _encode_game_value(game, row.get(GAME_TO_FIELD[game]))


def _ai_decision(
    *,
    ai_library: pd.DataFrame,
    unpersonalized_indices: list[int],
    personalized: bool,
    subject_id: str,
    ai_flavour: Any,
    game: str,
    seed: int,
    context_parts: list[str],
) -> float | None:
    if personalized and pd.notna(ai_flavour):
        ai_index = int(ai_flavour)
    else:
        key = "|".join([str(seed), subject_id, game, *context_parts])
        ai_index = unpersonalized_indices[_stable_hash_int(key) % len(unpersonalized_indices)]
    value = ai_library.loc[ai_index, game]
    if pd.isna(value):
        return None
    return float(value)


def _payoff_for_game(game: str, focal_value: float | None, other_value: float | None) -> float | None:
    if focal_value is None or other_value is None or np.isnan(focal_value) or np.isnan(other_value):
        return None
    if game == "C":
        return float(2 + 3 * int(focal_value == other_value))
    if game == "Sender":
        return float(5 + other_value) if int(focal_value) == 1 else 5.0
    if game == "Receiver":
        return float(11 - focal_value) if int(other_value) == 1 else 5.0
    if game == "Proposer":
        return float(int(other_value <= focal_value) * (10 - focal_value))
    if game == "Responder":
        return float(int(focal_value <= other_value) * (10 - other_value))
    if game == "PD":
        return float([3, 8, 1, 5][int(1 + 2 * focal_value + other_value - 1)])
    if game == "SH":
        return float([4, 5, 1, 8][int(1 + 2 * focal_value + other_value - 1)])
    raise ValueError(f"Unsupported game: {game}")


def _random_role_payoff(
    *,
    game: str,
    comparison: str,
    focal_role: str,
    treatment_code: str,
    focal_id: str,
    other_id: str,
    focal_session: dict[str, Any],
    other_session: dict[str, Any],
    focal_scenarios: dict[tuple[str, str, str], dict[str, Any]],
    other_scenarios: dict[tuple[str, str, str], dict[str, Any]],
    ai_library: pd.DataFrame,
    unpersonalized_indices: list[int],
    personalized: bool,
    seed: int,
) -> float | None:
    if focal_role == "NoAISupport":
        focal_value = _human_decision(
            scenario_lookup=focal_scenarios,
            subject_id=focal_id,
            scenario="NoAISupport",
            case=comparison,
            game=game,
        )
        if comparison == "AgainstAI":
            other_value = _ai_decision(
                ai_library=ai_library,
                unpersonalized_indices=unpersonalized_indices,
                personalized=personalized,
                subject_id=other_id,
                ai_flavour=other_session.get("ai_flavour"),
                game=game,
                seed=seed,
                context_parts=[treatment_code, focal_role, comparison, focal_id, other_id, "counterpart_ai"],
            )
        else:
            other_value = _human_decision(
                scenario_lookup=other_scenarios,
                subject_id=other_id,
                scenario="AISupport",
                case="AgainstHuman",
                game=game,
            )
    else:
        other_value = _human_decision(
            scenario_lookup=other_scenarios,
            subject_id=other_id,
            scenario="NoAISupport",
            case=comparison,
            game=game,
        )
        if comparison == "AgainstAI":
            focal_value = _ai_decision(
                ai_library=ai_library,
                unpersonalized_indices=unpersonalized_indices,
                personalized=personalized,
                subject_id=focal_id,
                ai_flavour=focal_session.get("ai_flavour"),
                game=game,
                seed=seed,
                context_parts=[treatment_code, focal_role, comparison, focal_id, other_id, "focal_ai"],
            )
        else:
            focal_value = _human_decision(
                scenario_lookup=focal_scenarios,
                subject_id=focal_id,
                scenario="AISupport",
                case="AgainstHuman",
                game=game,
            )
    return _payoff_for_game(game, focal_value, other_value)


def _delegation_role_payoff(
    *,
    game: str,
    transparency: str,
    treatment_code: str,
    focal_role: str,
    focal_id: str,
    other_id: str,
    focal_session: dict[str, Any],
    other_session: dict[str, Any],
    focal_scenarios: dict[tuple[str, str, str], dict[str, Any]],
    other_scenarios: dict[tuple[str, str, str], dict[str, Any]],
    ai_library: pd.DataFrame,
    unpersonalized_indices: list[int],
    personalized: bool,
    seed: int,
) -> float | None:
    delegation_field = GAME_TO_DELEGATION_FIELD[game]
    if focal_role == "NoAISupport":
        other_delegated = int(other_session.get(delegation_field) or 0)
        focal_case = "AgainstHuman" if transparency == "T" and other_delegated == 0 else "AgainstAI" if transparency == "T" else "Opaque"
        focal_value = _human_decision(
            scenario_lookup=focal_scenarios,
            subject_id=focal_id,
            scenario="NoAISupport",
            case=focal_case,
            game=game,
        )
        if other_delegated == 1:
            other_value = _ai_decision(
                ai_library=ai_library,
                unpersonalized_indices=unpersonalized_indices,
                personalized=personalized,
                subject_id=other_id,
                ai_flavour=other_session.get("ai_flavour"),
                game=game,
                seed=seed,
                context_parts=[treatment_code, focal_role, focal_case, focal_id, other_id, "counterpart_ai"],
            )
        else:
            other_value = _human_decision(
                scenario_lookup=other_scenarios,
                subject_id=other_id,
                scenario="AISupport",
                case="AgainstHuman",
                game=game,
            )
    else:
        focal_delegated = int(focal_session.get(delegation_field) or 0)
        if focal_delegated == 1:
            focal_value = _ai_decision(
                ai_library=ai_library,
                unpersonalized_indices=unpersonalized_indices,
                personalized=personalized,
                subject_id=focal_id,
                ai_flavour=focal_session.get("ai_flavour"),
                game=game,
                seed=seed,
                context_parts=[treatment_code, focal_role, "Delegation", focal_id, other_id, "focal_ai"],
            )
            other_case = "AgainstAI" if transparency == "T" else "Opaque"
        else:
            focal_value = _human_decision(
                scenario_lookup=focal_scenarios,
                subject_id=focal_id,
                scenario="AISupport",
                case="AgainstHuman",
                game=game,
            )
            other_case = "AgainstHuman" if transparency == "T" else "Opaque"
        other_value = _human_decision(
            scenario_lookup=other_scenarios,
            subject_id=other_id,
            scenario="NoAISupport",
            case=other_case,
            game=game,
        )
    return _payoff_for_game(game, focal_value, other_value)


def simulate_expected_payoffs(
    *,
    focal_sessions: pd.DataFrame,
    focal_scenarios: pd.DataFrame,
    counterpart_sessions: pd.DataFrame,
    counterpart_scenarios: pd.DataFrame,
    repo_root: Path,
    seed: int = 0,
) -> pd.DataFrame:
    ai_flavour_map = _load_subject_ai_flavour_map(repo_root)
    ai_library, unpersonalized_indices = _load_ai_library(repo_root)
    focal_session_lookup, focal_scenario_lookup = _prepare_session_context(
        sessions=focal_sessions,
        scenarios=focal_scenarios,
        ai_flavour_map=ai_flavour_map,
    )
    counterpart_session_lookup, counterpart_scenario_lookup = _prepare_session_context(
        sessions=counterpart_sessions,
        scenarios=counterpart_scenarios,
        ai_flavour_map=ai_flavour_map,
    )

    rows: list[dict[str, Any]] = []
    treatment_codes = sorted(
        set(focal_sessions["TreatmentCode"].dropna().astype(str))
        & set(counterpart_sessions["TreatmentCode"].dropna().astype(str))
    )

    for treatment_code in treatment_codes:
        focal_ids = [
            str(record_id)
            for record_id in focal_sessions.loc[
                focal_sessions["TreatmentCode"].astype(str) == treatment_code, "record_id"
            ].astype(str)
        ]
        counterpart_ids = [
            str(record_id)
            for record_id in counterpart_sessions.loc[
                counterpart_sessions["TreatmentCode"].astype(str) == treatment_code, "record_id"
            ].astype(str)
        ]
        if len(focal_ids) == 0 or len(counterpart_ids) <= 1:
            continue

        personalized = treatment_code.endswith("P")
        transparency = treatment_code[0]
        delegation_mode = treatment_code[1]

        case_specs: list[tuple[str, str | None]]
        if delegation_mode == "R":
            case_specs = [("Benchmark", "AgainstHuman"), ("AIR", "AgainstAI")]
        elif transparency == "T":
            case_specs = [("TD", None)]
        else:
            case_specs = [("OD", None)]

        for focal_id in focal_ids:
            focal_session = focal_session_lookup.get(focal_id)
            if focal_session is None:
                continue
            focal_pair_rows: list[dict[str, Any]] = []
            for other_id in counterpart_ids:
                if other_id == focal_id:
                    continue
                other_session = counterpart_session_lookup.get(other_id)
                if other_session is None:
                    continue
                for case_group, comparison in case_specs:
                    for game in GAME_ORDER:
                        if delegation_mode == "R":
                            payoff_no = _random_role_payoff(
                                game=game,
                                comparison=str(comparison),
                                focal_role="NoAISupport",
                                treatment_code=treatment_code,
                                focal_id=focal_id,
                                other_id=other_id,
                                focal_session=focal_session,
                                other_session=other_session,
                                focal_scenarios=focal_scenario_lookup,
                                other_scenarios=counterpart_scenario_lookup,
                                ai_library=ai_library,
                                unpersonalized_indices=unpersonalized_indices,
                                personalized=personalized,
                                seed=seed,
                            )
                            payoff_yes = _random_role_payoff(
                                game=game,
                                comparison=str(comparison),
                                focal_role="AISupport",
                                treatment_code=treatment_code,
                                focal_id=focal_id,
                                other_id=other_id,
                                focal_session=focal_session,
                                other_session=other_session,
                                focal_scenarios=focal_scenario_lookup,
                                other_scenarios=counterpart_scenario_lookup,
                                ai_library=ai_library,
                                unpersonalized_indices=unpersonalized_indices,
                                personalized=personalized,
                                seed=seed,
                            )
                        else:
                            payoff_no = _delegation_role_payoff(
                                game=game,
                                transparency=transparency,
                                treatment_code=treatment_code,
                                focal_role="NoAISupport",
                                focal_id=focal_id,
                                other_id=other_id,
                                focal_session=focal_session,
                                other_session=other_session,
                                focal_scenarios=focal_scenario_lookup,
                                other_scenarios=counterpart_scenario_lookup,
                                ai_library=ai_library,
                                unpersonalized_indices=unpersonalized_indices,
                                personalized=personalized,
                                seed=seed,
                            )
                            payoff_yes = _delegation_role_payoff(
                                game=game,
                                transparency=transparency,
                                treatment_code=treatment_code,
                                focal_role="AISupport",
                                focal_id=focal_id,
                                other_id=other_id,
                                focal_session=focal_session,
                                other_session=other_session,
                                focal_scenarios=focal_scenario_lookup,
                                other_scenarios=counterpart_scenario_lookup,
                                ai_library=ai_library,
                                unpersonalized_indices=unpersonalized_indices,
                                personalized=personalized,
                                seed=seed,
                            )
                        payoff_both = float(np.nanmean([payoff_no, payoff_yes])) if any(
                            pd.notna(value) for value in [payoff_no, payoff_yes]
                        ) else np.nan
                        focal_pair_rows.extend(
                            [
                                {
                                    "record_id": focal_id,
                                    "counterpart_id": other_id,
                                    "treatment_code": treatment_code,
                                    "case_group": case_group,
                                    "game": game,
                                    "role_slice": "no",
                                    "pair_payoff": payoff_no,
                                },
                                {
                                    "record_id": focal_id,
                                    "counterpart_id": other_id,
                                    "treatment_code": treatment_code,
                                    "case_group": case_group,
                                    "game": game,
                                    "role_slice": "yes",
                                    "pair_payoff": payoff_yes,
                                },
                                {
                                    "record_id": focal_id,
                                    "counterpart_id": other_id,
                                    "treatment_code": treatment_code,
                                    "case_group": case_group,
                                    "game": game,
                                    "role_slice": "both",
                                    "pair_payoff": payoff_both,
                                },
                            ]
                        )
            if not focal_pair_rows:
                continue
            pair_df = pd.DataFrame(focal_pair_rows)
            subject_summary = (
                pair_df.groupby(
                    ["record_id", "treatment_code", "case_group", "game", "role_slice"],
                    as_index=False,
                    observed=False,
                )
                .agg(expected_payoff=("pair_payoff", "mean"))
            )
            for row in subject_summary.to_dict(orient="records"):
                rows.append(
                    {
                        **row,
                        "data_source": str(focal_session.get("data_source", "unknown")),
                        "variant": focal_session.get("variant"),
                        "model": focal_session.get("model"),
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["role_slice"] = pd.Categorical(frame["role_slice"], categories=ROLE_SLICE_ORDER, ordered=True)
    frame["case_group"] = pd.Categorical(frame["case_group"], categories=CASE_GROUP_ORDER, ordered=True)
    frame["game"] = pd.Categorical(frame["game"], categories=GAME_ORDER, ordered=True)
    return frame.sort_values(["case_group", "role_slice", "game", "record_id"]).reset_index(drop=True)


def compute_payoff_alignment_tables(
    *,
    generated_expected: pd.DataFrame,
    human_expected: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    common_cells = sorted(
        set(
            zip(
                generated_expected["case_group"].astype(str),
                generated_expected["role_slice"].astype(str),
                generated_expected["game"].astype(str),
            )
        )
        & set(
            zip(
                human_expected["case_group"].astype(str),
                human_expected["role_slice"].astype(str),
                human_expected["game"].astype(str),
            )
        )
    )

    dist_rows: list[dict[str, Any]] = []
    for case_group, role_slice, game in common_cells:
        generated_group = generated_expected[
            (generated_expected["case_group"].astype(str) == case_group)
            & (generated_expected["role_slice"].astype(str) == role_slice)
            & (generated_expected["game"].astype(str) == game)
        ].copy()
        human_group = human_expected[
            (human_expected["case_group"].astype(str) == case_group)
            & (human_expected["role_slice"].astype(str) == role_slice)
            & (human_expected["game"].astype(str) == game)
        ].copy()
        score = wasserstein_distance_1d(generated_group["expected_payoff"], human_group["expected_payoff"])
        dist_rows.append(
            {
                "metric_family": PAYOFF_METRIC_FAMILY,
                "case_group": case_group,
                "role_slice": role_slice,
                "game": game,
                "metric": f"{game.lower()}_expected_payoff",
                "distance_kind": "wasserstein_1d",
                "score": score,
                "generated_n": int(len(generated_group)),
                "human_n": int(len(human_group)),
            }
        )

    dist_df = pd.DataFrame(dist_rows)
    overall_rows: list[dict[str, Any]] = []
    if not dist_df.empty:
        game_summary = (
            dist_df[dist_df["role_slice"] == "both"]
            .groupby("game", as_index=False)
            .agg(
                mean_value=("score", "mean"),
                median_value=("score", "median"),
                stderr=("score", lambda s: float(pd.Series(s).std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else float("nan")),
                n_groups=("case_group", "nunique"),
            )
        )
        for row in game_summary.to_dict(orient="records"):
            overall_rows.append(
                {
                    "metric_family": PAYOFF_METRIC_FAMILY,
                    "metric": str(row["game"]).lower() + "_expected_payoff",
                    "distance_kind": "wasserstein_1d",
                    "n_groups": int(row["n_groups"]),
                    "mean_value": float(row["mean_value"]),
                    "median_value": float(row["median_value"]),
                    "stderr": float(row["stderr"]) if pd.notna(row["stderr"]) else float("nan"),
                }
            )
        mean_score = dist_df.loc[dist_df["role_slice"] == "both", "score"]
        if not mean_score.empty:
            overall_rows.append(
                {
                    "metric_family": PAYOFF_METRIC_FAMILY,
                    "metric": "mean_expected_payoff",
                    "distance_kind": "mean_wasserstein_1d",
                    "n_groups": int(dist_df.loc[dist_df["role_slice"] == "both", "case_group"].nunique()),
                    "mean_value": float(mean_score.mean()),
                    "median_value": float(mean_score.median()),
                    "stderr": float(mean_score.std(ddof=1) / np.sqrt(len(mean_score))) if len(mean_score) > 1 else float("nan"),
                }
            )

    overall_df = pd.DataFrame(overall_rows)

    human_benchmark = (
        human_expected[human_expected["case_group"].astype(str) == "Benchmark"]
        .groupby(["role_slice", "game"], as_index=False, observed=False)
        .agg(benchmark_mean_expected_payoff=("expected_payoff", "mean"))
    )
    generated_summary = (
        generated_expected.groupby(["case_group", "role_slice", "game"], as_index=False, observed=False)
        .agg(mean_expected_payoff=("expected_payoff", "mean"))
        .merge(human_benchmark, on=["role_slice", "game"], how="left")
    )
    generated_summary["data_source"] = "generated"
    human_summary = (
        human_expected.groupby(["case_group", "role_slice", "game"], as_index=False, observed=False)
        .agg(mean_expected_payoff=("expected_payoff", "mean"))
        .merge(human_benchmark, on=["role_slice", "game"], how="left")
    )
    human_summary["data_source"] = "human"
    relative_df = pd.concat([generated_summary, human_summary], ignore_index=True)
    relative_df["relative_payoff_diff_pct"] = (
        (relative_df["mean_expected_payoff"] / relative_df["benchmark_mean_expected_payoff"]) - 1.0
    ) * 100.0

    generated_relative = relative_df[relative_df["data_source"] == "generated"].rename(
        columns={
            "mean_expected_payoff": "generated_mean_expected_payoff",
            "relative_payoff_diff_pct": "generated_relative_payoff_diff_pct",
        }
    )
    human_relative = relative_df[relative_df["data_source"] == "human"].rename(
        columns={
            "mean_expected_payoff": "human_mean_expected_payoff",
            "relative_payoff_diff_pct": "human_relative_payoff_diff_pct",
        }
    )
    relative_compare = generated_relative.merge(
        human_relative[
            [
                "case_group",
                "role_slice",
                "game",
                "human_mean_expected_payoff",
                "human_relative_payoff_diff_pct",
            ]
        ],
        on=["case_group", "role_slice", "game"],
        how="left",
    )
    relative_compare["abs_error_relative_payoff_diff_pct"] = (
        relative_compare["generated_relative_payoff_diff_pct"]
        - relative_compare["human_relative_payoff_diff_pct"]
    ).abs()
    return dist_df, overall_df, relative_df, relative_compare


def sort_payoff_distance_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    sorted_frame = frame.copy()
    sorted_frame["_case_group_order"] = (
        sorted_frame["case_group"].astype(str).map({name: idx for idx, name in enumerate(CASE_GROUP_ORDER)}).fillna(99).astype(int)
    )
    sorted_frame["_role_slice_order"] = (
        sorted_frame["role_slice"].astype(str).map({name: idx for idx, name in enumerate(ROLE_SLICE_ORDER)}).fillna(99).astype(int)
    )
    sorted_frame["_game_order"] = (
        sorted_frame["game"].astype(str).map({name: idx for idx, name in enumerate(GAME_ORDER)}).fillna(99).astype(int)
    )
    sorted_frame = sorted_frame.sort_values(
        ["_case_group_order", "_role_slice_order", "_game_order", "metric"],
        kind="stable",
    )
    return sorted_frame.drop(
        columns=["_case_group_order", "_role_slice_order", "_game_order"]
    ).reset_index(drop=True)


def sort_payoff_overall_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    sorted_frame = frame.copy()
    sorted_frame["_metric_order"] = (
        sorted_frame["metric"].astype(str).map(PAYOFF_METRIC_ORDER).fillna(99).astype(int)
    )
    sorted_frame = sorted_frame.sort_values(["_metric_order", "metric"], kind="stable")
    return sorted_frame.drop(columns=["_metric_order"]).reset_index(drop=True)
