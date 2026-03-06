from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


PAIR_NAME_RE = re.compile(r"^(?P<base>.+)_(?P<side>[CT])$")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source_csv = root / "data" / "processed_data" / "df_analysis_learn.csv"

    seed_pairs_n = 10
    random_state = 42
    out_dir_name = f"data_seed_random_learning_paired300_pairs{seed_pairs_n}_rs{random_state}"
    out_root = root / "benchmark_sequential" / out_dir_name
    out_processed = out_root / "processed_data"
    out_processed.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(source_csv)
    if "paired_config" not in df.columns or "name" not in df.columns:
        raise ValueError("Expected columns `paired_config` and `name` in df_analysis_learn.csv")

    # Step 1: keep learning rows marked as paired by preprocessing.
    paired = df[df["paired_config"].eq(True)].copy()

    # Step 2: deduplicate duplicate names (18 duplicate rows in current file).
    paired = paired.drop_duplicates(subset=["name"], keep="first").copy()

    # Step 3: keep only rows with explicit {_C,_T} suffix and build pair keys.
    parsed = paired["name"].astype(str).str.extract(PAIR_NAME_RE)
    paired["pair_base"] = parsed["base"]
    paired["pair_side"] = parsed["side"]
    paired = paired[paired["pair_side"].isin(["C", "T"])].copy()

    # Step 4: complete pairs only (both C and T available) -> 150 pairs / 300 rows.
    side_counts = paired.groupby("pair_base")["pair_side"].nunique()
    complete_bases = side_counts[side_counts == 2].index.tolist()
    pool = paired[paired["pair_base"].isin(complete_bases)].copy()
    pool = pool.sort_values(["pair_base", "pair_side"]).reset_index(drop=True)

    if len(pool) != 300:
        raise ValueError(f"Expected 300 pooled rows, found {len(pool)}")

    # Step 5: random, outcome-blind seed selection at pair level.
    seed_pair_bases = (
        pd.Series(sorted(complete_bases), name="pair_base")
        .sample(n=seed_pairs_n, random_state=random_state, replace=False)
        .sort_values()
        .tolist()
    )
    seed = pool[pool["pair_base"].isin(seed_pair_bases)].copy()
    nonseed = pool[~pool["pair_base"].isin(seed_pair_bases)].copy()

    if len(seed) != 2 * seed_pairs_n:
        raise ValueError(f"Expected {2 * seed_pairs_n} seed rows, found {len(seed)}")

    # Save outputs.
    pool.to_csv(out_processed / "df_analysis_learn_complete_pairs_paired_config_true.csv", index=False)
    seed.to_csv(out_processed / "df_analysis_learn_seed_pairs_random.csv", index=False)
    nonseed.to_csv(out_processed / "df_analysis_learn_nonseed_pairs_random.csv", index=False)

    seed_pairs_df = (
        seed.sort_values(["pair_base", "pair_side"])[["pair_base", "pair_side", "name", "gameId"]]
        .reset_index(drop=True)
    )
    seed_pairs_df.to_csv(out_processed / "seed_pair_rows.csv", index=False)

    summary = {
        "source_csv": str(source_csv),
        "selection_policy": "outcome_blind_random_pair_sampling",
        "random_state": random_state,
        "pool_rows": int(len(pool)),
        "pool_pairs": int(len(complete_bases)),
        "seed_pairs": int(seed_pairs_n),
        "seed_rows": int(len(seed)),
        "nonseed_rows": int(len(nonseed)),
        "assertions": {
            "paired_config_all_true_in_pool": bool(pool["paired_config"].eq(True).all()),
            "seed_has_both_sides_for_each_pair": bool(
                (seed.groupby("pair_base")["pair_side"].nunique() == 2).all()
            ),
        },
        "seed_pair_bases": seed_pair_bases,
    }
    (out_root / "seed_selection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
