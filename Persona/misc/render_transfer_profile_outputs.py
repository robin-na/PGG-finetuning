#!/usr/bin/env python3
"""Render transfer-profile batch outputs into readable Markdown and CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


OBSERVED_ORDER = [
    "contribution_level",
    "contribution_stability",
    "response_to_others",
    "sanctioning_behavior",
    "response_to_sanctions",
    "reward_behavior",
    "communication_style",
]

LATENT_ORDER = [
    "generalized_prosociality",
    "reciprocity",
    "trust_in_others",
    "strategic_cooperation",
    "fairness_sensitivity",
    "inequality_aversion",
    "caution_about_exploitation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-jsonl", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("error"):
                continue
            content = (
                row.get("response", {})
                .get("body", {})
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content")
            )
            if not content:
                continue
            parsed = json.loads(content)
            rows.append(
                {
                    "custom_id": row.get("custom_id", ""),
                    "model": row.get("response", {}).get("body", {}).get("model", ""),
                    "profile": parsed,
                }
            )
    return rows


def short_trait_line(data: Dict, key: str) -> str:
    item = data.get(key, {})
    return f"{key}: {item.get('label', 'unknown')} ({item.get('score_0_to_100', 'NA')})"


def build_markdown(rows: List[Dict]) -> str:
    lines: List[str] = []
    lines.append("# Transfer Profile Smoke Test")
    lines.append("")
    lines.append(f"Profiles rendered: {len(rows)}")
    if rows:
        model = rows[0].get("model", "")
        if model:
            lines.append(f"Model: `{model}`")
    lines.append("")
    lines.append("## Snapshot")
    lines.append("")
    lines.append("| # | Custom ID | Headline | Contribution | Sanctions | Prosociality | Reciprocity |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for idx, row in enumerate(rows, start=1):
        profile = row["profile"]
        persona_card = profile["persona_card"]
        observed = profile["observed_in_pgg"]
        latent = profile["latent_traits"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    f"`{row['custom_id']}`",
                    persona_card["headline"].replace("|", "/"),
                    short_trait_line(observed, "contribution_level").replace("|", "/"),
                    short_trait_line(observed, "sanctioning_behavior").replace("|", "/"),
                    short_trait_line(latent, "generalized_prosociality").replace("|", "/"),
                    short_trait_line(latent, "reciprocity").replace("|", "/"),
                ]
            )
            + " |"
        )
    lines.append("")

    for idx, row in enumerate(rows, start=1):
        profile = row["profile"]
        persona_card = profile["persona_card"]
        observed = profile["observed_in_pgg"]
        latent = profile["latent_traits"]

        lines.append(f"## Profile {idx}")
        lines.append("")
        lines.append(f"**Custom ID**: `{row['custom_id']}`")
        lines.append("")
        lines.append(f"**Headline**: {persona_card['headline']}")
        lines.append("")
        lines.append(f"**Summary**: {persona_card['summary']}")
        lines.append("")
        lines.append("**Behavioral Signature**")
        for item in persona_card["behavioral_signature"]:
            lines.append(f"- {item}")
        lines.append("")
        lines.append("**Observed In PGG**")
        for key in OBSERVED_ORDER:
            item = observed[key]
            lines.append(f"- `{key}`: {item['label']} ({item['score_0_to_100']})")
            lines.append(f"  {item['rationale']}")
        lines.append("")
        lines.append("**Latent Traits**")
        for key in LATENT_ORDER:
            item = latent[key]
            lines.append(f"- `{key}`: {item['label']} ({item['score_0_to_100']})")
            lines.append(f"  {item['rationale']}")
        lines.append("")
        lines.append("**Retrieval Relevance**")
        for item in persona_card["transfer_relevance"]:
            lines.append(f"- {item}")
        lines.append("")
        lines.append("**Limits / Uncertainties**")
        for item in persona_card["limits"]:
            lines.append(f"- {item}")
        for item in profile["uncertainties"]:
            if item not in persona_card["limits"]:
                lines.append(f"- {item}")
        lines.append("")

    return "\n".join(lines)


def build_csv_rows(rows: List[Dict]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in rows:
        profile = row["profile"]
        persona_card = profile["persona_card"]
        flat: Dict[str, object] = {
            "custom_id": row["custom_id"],
            "headline": persona_card["headline"],
            "summary": persona_card["summary"],
            "retrieval_relevance_1": persona_card["transfer_relevance"][0] if persona_card["transfer_relevance"] else "",
            "retrieval_relevance_2": persona_card["transfer_relevance"][1] if len(persona_card["transfer_relevance"]) > 1 else "",
            "retrieval_relevance_3": persona_card["transfer_relevance"][2] if len(persona_card["transfer_relevance"]) > 2 else "",
        }
        for key in OBSERVED_ORDER:
            item = profile["observed_in_pgg"][key]
            flat[f"{key}_label"] = item["label"]
            flat[f"{key}_score"] = item["score_0_to_100"]
        for key in LATENT_ORDER:
            item = profile["latent_traits"][key]
            flat[f"{key}_label"] = item["label"]
            flat[f"{key}_score"] = item["score_0_to_100"]
        out.append(flat)
    return out


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.outputs_jsonl)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(build_markdown(rows), encoding="utf-8")
    write_csv(build_csv_rows(rows), args.output_csv)
    print(f"Wrote markdown: {args.output_md}")
    print(f"Wrote csv:      {args.output_csv}")
    print(f"Rows rendered:  {len(rows)}")


if __name__ == "__main__":
    main()
