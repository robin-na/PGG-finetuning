from __future__ import annotations

from typing import Any

import pandas as pd

from simulation_statistical.archetype_distribution_embedding.utils.constants import CANONICAL_TAGS


def split_tag_blocks(text: str) -> dict[str, Any]:
    blocks: dict[str, list[str]] = {}
    tag_sequence: list[str] = []
    untagged_lines: list[str] = []
    current_tag: str | None = None

    for line in (text or "").split("\n"):
        stripped = line.strip()
        if stripped.startswith("<") and stripped.endswith(">"):
            candidate = stripped[1:-1].strip()
            if candidate in CANONICAL_TAGS:
                if candidate in blocks and blocks[candidate]:
                    blocks[candidate].append("")
                current_tag = candidate
                tag_sequence.append(candidate)
                blocks.setdefault(candidate, [])
                continue
        if current_tag is None:
            untagged_lines.append(line)
        else:
            blocks[current_tag].append(line)

    block_text = {
        tag: "\n".join(lines).strip()
        for tag, lines in blocks.items()
    }
    return {
        "blocks": block_text,
        "tag_sequence": tag_sequence,
        "untagged_text": "\n".join(untagged_lines).strip(),
    }


def build_tag_block_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in df[["row_id", "wave", "game_id", "player_id", "archetype_text_clean"]].to_dict(orient="records"):
        parsed = split_tag_blocks(record["archetype_text_clean"])
        ordered_tags = list(dict.fromkeys(parsed["tag_sequence"]))
        for order_index, tag in enumerate(ordered_tags):
            if tag not in parsed["blocks"]:
                continue
            rows.append(
                {
                    "row_id": record["row_id"],
                    "wave": record["wave"],
                    "game_id": record["game_id"],
                    "player_id": record["player_id"],
                    "tag": tag,
                    "tag_order": order_index,
                    "tag_text": parsed["blocks"][tag],
                    "untagged_text": parsed["untagged_text"],
                }
            )
        if not parsed["tag_sequence"]:
            rows.append(
                {
                    "row_id": record["row_id"],
                    "wave": record["wave"],
                    "game_id": record["game_id"],
                    "player_id": record["player_id"],
                    "tag": "__NO_TAGS__",
                    "tag_order": -1,
                    "tag_text": "",
                    "untagged_text": parsed["untagged_text"],
                }
            )
    return pd.DataFrame(rows)
