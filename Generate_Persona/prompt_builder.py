from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


_FINETUNING_DIR = Path(__file__).resolve().parents[1] / "Finetuning"
if str(_FINETUNING_DIR) not in sys.path:
    sys.path.append(str(_FINETUNING_DIR))

try:
    from pgg_prompt_utils import SYSTEM_PROMPT, build_user_prompt
except Exception as exc:  # pragma: no cover - import-time failure
    raise ImportError(
        "Could not import pgg_prompt_utils from Finetuning. "
        "Ensure Finetuning/pgg_prompt_utils.py exists."
    ) from exc


def build_messages(cfg: Dict[str, Any], n_players: int) -> List[Dict[str, str]]:
    user_prompt = build_user_prompt(cfg, n_players)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_chat_prompt(
    tokenizer: Any,
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = True,
    enable_thinking: Optional[bool] = None,
) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
        }
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        return tokenizer.apply_chat_template(messages, **kwargs)

    parts: List[str] = []
    for msg in messages:
        role = msg.get("role", "user").upper()
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")
    if add_generation_prompt:
        parts.append("ASSISTANT:")
    return "\n\n".join(parts)
