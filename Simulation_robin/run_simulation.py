# run_simulation_ft_participant_fullhistory_debug.py
# ==================================================
# PARTICIPANT-VIEW simulator (full-history prompts + full debug prints)
# - One inference per agent per stage per round (contrib first for ALL, then actions for ALL)
# - Full transcript history is used for every prompt (per agent)
# - No leakage during contribution: prompt ends at that agent's <CONTRIB> opening in this round
# - ACTIONS: model now outputs an ARRAY aligned to <PEERS_CONTRIBUTIONS> order, NOT a JSON object
#   · reward-only: array of nonnegative ints  (each index = peer in <PEERS_CONTRIBUTIONS>)
#   · punish-only: array of nonpositive ints  (each index = peer in <PEERS_CONTRIBUTIONS>)
#   · both:        array of signed ints       (positive = reward units, negative = punishment units)
# - We still save CSV with dicts: data.punished, data.rewarded = {AVATAR: units}
# - SDPA/Flash attention on CUDA when available
# - Dedup env_df by 'name'
# - Auto-timestamp outputs: YYMMDDHHMM appended
# - Base-model-only mode (skip PEFT) via --use_peft False or empty adapter_path
# - Parallelism:
#   * --openai_async enables concurrent OpenAI requests within a round.
#   * --max_parallel_games runs multiple environments concurrently.
#   * Local HF models on a single GPU should not be parallelized across games.
# - Per-call timing + full prompt and raw output logging; optional JSONL sink

from __future__ import annotations
import os, sys, json, math, re, time, random, csv, datetime, ast, concurrent.futures
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from peft import PeftModel

from llm_client import LLMClient
from debug import build_debug_record, build_full_debug_record

# -------------------------
# Always-flushed logging
# -------------------------
def log(*args, **kwargs):
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# -------------------------
# Fast attention (SDPA/Flash) toggles
# -------------------------
if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)
        log("[attn] SDPA flash enabled (flash=True, mem_efficient=True, math=False)")
    except Exception as e:
        log("[attn] Could not enable SDPA flash:", e)

# -------------------------
# Small utils
# -------------------------
def _timestamp_YYMMDDHHMM() -> str:
    return datetime.datetime.now().strftime("%y%m%d%H%M")

def _with_timestamp(path: Optional[str], ts: str) -> Optional[str]:
    if not path:
        return None
    d, base = os.path.split(path)
    if not base:
        return path
    if "." in base:
        stem, ext = base.rsplit(".", 1)
        base_ts = f"{stem}_{ts}.{ext}"
    else:
        base_ts = f"{base}_{ts}"
    return os.path.join(d, base_ts)

def _with_timestamp_and_game(path: Optional[str], ts: str, game_id: str) -> Optional[str]:
    if not path:
        return None
    d, base = os.path.split(path)
    if not base:
        return path
    safe_game = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(game_id))
    if "." in base:
        stem, ext = base.rsplit(".", 1)
        base_ts = f"{stem}_{ts}_{safe_game}.{ext}"
    else:
        base_ts = f"{base}_{ts}_{safe_game}"
    return os.path.join(d, base_ts)

def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))

def _resolve_run_dir(output_root: str, run_id: Optional[str]) -> str:
    run_id = run_id or _timestamp_YYMMDDHHMM()
    return os.path.join(output_root, run_id)

def _relocate_output(path: Optional[str], directory: str) -> Optional[str]:
    if not path:
        return None
    base = os.path.basename(path)
    return os.path.join(directory, base) if base else directory

def _per_game_dir(run_dir: str, game_id: str) -> str:
    return os.path.join(run_dir, _safe_name(game_id))

# -------------------------
# Low-level generation utils
# -------------------------
def _load_model(base_model: str, adapter_path: Optional[str], use_peft: bool):
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = (
        torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported())
        else (torch.float16 if device in ("cuda", "mps") else torch.float32)
    )

    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"   # left pad for decoder-only

    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

    if device != "cpu":
        model = model.to(device)

    if use_peft and adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        if device != "cpu":
            model = model.to(device)

    model.eval()
    log(f"[model] device={model.device} dtype={dtype} attn_impl={getattr(model.config,'attn_implementation',None)} use_cache={getattr(model.config,'use_cache',None)} peft={use_peft and bool(adapter_path)}")
    if str(model.device).startswith("cpu"):
        log("[warn] model on CPU → expect slow inference")
    return tok, model


def _extract_answer_tagged(s: str, tag: str) -> Optional[str]:
    if not isinstance(s, str):
        return None
    pattern = rf"Answer:\s*<\s*{re.escape(tag)}\s*>(.*?)</\s*{re.escape(tag)}\s*>"
    match = re.search(pattern, s, flags=re.DOTALL)
    if not match:
        return None
    return match.group(1)


def _first_int(s: str, tag: Optional[str] = None) -> Tuple[int, bool]:
    if not isinstance(s, str):
        log("[parse] expected string for integer output; defaulting to 0")
        return 0, False
    target = s
    if tag:
        tagged = _extract_answer_tagged(s, tag)
        if tagged is not None:
            target = tagged
    m = re.search(r"-?\d+", target)
    if not m:
        log(f"[parse] no integer found for tag={tag or 'raw'}; defaulting to 0")
        return 0, False
    return int(m.group(0)), True


# --- NEW: robust array parser (first top-level [ ... ]) ---
def _parse_first_int_array(s: str, tag: Optional[str] = None) -> Tuple[Optional[List[int]], bool]:
    if not isinstance(s, str):
        log("[parse] expected string for array output; defaulting to []")
        return None, False

    target = s
    if tag:
        tagged = _extract_answer_tagged(s, tag)
        if tagged is not None:
            target = tagged

    cleaned = target.strip()
    if cleaned.startswith("<<") and ">>" in cleaned:
        cleaned = cleaned[2:cleaned.find(">>")].strip()

    # Find first bracketed region
    lb = cleaned.find("[")
    rb = cleaned.find("]", lb + 1) if lb != -1 else -1
    if lb == -1 or rb == -1 or rb <= lb:
        # try to be forgiving: maybe the caller clipped at ']'
        if lb != -1:
            tail = cleaned[lb:] + "]"
            try:
                arr = json.loads(tail)
                if isinstance(arr, list):
                    return [int(x) for x in arr], True
            except Exception:
                try:
                    arr = ast.literal_eval(tail)
                    if isinstance(arr, list):
                        return [int(x) for x in arr], True
                except Exception:
                    log(f"[parse] malformed array for tag={tag or 'raw'}; defaulting to []")
                    return None, False
        log(f"[parse] no array brackets found for tag={tag or 'raw'}; defaulting to []")
        return None, False

    chunk = cleaned[lb:rb + 1]

    # Try JSON then literal_eval
    for loader in (json.loads, ast.literal_eval):
        try:
            arr = loader(chunk)
            if isinstance(arr, list):
                return [int(x) for x in arr], True
        except Exception:
            pass
    log(f"[parse] failed to parse array for tag={tag or 'raw'}; defaulting to []")
    return None, False


# -------------------------
# Text protocol builders
# -------------------------
def _system_header_lines(env: Dict, include_reasoning: bool) -> List[str]:
    lines = []
    lines.append("You are playing an online public goods game (PGG).")
    if include_reasoning:
        lines.append("For contributions, respond with two lines: Reasoning: <short rationale> then Answer: <CONTRIB> 3 </CONTRIB>.")
    else:
        lines.append("For contributions, output ONLY a single integer at the <CONTRIB> tag (no extra text).")
    # If chat exists, mention it but do not include chat content anywhere.
    if env.get("CONFIG_chat", False):
        lines.append("You can chat with other players during the round.")

    # ACTIONS: arrays aligned to <PEERS_CONTRIBUTIONS>
    if env.get("CONFIG_punishmentExists", False) and env.get("CONFIG_rewardExists", False):
        lines.append("After contributions, decide whom to punish/reward and by how many units.")
        if include_reasoning:
            lines.append("Respond with two lines: Reasoning: <short rationale> then Answer: <PUNISHMENTS_REWARDS> <<[...]>> </PUNISHMENTS_REWARDS>.")
        lines.append("At the <PUNISHMENTS_REWARDS> tag, output ONLY an array of integers aligned to the avatar order shown in <PEERS_CONTRIBUTIONS> (positive=rewards, negative=punishments, 0=neither).")
    elif env.get("CONFIG_punishmentExists", False):
        lines.append("After contributions, decide whom to punish and by how many units.")
        if include_reasoning:
            lines.append("Respond with two lines: Reasoning: <short rationale> then Answer: <PUNISHMENTS> <<[...]>> </PUNISHMENTS>.")
        lines.append("At the <PUNISHMENTS> tag, output ONLY an array of integers aligned to the avatar order shown in <PEERS_CONTRIBUTIONS>, each ≤ 0 (−n means punish by n units).")
    elif env.get("CONFIG_rewardExists", False):
        lines.append("After contributions, decide whom to reward and by how many units.")
        if include_reasoning:
            lines.append("Respond with two lines: Reasoning: <short rationale> then Answer: <REWARDS> <<[...]>> </REWARDS>.")
        lines.append("At the <REWARDS> tag, output ONLY an array of integers aligned to the avatar order shown in <PEERS_CONTRIBUTIONS>, each ≥ 0.")
    return lines


def _system_header(env: Dict, include_reasoning: bool) -> str:
    lines = ["<|begin_of_text|><|start_header_id|>system<|end_header_id|>"]
    lines.extend(_system_header_lines(env, include_reasoning))
    lines.append("<|eot_id|>")
    return "\n".join(lines)


def _system_header_plain(env: Dict, include_reasoning: bool) -> str:
    return "\n".join(_system_header_lines(env, include_reasoning))


def _build_openai_messages(system_text: str, history_chunks: List[str]) -> List[Dict[str, str]]:
    history = "\n".join(history_chunks)
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": history},
    ]


def _round_open(env: Dict, r: int) -> str:
    if env.get("CONFIG_showNRounds", False):
        return f'<ROUND i="{r} of {env["CONFIG_numRounds"]}">'
    return f'<ROUND i="{r}">'

def _round_info_line(env: Dict) -> str:
    endow = int(env.get("CONFIG_endowment", 0) or 0)
    aon = bool(env.get("CONFIG_allOrNothing", False))
    contrib_mode = f"either 0 or {endow}" if aon else f"integer from 0 to {endow}"
    if env.get("CONFIG_defaultContribProp", False):
        pre = (f"{endow} coins are currently in the public fund, and you will contribute the remainder of the coins "
               f"you choose to take for yourself. Choose the amount to contribute ({contrib_mode}).")
    else:
        pre = (f"{endow} coins are currently in your private pocket. "
               f"Choose the amount to contribute ({contrib_mode}).")
    mult = env.get("CONFIG_multiplier", "Unknown")
    return f"<ROUND_INFO> {pre} (multiplier: {mult}×). </ROUND_INFO>"

def _contrib_open() -> str:
    return "<CONTRIB>"

def _contrib_close_filled(val: Any) -> str:
    return f"<CONTRIB> {val} </CONTRIB>"

def _format_contrib_answer(val: Any, include_reasoning: bool) -> str:
    base = _contrib_close_filled(val)
    return f"Answer: {base}" if include_reasoning else base

def _contrib_format_line() -> str:
    return "FORMAT: Reasoning: <short rationale> Answer: <CONTRIB> <<...>> </CONTRIB>"

def _actions_format_line(tag: str) -> str:
    return f"FORMAT: Reasoning: <short rationale> Answer: <{tag}> <<[...]>> </{tag}>"

def _extract_reasoning(gen: str) -> str:
    if not isinstance(gen, str):
        return ""
    text = gen
    if "Reasoning:" in text:
        text = text.split("Reasoning:", 1)[1]
    if "Answer:" in text:
        text = text.split("Answer:", 1)[0]
    return text.strip()

def _redist_line(total_contrib: int, multiplied: float, active_players: int) -> str:
    m_str = f"{multiplied:.1f}" if isinstance(multiplied, (int, float)) and not math.isnan(multiplied) else ""
    # Also report redistributed_each like transcript2 does
    per = (multiplied / active_players) if active_players > 0 else float("nan")
    per_str = f'{per:.1f}' if isinstance(per, (int, float)) and not math.isnan(per) else "NA"
    return f'<REDIST total_contrib="{total_contrib}" multiplied_contrib="{m_str}" active_players="{active_players}" redistributed_each="{per_str}"/>'

def _peers_contributions_csv(roster: List[str], focal: str, contrib_math: Dict[str, int]) -> Tuple[str, List[str]]:
    """Return 'AV1=val,AV2=val,...' for peers (excluding focal) in roster order + list order."""
    peer_order = [av for av in roster if av != focal]
    parts = [f"{av}={contrib_math.get(av, 'NA')}" for av in peer_order]
    return ",".join(parts), peer_order

def _mech_info(env: Dict) -> Optional[str]:
    r_on = env.get("CONFIG_rewardExists", False)
    p_on = env.get("CONFIG_punishmentExists", False)
    if not (r_on or p_on):
        return None
    if r_on and p_on:
        return (f"It will cost you, per reward unit, {env['CONFIG_rewardCost']} coins to give a reward of {env['CONFIG_rewardMagnitude']} coins. "
                f"It will cost you, per punishment unit, {env['CONFIG_punishmentCost']} coins to impose a deduction of {env['CONFIG_punishmentMagnitude']} coins. "
                "Choose whom to punish/reward and by how many units.")
    if r_on:
        return (f"It will cost you, per unit, {env['CONFIG_rewardCost']} coins to give a reward of {env['CONFIG_rewardMagnitude']} coins. "
                "Choose whom to reward and by how many units.")
    return (f"It will cost you, per unit, {env['CONFIG_punishmentCost']} coins to impose a deduction of {env['CONFIG_punishmentMagnitude']} coins. "
            "Choose whom to punish and by how many units.")

def _actions_tag(env: Dict) -> Optional[str]:
    if env.get("CONFIG_punishmentExists", False) and env.get("CONFIG_rewardExists", False):
        return "PUNISHMENTS_REWARDS"
    if env.get("CONFIG_punishmentExists", False):
        return "PUNISHMENTS"
    if env.get("CONFIG_rewardExists", False):
        return "REWARDS"
    return None

def _actions_open_array(tag: str) -> str:
    # we open with tag and ' <<' so the model writes just the array
    return f"<{tag}> <<"

def _actions_close_filled_array(tag: str, vec: List[int]) -> str:
    return f"{json.dumps([int(x) for x in vec])}>> </{tag}>"

def _format_actions_answer(tag: str, vec: List[int], include_reasoning: bool) -> str:
    base = _actions_close_filled_array(tag, vec)
    return f"Answer: {base}" if include_reasoning else base

# Add a roster sampler (avatars always CAPITALIZED and without replacement):
AVATAR_POOL = {
    'chick','chicken','cow','crocodile','dog','duck','elephant','frog','gorilla',
    'horse','monkey','moose','owl','parrot','pinguin','rabbit','sloth','snake',
    'walrus','whale'
}
def sample_roster(env: pd.core.series.Series, seed: int = 0) -> list[str]:
    import random
    random.seed(seed)
    pool = [a.upper() for a in AVATAR_POOL]
    n = int(env["CONFIG_playerCount"])
    if n > len(pool):
        raise ValueError(f"ENV.players={n} exceeds avatar pool size={len(pool)}")
    return random.sample(pool, k=n)

# -----------------------------
# Arguments
# -----------------------------
@dataclass
class Args:
    # provider
    provider: str = field(default="local")  # local | openai
    openai_model: Optional[str] = field(default=None)
    openai_api_key: Optional[str] = field(default=None)
    openai_api_key_env: str = field(default="OPENAI_API_KEY")
    openai_async: bool = field(
        default=False,
        metadata={"help": "If true, use asyncio.gather with a thread pool for concurrent OpenAI calls."},
    )
    openai_max_concurrency: int = field(
        default=8,
        metadata={"help": "Max concurrent OpenAI calls when --openai_async is enabled."},
    )

    # model + adapter
    base_model: str = field(default="meta-llama/Llama-3.1-8B-Instruct")
    adapter_path: Optional[str] = field(default="out/llama31-8b-lora-pgg-ptc/checkpoint-489")
    use_peft: bool = field(default=True)  # set False to use base model only

    # data I/O
    env_csv: str = field(default="df_analysis_val.csv")
    output_root: str = field(
        default="output",
        metadata={"help": "Base output directory to store run-specific folders."},
    )
    run_id: Optional[str] = field(
        default=None,
        metadata={"help": "Optional run identifier; defaults to timestamp."},
    )
    group_by_game: bool = field(
        default=True,
        metadata={"help": "If true, also write per-game outputs under run/<game_id>/."},
    )
    rows_out_path: Optional[str] = field(default="output/participant_sim.csv")
    transcripts_out_path: Optional[str] = field(default="output/participant_transcripts.jsonl")
    debug_jsonl_path: Optional[str] = field(default="output/participant_debug.jsonl")

    # generation params
    temperature: float = 0.7
    top_p: float = 0.9
    seed: int = 0
    contrib_max_new_tokens: int = 100
    actions_max_new_tokens: int = 100  # arrays are short
    include_reasoning: bool = field(
        default=False,
        metadata={"help": "If true, request a short Reasoning line followed by a strict Answer line."},
    )

    # debug
    debug_print: bool = True
    debug_level: str = field(
        default="full",
        metadata={"help": "Debug output level: full | compact | off"},
    )
    debug_compact: bool = field(
        default=False,
        metadata={"help": "If true, store compact debug records (metadata + excerpt/hash)."},
    )
    debug_full_jsonl_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional JSONL path to store full prompts/outputs when compact logging is enabled."},
    )
    max_parallel_games: int = field(
        default=1,
        metadata={"help": "Run up to N environments concurrently. Use 1 for local HF GPU models."},
    )

# -------------------------
# Simulation (full history; per-agent prompts)
# -------------------------
def simulate_game(
    env: pd.Series,
    client: LLMClient,
    tok: Optional[AutoTokenizer],
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int = 0,
    contrib_max_new_tokens: int = 6,
    actions_max_new_tokens: int = 96,
    include_reasoning: bool = False,
    rows_out_path: Optional[str] = None,
    transcripts_out_path: Optional[str] = None,
    debug_jsonl_path: Optional[str] = None,
    debug_print: bool = True,
    debug_level: str = "full",
    debug_full_jsonl_path: Optional[str] = None,
    openai_async: bool = False,
    openai_max_concurrency: int = 8,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:

    debug_records: List[Dict[str, Any]] = []
    debug_full_records: List[Dict[str, Any]] = []
    debug_excerpt_chars = 200

    roster = sample_roster(env, seed=seed)
    assert len(roster) == env["CONFIG_playerCount"], "Roster length must match ENV.players"

    sys_text = _system_header(env, include_reasoning=include_reasoning)
    sys_text_plain = _system_header_plain(env, include_reasoning=include_reasoning)

    transcripts: Dict[str, List[str]] = {}
    rows = []
    game_id = env.get("name", "GAME")

    for av in roster:
        header = f"<META gameId='{game_id}' avatar='{av}'/>"
        transcripts[av] = [sys_text, header, "# GAME STARTS"]

    # CSV writer
    csv_writer = None
    csv_file = None
    if rows_out_path:
        os.makedirs(os.path.dirname(rows_out_path), exist_ok=True)
        csv_file = open(rows_out_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(csv_file, fieldnames=[
            "playerAvatar", "roundIndex", "gameId",
            "data.contribution", "data.contribution_reasoning",
            "data.punished", "data.rewarded", "data.actions_reasoning"
        ])
        csv_writer.writeheader()

    # per-round stores
    contrib_math: Dict[str, int] = {}
    contrib_rec: Dict[str, Any] = {}
    contrib_reasoning: Dict[str, Optional[str]] = {}
    actions_reasoning: Dict[str, Optional[str]] = {}

    # Round loop
    for r in range(1, int(env["CONFIG_numRounds"]) + 1):
        # ---- Phase A: contributions (per agent; NO leakage) ----
        actions_reasoning.clear()
        contrib_prompts, contrib_meta, contrib_messages = [], [], []
        for av in roster:
            transcripts[av].append(_round_open(env, r))
            transcripts[av].append(_round_info_line(env))   # NEW: per-round reminder
            if include_reasoning:
                transcripts[av].append(_contrib_format_line())
                transcripts[av].append("Reasoning:")
            else:
                transcripts[av].append(_contrib_open())
            prompt = "\n".join(transcripts[av])
            contrib_prompts.append(prompt)
            contrib_meta.append(av)
            contrib_messages.append(_build_openai_messages(sys_text_plain, transcripts[av][1:]))

        # Log prompts if debugging
        for av, ptxt in zip(contrib_meta, contrib_prompts):
            if debug_print:
                if tok is not None:
                    tok_len = len(tok(ptxt, add_special_tokens=False)["input_ids"])
                    log(f"[ptc] {game_id} r={r:02d} {av} CONTRIB prompt_tokens≈{tok_len}")
                else:
                    log(f"[ptc] {game_id} r={r:02d} {av} CONTRIB prompt_chars={len(ptxt)}")
                log("----- PROMPT (CONTRIB) -----"); log(ptxt)

        t0 = time.perf_counter()
        contrib_raw = client.generate_batch(
            prompts=contrib_prompts,
            messages_list=contrib_messages,
            stop=None,  # integer; we'll parse _first_int
            max_new_tokens=contrib_max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            async_openai=openai_async,
            max_concurrency=openai_max_concurrency,
        )
        dt = time.perf_counter() - t0

        contrib_math.clear(); contrib_rec.clear(); contrib_reasoning.clear()
        endow = int(env["CONFIG_endowment"])
        for av, gen in zip(contrib_meta, contrib_raw):
            if debug_print:
                log(f"[ptc] {game_id} r={r:02d} {av} CONTRIB dt={dt/len(contrib_raw):.3f}s (avg per agent) out='{gen}'")
            prompt_text = contrib_prompts[contrib_meta.index(av)]
            dt_per_agent = dt / len(contrib_raw)
            if debug_level != "off":
                debug_records.append(build_debug_record(
                    game_id=game_id,
                    round_idx=r,
                    agent=av,
                    phase="contrib",
                    dt_sec=dt_per_agent,
                    prompt=prompt_text,
                    raw_output=gen,
                    debug_level=debug_level,
                    excerpt_chars=debug_excerpt_chars,
                ))
            if debug_full_jsonl_path:
                debug_full_records.append(build_full_debug_record(
                    game_id=game_id,
                    round_idx=r,
                    agent=av,
                    phase="contrib",
                    dt_sec=dt_per_agent,
                    prompt=prompt_text,
                    raw_output=gen,
                ))
            val, parsed_ok = _first_int(gen, tag="CONTRIB")
            if env.get("CONFIG_allOrNothing", False):
                val = endow if val >= (endow // 2) else 0
            else:
                val = max(0, min(endow, val))
            contrib_math[av] = int(val)
            contrib_rec[av] = (float('nan') if not parsed_ok else int(val))
            if include_reasoning:
                contrib_reasoning[av] = _extract_reasoning(gen)
                transcripts[av][-1] = f"Reasoning: {contrib_reasoning[av]}"
                transcripts[av].append(
                    _format_contrib_answer(contrib_rec[av] if parsed_ok else "NaN", include_reasoning=True)
                )
            else:
                contrib_reasoning[av] = None
                transcripts[av][-1] = _contrib_close_filled(contrib_rec[av] if parsed_ok else "NaN")

        # ---- Phase B: redistribution & peers' contributions ----
        total_contrib = sum(contrib_math.values())
        try:
            multiplied = float(env["CONFIG_multiplier"]) * float(total_contrib)
        except Exception:
            multiplied = float("nan")

        active_players = len(roster)
        for av in roster:
            transcripts[av].append(_redist_line(total_contrib, multiplied, active_players))
            peers_csv, peer_order = _peers_contributions_csv(roster, av, contrib_math)
            transcripts[av].append(f"<PEERS_CONTRIBUTIONS> {peers_csv} </PEERS_CONTRIBUTIONS>")
            # Store the peer order for this agent this round by sneaking it into a small marker line for debugging
            transcripts[av].append(f"<!-- PEER_ORDER {json.dumps(peer_order)} -->")

        # ---- Phase C: actions (per agent; after contributions are done) ----
        reward_on = env.get("CONFIG_rewardExists", False)
        punish_on = env.get("CONFIG_punishmentExists", False)
        rewards_given: Dict[str, Dict[str, int]] = {av: {} for av in roster}
        punish_given: Dict[str, Dict[str, int]] = {av: {} for av in roster}

        if reward_on or punish_on:
            actions_prompts, actions_meta, peer_orders, actions_messages = [], [], {}, []
            tag = _actions_tag(env)
            mech = _mech_info(env)

            for av in roster:
                # Recompute peer order from the debug marker we just wrote to transcripts[av]
                # (or just use roster order minus self, identical here)
                peer_order = [x for x in roster if x != av]
                peer_orders[av] = peer_order

                if mech:
                    transcripts[av].append(f"<MECHANISM_INFO> {mech} </MECHANISM_INFO>")
                if include_reasoning:
                    transcripts[av].append(_actions_format_line(tag))
                    transcripts[av].append("Reasoning:")
                else:
                    transcripts[av].append(_actions_open_array(tag))  # e.g., "<PUNISHMENTS_REWARDS> <<"

                prompt = "\n".join(transcripts[av])  # FULL history including this round's REDIST/PEERS + opening tag
                actions_prompts.append(prompt)
                actions_meta.append(av)
                actions_messages.append(_build_openai_messages(sys_text_plain, transcripts[av][1:]))
                if debug_print:
                    if tok is not None:
                        tok_len = len(tok(prompt, add_special_tokens=False)["input_ids"])
                        log(f"[ptc] {game_id} r={r:02d} {av} ACTIONS prompt_tokens≈{tok_len}")
                    else:
                        log(f"[ptc] {game_id} r={r:02d} {av} ACTIONS prompt_chars={len(prompt)}")
                    log("----- PROMPT (ACTIONS) -----"); log(prompt)

            t1 = time.perf_counter()
            # Stop at first closing bracket of the array
            actions_raw = client.generate_batch(
                prompts=actions_prompts,
                messages_list=actions_messages,
                stop="]",  # ARRAY end
                max_new_tokens=actions_max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                async_openai=openai_async,
                max_concurrency=openai_max_concurrency,
            )
            dt_actions = time.perf_counter() - t1

            for av, gen in zip(actions_meta, actions_raw):
                raw = gen + "]"  # we truncated before ']', add it back for parsing
                if debug_print:
                    log(f"[ptc] {game_id} r={r:02d} {av} ACTIONS dt={dt_actions/len(actions_raw):.3f}s out='{raw}'")
                prompt_text = actions_prompts[actions_meta.index(av)]
                dt_per_agent = dt_actions / len(actions_raw)
                if debug_level != "off":
                    debug_records.append(build_debug_record(
                        game_id=game_id,
                        round_idx=r,
                        agent=av,
                        phase="actions",
                        dt_sec=dt_per_agent,
                        prompt=prompt_text,
                        raw_output=raw,
                        debug_level=debug_level,
                        excerpt_chars=debug_excerpt_chars,
                    ))
                if debug_full_jsonl_path:
                    debug_full_records.append(build_full_debug_record(
                        game_id=game_id,
                        round_idx=r,
                        agent=av,
                        phase="actions",
                        dt_sec=dt_per_agent,
                        prompt=prompt_text,
                        raw_output=raw,
                    ))

                arr, parsed_ok = _parse_first_int_array(raw, tag=tag)
                if arr is None:
                    arr = []
                peer_order = peer_orders[av]
                # Normalize length (truncate/pad)
                if len(arr) < len(peer_order):
                    arr = arr + [0] * (len(peer_order) - len(arr))
                elif len(arr) > len(peer_order):
                    arr = arr[:len(peer_order)]

                # Enforce sign constraints
                if reward_on and not punish_on:
                    arr = [max(0, int(v)) for v in arr]
                elif punish_on and not reward_on:
                    arr = [min(0, int(v)) for v in arr]
                else:
                    arr = [int(v) for v in arr]

                # Close the tag in transcript with the CLEANED array
                if include_reasoning:
                    actions_reasoning[av] = _extract_reasoning(gen)
                    transcripts[av][-1] = f"Reasoning: {actions_reasoning[av]}"
                    transcripts[av].append(_format_actions_answer(tag, arr, include_reasoning=True))
                else:
                    actions_reasoning[av] = None
                    transcripts[av].append(_actions_close_filled_array(tag, arr))

                # Split into dicts for CSV
                if reward_on:
                    for j, v in enumerate(arr):
                        if v > 0:
                            tgt = peer_order[j]
                            rewards_given[av][tgt] = int(v)
                if punish_on:
                    for j, v in enumerate(arr):
                        if v < 0:
                            tgt = peer_order[j]
                            punish_given[av][tgt] = int(-v)  # store positive units in punished dict

        # ---- Phase D: inbound + ROUND SUMMARY ----
        showPunishId = env.get("CONFIG_showPunishmentId", False)
        showRewardId = env.get("CONFIG_showRewardId", False)

        inbound_reward_units = {av: 0 for av in roster}
        inbound_punish_units = {av: 0 for av in roster}
        for src in roster:
            for tgt, u in rewards_given[src].items():
                inbound_reward_units[tgt] += int(u)
            for tgt, u in punish_given[src].items():
                inbound_punish_units[tgt] += int(u)

        num_players = int(env["CONFIG_playerCount"])
        share = (float(multiplied) / num_players) if (isinstance(multiplied, (int, float)) and num_players > 0) else 0.0

        for av in roster:
            if showPunishId and env.get("CONFIG_punishmentExists", False):
                punishers = {src: u for src in roster for (tgt, u) in punish_given[src].items() if tgt == av and u > 0}
                if punishers:
                    transcripts[av].append(f"<PUNISHED_BY json='{json.dumps(punishers, separators=(',', ':'))}'/>")
            if showRewardId and env.get("CONFIG_rewardExists", False):
                rewarders = {src: u for src in roster for (tgt, u) in rewards_given[src].items() if tgt == av and u > 0}
                if rewarders:
                    transcripts[av].append(f"<REWARDED_BY json='{json.dumps(rewarders, separators=(',', ':'))}'/>")

        for av in roster:
            f_pun_units = sum(punish_given[av].values()) if env.get("CONFIG_punishmentExists", False) else 0
            f_rew_units = sum(rewards_given[av].values()) if env.get("CONFIG_rewardExists", False) else 0

            you_dict: Dict[str, Any] = {}
            if env.get("CONFIG_punishmentExists", False):
                you_dict["coins_spent_on_punish"] = f_pun_units * int(env.get("CONFIG_punishmentCost", 0) or 0)
                you_dict["coins_deducted_from_you"] = inbound_punish_units[av] * int(env.get("CONFIG_punishmentMagnitude", 0) or 0)
            if env.get("CONFIG_rewardExists", False):
                you_dict["coins_spent_on_reward"] = f_rew_units * int(env.get("CONFIG_rewardCost", 0) or 0)
                you_dict["coins_rewarded_to_you"] = inbound_reward_units[av] * int(env.get("CONFIG_rewardMagnitude", 0) or 0)

            endow = int(env["CONFIG_endowment"])
            private_kept = endow - (0 if (isinstance(contrib_rec[av], float) and math.isnan(contrib_rec[av])) else int(contrib_rec[av]))
            coins_spent_on_punish = you_dict.get("coins_spent_on_punish", 0)
            coins_spent_on_reward = you_dict.get("coins_spent_on_reward", 0)
            coins_deducted_from_you = you_dict.get("coins_deducted_from_you", 0)
            coins_rewarded_to_you = you_dict.get("coins_rewarded_to_you", 0)
            payoff = private_kept + share - coins_spent_on_punish - coins_spent_on_reward - coins_deducted_from_you + coins_rewarded_to_you
            you_dict["payoff"] = int(payoff)

            summary_obj = {f"{av} (YOU)": you_dict}

            if env.get("CONFIG_showOtherSummaries", False):
                others_block: Dict[str, Dict[str, int]] = {}
                for other in roster:
                    if other == av:
                        continue
                    ob: Dict[str, int] = {}
                    if env.get("CONFIG_punishmentExists", False):
                        o_pun_units = sum(punish_given[other].values())
                        ob["coins_spent_on_punish"] = o_pun_units * int(env.get("CONFIG_punishmentCost", 0) or 0)
                        coins_deducted_from_them = (
                            sum(punish_given[src].get(other, 0) for src in roster) * int(env.get("CONFIG_punishmentMagnitude", 0) or 0)
                        )
                        ob["coins_deducted_from_them"] = coins_deducted_from_them
                    if env.get("CONFIG_rewardExists", False):
                        o_rew_units = sum(rewards_given[other].values())
                        ob["coins_spent_on_reward"] = o_rew_units * int(env.get("CONFIG_rewardCost", 0) or 0)
                        coins_rewarded_to_them = (
                            sum(rewards_given[src].get(other, 0) for src in roster) * int(env.get("CONFIG_rewardMagnitude", 0) or 0)
                        )
                        ob["coins_rewarded_to_them"] = coins_rewarded_to_them

                    endow = int(env["CONFIG_endowment"])
                    private_kept_other = endow - int(contrib_math[other])
                    spend_pun_other = ob.get("coins_spent_on_punish", 0)
                    spend_rew_other = ob.get("coins_spent_on_reward", 0)
                    payoff_other = (
                        private_kept_other + share
                        - spend_pun_other - spend_rew_other
                        - ob.get("coins_deducted_from_them", 0) + ob.get("coins_rewarded_to_them", 0)
                    )
                    ob["payoff"] = int(payoff_other)
                    others_block[other] = ob

                summary_obj.update(others_block)

            transcripts[av].append(
                f"<ROUND SUMMARY json='{json.dumps(summary_obj, separators=(',', ':'))}'/>"
            )
            transcripts[av].append("</ROUND>")

        # ---- Phase E: record CSV rows ----
        for av in roster:
            punished_str = json.dumps(punish_given[av], separators=(",", ":")) if env.get("CONFIG_punishmentExists", False) else None
            rewarded_str = json.dumps(rewards_given[av], separators=(",", ":")) if env.get("CONFIG_rewardExists", False) else None

            row = {
                "playerAvatar": av,
                "roundIndex": r,
                "gameId": game_id,
                "data.contribution": contrib_rec[av],
                "data.contribution_reasoning": contrib_reasoning.get(av),
                "data.punished": punished_str,
                "data.rewarded": rewarded_str,
                "data.actions_reasoning": actions_reasoning.get(av),
            }
            if csv_writer:
                csv_writer.writerow(row)
            rows.append(row)

        if csv_file:
            csv_file.flush()
            os.fsync(csv_file.fileno())

    if csv_file:
        csv_file.close()

    df = pd.DataFrame(rows)

    # transcripts: one per agent
    if transcripts_out_path:
        os.makedirs(os.path.dirname(transcripts_out_path), exist_ok=True)
        transcripts_str = {av: "\n".join(chunks) for av, chunks in transcripts.items()}
        with open(transcripts_out_path, "w", encoding="utf-8") as f:
            for av, text in transcripts_str.items():
                f.write(json.dumps({"experiment": game_id, "participant": av, "text": text}, ensure_ascii=False) + "\n")

    # debug JSONL
    if debug_jsonl_path and debug_level != "off":
        os.makedirs(os.path.dirname(debug_jsonl_path), exist_ok=True)
        with open(debug_jsonl_path, "a", encoding="utf-8") as fdbg:
            for rec in debug_records:
                fdbg.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if debug_full_jsonl_path:
        os.makedirs(os.path.dirname(debug_full_jsonl_path), exist_ok=True)
        with open(debug_full_jsonl_path, "a", encoding="utf-8") as fdbg_full:
            for rec in debug_full_records:
                fdbg_full.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return df, transcripts

# -------------------------
# Orchestrate over envs
# -------------------------
def simulate_games(
    env_df: pd.DataFrame,
    provider: str,
    base_model: str,
    adapter_path: Optional[str],
    use_peft: bool,
    openai_model: Optional[str],
    openai_api_key: Optional[str],
    openai_api_key_env: str,
    openai_async: bool,
    openai_max_concurrency: int,
    temperature: float,
    top_p: float,
    seed: int,
    contrib_max_new_tokens: int,
    actions_max_new_tokens: int,
    include_reasoning: bool,
    rows_out_path: Optional[str],
    transcripts_out_path: Optional[str],
    debug_jsonl_path: Optional[str],
    debug_print: bool,
    debug_level: str,
    debug_full_jsonl_path: Optional[str],
    output_root: str,
    run_id: Optional[str],
    group_by_game: bool,
    max_parallel_games: int,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, str]], str, str, str, Optional[str]]:

    run_dir = _resolve_run_dir(output_root, run_id)
    rows_out_path_run = _relocate_output(rows_out_path, run_dir)
    transcripts_out_path_run = _relocate_output(transcripts_out_path, run_dir)
    debug_jsonl_path_run = _relocate_output(debug_jsonl_path, run_dir)
    debug_full_jsonl_path_run = _relocate_output(debug_full_jsonl_path, run_dir)

    all_rows = []
    all_transcripts = {}

    tok = None
    model = None
    if provider == "local":
        tok, model = _load_model(base_model=base_model, adapter_path=adapter_path, use_peft=use_peft)
        if max_parallel_games > 1:
            log("[warn] local HF models should not be parallelized on a single GPU; forcing --max_parallel_games=1")
            max_parallel_games = 1

    def _build_client() -> LLMClient:
        return LLMClient(
            provider=provider,
            tok=tok,
            model=model,
            openai_model=openai_model,
            openai_api_key=openai_api_key,
            openai_api_key_env=openai_api_key_env,
        )

    client = _build_client()

    csv_writer = None
    csv_file = None
    if rows_out_path_run:
        os.makedirs(os.path.dirname(rows_out_path_run), exist_ok=True)
        csv_file = open(rows_out_path_run, "w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(csv_file, fieldnames=[
            "playerAvatar", "roundIndex", "gameId",
            "data.contribution", "data.contribution_reasoning",
            "data.punished", "data.rewarded", "data.actions_reasoning"
        ])
        csv_writer.writeheader()

    rng = random.Random(seed)
    env_rows = [(idx, env, rng.randrange(0, 2**32 - 1)) for idx, env in env_df.iterrows()]

    def _run_one(idx: int, env: pd.Series, game_seed: int):
        game_id = env.get("name", f"GAME_{idx}")
        log(f"[ptc] starting game {game_id} (idx={idx})")
        game_dir = _per_game_dir(run_dir, game_id) if group_by_game else run_dir
        per_game_rows = _relocate_output(rows_out_path, game_dir) if group_by_game else None
        per_game_transcripts = _relocate_output(transcripts_out_path, game_dir) if group_by_game else None
        per_game_debug = _relocate_output(debug_jsonl_path, game_dir) if group_by_game else debug_jsonl_path_run
        per_game_full_debug = (
            _relocate_output(debug_full_jsonl_path, game_dir) if group_by_game else debug_full_jsonl_path_run
        )
        df_game, transcripts_game = simulate_game(
            env=env,
            client=_build_client() if max_parallel_games > 1 else client,
            tok=tok,
            temperature=temperature,
            top_p=top_p,
            seed=game_seed,
            contrib_max_new_tokens=contrib_max_new_tokens,
            actions_max_new_tokens=actions_max_new_tokens,
            include_reasoning=include_reasoning,
            rows_out_path=per_game_rows,
            transcripts_out_path=per_game_transcripts,
            debug_jsonl_path=per_game_debug,
            debug_print=debug_print,
            debug_level=debug_level,
            debug_full_jsonl_path=per_game_full_debug,
            openai_async=openai_async,
            openai_max_concurrency=openai_max_concurrency,
        )
        log(f"[ptc] done game {game_id} with {len(df_game)} rows")
        return idx, game_id, df_game, transcripts_game, per_game_debug, per_game_full_debug

    per_game_debug_paths = []
    per_game_full_debug_paths = []

    if max_parallel_games > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel_games) as executor:
            futures = [executor.submit(_run_one, idx, env, game_seed) for idx, env, game_seed in env_rows]
            for fut in concurrent.futures.as_completed(futures):
                idx, game_id, df_game, transcripts_game, per_game_debug, per_game_full_debug = fut.result()
                all_rows.extend(df_game.to_dict("records"))
                all_transcripts[game_id] = {av: "\n".join(t) for av, t in transcripts_game.items()}
                if csv_writer:
                    for rec in df_game.to_dict("records"):
                        csv_writer.writerow(rec)
                if per_game_debug and per_game_debug != debug_jsonl_path_run:
                    per_game_debug_paths.append(per_game_debug)
                if per_game_full_debug and per_game_full_debug != debug_full_jsonl_path_run:
                    per_game_full_debug_paths.append(per_game_full_debug)
        if csv_file:
            csv_file.flush()
            os.fsync(csv_file.fileno())
        if debug_jsonl_path_run and per_game_debug_paths:
            os.makedirs(os.path.dirname(debug_jsonl_path_run), exist_ok=True)
            with open(debug_jsonl_path_run, "a", encoding="utf-8") as fdbg:
                for per_path in per_game_debug_paths:
                    if os.path.exists(per_path):
                        with open(per_path, "r", encoding="utf-8") as per_f:
                            fdbg.write(per_f.read())
        if debug_full_jsonl_path_run and per_game_full_debug_paths:
            os.makedirs(os.path.dirname(debug_full_jsonl_path_run), exist_ok=True)
            with open(debug_full_jsonl_path_run, "a", encoding="utf-8") as fdbg:
                for per_path in per_game_full_debug_paths:
                    if os.path.exists(per_path):
                        with open(per_path, "r", encoding="utf-8") as per_f:
                            fdbg.write(per_f.read())
    else:
        for idx, env, game_seed in env_rows:
            game_id = env.get("name", f"GAME_{idx}")
            log(f"[ptc] starting game {game_id} (idx={idx})")

            df_game, transcripts_game = simulate_game(
                env=env,
                client=client,
                tok=tok,
                temperature=temperature,
                top_p=top_p,
                seed=game_seed,
                contrib_max_new_tokens=contrib_max_new_tokens,
                actions_max_new_tokens=actions_max_new_tokens,
                include_reasoning=include_reasoning,
                rows_out_path=_relocate_output(rows_out_path, _per_game_dir(run_dir, game_id)) if group_by_game else None,
                transcripts_out_path=_relocate_output(transcripts_out_path, _per_game_dir(run_dir, game_id)) if group_by_game else None,
                debug_jsonl_path=_relocate_output(debug_jsonl_path, _per_game_dir(run_dir, game_id)) if group_by_game else debug_jsonl_path_run,
                debug_print=debug_print,
                openai_async=openai_async,
                debug_level=debug_level,
                debug_full_jsonl_path=_relocate_output(debug_full_jsonl_path, _per_game_dir(run_dir, game_id)) if group_by_game else debug_full_jsonl_path_run,
                openai_max_concurrency=openai_max_concurrency,
            )

            all_rows.extend(df_game.to_dict("records"))
            all_transcripts[game_id] = {av: "\n".join(t) for av, t in transcripts_game.items()}

            if csv_writer:
                for rec in df_game.to_dict("records"):
                    csv_writer.writerow(rec)
                csv_file.flush()
                os.fsync(csv_file.fileno())

            if group_by_game:
                per_game_debug = _relocate_output(debug_jsonl_path, _per_game_dir(run_dir, game_id))
                per_game_full_debug = _relocate_output(debug_full_jsonl_path, _per_game_dir(run_dir, game_id))
                if per_game_debug and per_game_debug != debug_jsonl_path_run:
                    per_game_debug_paths.append(per_game_debug)
                if per_game_full_debug and per_game_full_debug != debug_full_jsonl_path_run:
                    per_game_full_debug_paths.append(per_game_full_debug)

            log(f"[ptc] done game {game_id} with {len(df_game)} rows")

        if debug_jsonl_path_run and per_game_debug_paths:
            os.makedirs(os.path.dirname(debug_jsonl_path_run), exist_ok=True)
            with open(debug_jsonl_path_run, "a", encoding="utf-8") as fdbg:
                for per_path in per_game_debug_paths:
                    if os.path.exists(per_path):
                        with open(per_path, "r", encoding="utf-8") as per_f:
                            fdbg.write(per_f.read())
        if debug_full_jsonl_path_run and per_game_full_debug_paths:
            os.makedirs(os.path.dirname(debug_full_jsonl_path_run), exist_ok=True)
            with open(debug_full_jsonl_path_run, "a", encoding="utf-8") as fdbg:
                for per_path in per_game_full_debug_paths:
                    if os.path.exists(per_path):
                        with open(per_path, "r", encoding="utf-8") as per_f:
                            fdbg.write(per_f.read())

    if csv_file:
        csv_file.close()

    if transcripts_out_path_run:
        os.makedirs(os.path.dirname(transcripts_out_path_run), exist_ok=True)
        with open(transcripts_out_path_run, "w", encoding="utf-8") as f:
            for gid, tdict in all_transcripts.items():
                for av, text in tdict.items():
                    f.write(json.dumps({"experiment": gid, "participant": av, "text": text}, ensure_ascii=False) + "\n")

    return (
        pd.DataFrame(all_rows),
        all_transcripts,
        rows_out_path_run,
        transcripts_out_path_run,
        debug_jsonl_path_run,
        debug_full_jsonl_path_run,
    )

# -------------------------
# CLI entry
# -------------------------
@dataclass
class CLIArgs(Args):
    pass

def main(args: CLIArgs):
    log(args)
    if args.debug_compact:
        args.debug_level = "compact"
    if args.debug_level not in {"full", "compact", "off"}:
        log(f"[warn] unknown debug_level='{args.debug_level}', defaulting to 'full'")
        args.debug_level = "full"
    if args.debug_level == "off":
        args.debug_jsonl_path = None
        args.debug_full_jsonl_path = None
    env_df = pd.read_csv(args.env_csv)
    if "name" in env_df.columns:
        before = len(env_df)
        env_df = env_df.drop_duplicates(subset="name", keep="first")
        log(f"[env] dedup by name: {before} -> {len(env_df)} rows")

    df_all, transcripts_all, rows_path, trans_path, dbg_path, dbg_full_path = simulate_games(
        env_df=env_df.iloc[:40],
        provider=args.provider,
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        use_peft=args.use_peft,
        openai_model=args.openai_model,
        openai_api_key=args.openai_api_key,
        openai_api_key_env=args.openai_api_key_env,
        openai_async=args.openai_async,
        openai_max_concurrency=args.openai_max_concurrency,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        contrib_max_new_tokens=args.contrib_max_new_tokens,
        actions_max_new_tokens=args.actions_max_new_tokens,
        include_reasoning=args.include_reasoning,
        rows_out_path=args.rows_out_path,
        transcripts_out_path=args.transcripts_out_path,
        debug_jsonl_path=args.debug_jsonl_path,
        debug_print=args.debug_print,
        debug_level=args.debug_level,
        debug_full_jsonl_path=args.debug_full_jsonl_path,
        output_root=args.output_root,
        run_id=args.run_id,
        group_by_game=args.group_by_game,
        max_parallel_games=args.max_parallel_games,
    )

    log(f"[ptc] saved CSV → {rows_path}")
    log(f"[ptc] saved transcripts JSONL → {trans_path}")
    if dbg_path:
        log(f"[ptc] saved raw debug JSONL → {dbg_path}")
    if dbg_full_path and args.debug_level != "off":
        log(f"[ptc] saved full debug JSONL → {dbg_full_path}")

if __name__ == "__main__":
    parser = HfArgumentParser(CLIArgs)
    parsed, _unknown = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    cfg = parsed[0] if isinstance(parsed, (list, tuple)) else parsed
    if _unknown:
        log("[note] unknown args (ignored):", _unknown)
    main(cfg)
