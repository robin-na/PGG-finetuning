"""Prepare and optionally run game-grounded Concordia persona generation."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from repo_env import load_repo_env

DEFAULT_CONFIG_DIR = THIS_DIR / "concordia_configs"
DEFAULT_OUTPUT_ROOT = THIS_DIR / "external" / "concordia"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _config_path_from_condition(condition: str) -> Path:
    path = DEFAULT_CONFIG_DIR / f"{condition}.json"
    if not path.is_file():
        available = ", ".join(sorted(item.stem for item in DEFAULT_CONFIG_DIR.glob("*.json")))
        raise FileNotFoundError(f"Unknown condition '{condition}'. Available: {available}")
    return path


def _slug_model(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_").replace(".", "_")


def _default_run_name(config: dict[str, Any], num_personas: int, model_name: str) -> str:
    return f"{config['condition']}_n{num_personas}_{_slug_model(model_name)}"


def _command(
    *,
    python_executable: str,
    output_path: Path,
    config: dict[str, Any],
    num_personas: int,
    generator: str,
    api_type: str,
    model_name: str,
) -> list[str]:
    return [
        python_executable,
        "-m",
        "concordia.contrib.persona_generators.generate_personas",
        "--output_path",
        str(output_path),
        "--num_personas",
        str(num_personas),
        "--generator",
        generator,
        "--initial_context",
        str(config["initial_context"]),
        "--diversity_axes",
        ",".join(config["diversity_axes"]),
        "--shared_memories",
        ",".join(config.get("shared_memories", [])),
        "--api_type",
        api_type,
        "--model_name",
        model_name,
    ]


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.expanduser().resolve() if args.config else _config_path_from_condition(args.condition)
    config = _read_json(config_path)
    num_personas = args.num_personas if args.num_personas is not None else int(config["num_personas"])
    generator = args.generator or str(config["generator"])
    api_type = args.api_type or str(config["api_type"])
    model_name = args.model_name or str(config["model_name"])
    run_name = args.run_name or _default_run_name(config, num_personas, model_name)
    output_dir = args.output_root.expanduser().resolve() / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    personas_path = output_dir / "personas.json"

    command = _command(
        python_executable=args.python_executable,
        output_path=personas_path,
        config=config,
        num_personas=num_personas,
        generator=generator,
        api_type=api_type,
        model_name=model_name,
    )
    run_config = {
        "created_at": _utc_now_iso(),
        "condition": config["condition"],
        "source_config": str(config_path),
        "num_personas": num_personas,
        "generator": generator,
        "api_type": api_type,
        "model_name": model_name,
        "initial_context": config["initial_context"],
        "diversity_axes": config["diversity_axes"],
        "shared_memories": config.get("shared_memories", []),
        "output_dir": str(output_dir),
        "personas_json": str(personas_path),
        "command": command,
        "install_hint": (
            "If Concordia is not installed in this Python environment, install it with "
            "`pip install 'gdm-concordia[openai]'` or the corresponding extra for the API provider."
        ),
    }
    _write_json(output_dir / "generation_config.json", run_config)
    (output_dir / "generation_command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    (output_dir / "initial_context.txt").write_text(config["initial_context"] + "\n", encoding="utf-8")
    (output_dir / "diversity_axes.txt").write_text("\n".join(config["diversity_axes"]) + "\n", encoding="utf-8")
    return run_config


def run(args: argparse.Namespace) -> None:
    run_config = prepare(args)
    print(json.dumps(run_config, indent=2, ensure_ascii=False))
    if not args.execute:
        return
    load_repo_env(search_from=THIS_DIR)
    command = [str(item) for item in run_config["command"]]
    subprocess.run(command, check=True)
    _trim_personas(Path(run_config["personas_json"]), int(run_config["num_personas"]))


def _trim_personas(path: Path, num_personas: int) -> None:
    personas = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(personas, list):
        raise ValueError(f"Expected generated personas to be a list: {path}")
    if len(personas) <= num_personas:
        return
    overflow_path = path.with_name("personas_untrimmed.json")
    overflow_path.write_text(json.dumps(personas, indent=2, ensure_ascii=False), encoding="utf-8")
    trimmed = personas[:num_personas]
    path.write_text(json.dumps(trimmed, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition",
        default="pgg_game_grounded_alphaevolve_5",
        choices=[
            "pgg_game_grounded_alphaevolve_5",
            "chip_bargain_game_grounded_alphaevolve_5",
        ],
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--num-personas", type=int, default=None)
    parser.add_argument("--generator", default=None)
    parser.add_argument("--api-type", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
