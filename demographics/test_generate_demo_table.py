import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().with_name("generate_demo_table.py")
SPEC = importlib.util.spec_from_file_location("generate_demo_table", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load module from {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
_normalize_gender = MODULE._normalize_gender


def test_normalize_gender_does_not_map_female_to_man() -> None:
    assert _normalize_gender("female") == "woman"
    assert _normalize_gender("Female") == "woman"


def test_normalize_gender_core_cases() -> None:
    assert _normalize_gender("male") == "man"
    assert _normalize_gender("Woman") == "woman"
    assert _normalize_gender("non-binary") == "non_binary"
    assert _normalize_gender("trans male") == "man"
    assert _normalize_gender("cabbage") == "unknown"
