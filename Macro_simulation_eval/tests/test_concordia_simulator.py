from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from Macro_simulation_eval.concordia_pgg import (  # noqa: E402
    MAKE_OBSERVATION_CALL,
    NEXT_ACTION_SPEC_CALL,
    NEXT_ACTING_CALL,
    RESOLVE_CALL,
    TERMINATE_CALL,
    _build_stage_plan,
    build_simultaneous_engine,
)
from Macro_simulation_eval.concordia_simulator import _build_simulation_config, _write_simulation_logs  # noqa: E402
from Macro_simulation_eval.simulator import GameContext  # noqa: E402


def test_build_stage_plan_uses_simultaneous_stages_per_round():
    env = {
        "CONFIG_numRounds": 2,
        "CONFIG_chat": True,
        "CONFIG_rewardExists": True,
        "CONFIG_punishmentExists": False,
    }

    stages = _build_stage_plan(env)

    assert [(stage.round_index, stage.name) for stage in stages] == [
        (1, "chat"),
        (1, "contribution"),
        (1, "actions"),
        (2, "chat"),
        (2, "contribution"),
        (2, "actions"),
    ]


def test_build_simultaneous_engine_uses_explicit_pgg_calls():
    engine = build_simultaneous_engine()

    assert engine._call_to_make_observation == MAKE_OBSERVATION_CALL
    assert engine._call_to_next_acting == NEXT_ACTING_CALL
    assert engine._call_to_next_action_spec == NEXT_ACTION_SPEC_CALL
    assert engine._call_to_resolve == RESOLVE_CALL
    assert engine._call_to_check_termination == TERMINATE_CALL


def test_build_simulation_config_assembles_entities_and_game_master():
    ctx = GameContext(
        game_id="g1",
        game_name="Test Game",
        env={
            "CONFIG_numRounds": 3,
            "CONFIG_chat": False,
            "CONFIG_rewardExists": False,
            "CONFIG_punishmentExists": False,
        },
        player_ids=["p1", "p2"],
        avatar_by_player={"p1": "DOG", "p2": "CAT"},
        player_by_avatar={"DOG": "p1", "CAT": "p2"},
        demographics_by_player={"p1": "", "p2": ""},
    )
    args = SimpleNamespace(concordia_agent_prefab="rational", concordia_goal="maximize payoff")

    config = _build_simulation_config(ctx=ctx, args=args, assigned_archetypes={})

    assert sorted(config.prefabs.keys()) == ["macro_pgg_game_master", "player"]
    assert len(config.instances) == 3
    assert [instance.role.value for instance in config.instances] == ["entity", "entity", "game_master"]
    assert config.instances[0].params["name"] == "DOG"
    assert config.instances[1].params["name"] == "CAT"
    assert config.instances[2].params["ctx"] is ctx


def test_write_simulation_logs_emits_json_and_html(tmp_path):
    class FakeSimulationLog:
        def to_json(self, indent=2):
            assert indent == 2
            return '{"ok": true}'

        def to_html(self, title="Simulation Log"):
            assert "game_1" in title
            return "<html>ok</html>"

    paths = _write_simulation_logs(str(tmp_path), "game_1", FakeSimulationLog())

    assert Path(paths["json"]).read_text(encoding="utf-8").strip() == '{"ok": true}'
    assert Path(paths["html"]).read_text(encoding="utf-8") == "<html>ok</html>"
