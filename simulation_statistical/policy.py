from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping

from .common import as_bool


@dataclass(frozen=True)
class SampledAction:
    contribution: int
    punished_avatar: Dict[str, int]
    rewarded_avatar: Dict[str, int]


@dataclass(frozen=True)
class RandomBaselinePolicyConfig:
    target_probability: float = 0.10
    action_magnitude: int = 1


def _sample_contribution(env: Mapping[str, Any], rng: random.Random) -> int:
    endowment = int(env.get("CONFIG_endowment", 20) or 20)
    if as_bool(env.get("CONFIG_allOrNothing", False)):
        return endowment if rng.random() < 0.5 else 0
    return rng.randint(0, endowment)


def _sample_peer_target(
    enabled: bool,
    focal_avatar: str,
    peer_avatars: List[str],
    rng: random.Random,
    probability: float,
    magnitude: int,
) -> Dict[str, int]:
    if not enabled or not peer_avatars:
        return {}
    if rng.random() >= probability:
        return {}
    candidates = [avatar for avatar in peer_avatars if avatar != focal_avatar]
    if not candidates:
        return {}
    return {rng.choice(candidates): int(magnitude)}


def sample_random_baseline_action(
    env: Mapping[str, Any],
    focal_avatar: str,
    peer_avatars: List[str],
    rng: random.Random,
    config: RandomBaselinePolicyConfig,
) -> SampledAction:
    contribution = _sample_contribution(env, rng)
    punished_avatar = _sample_peer_target(
        enabled=as_bool(env.get("CONFIG_punishmentExists", False)),
        focal_avatar=focal_avatar,
        peer_avatars=peer_avatars,
        rng=rng,
        probability=float(config.target_probability),
        magnitude=int(config.action_magnitude),
    )
    rewarded_avatar = _sample_peer_target(
        enabled=as_bool(env.get("CONFIG_rewardExists", False)),
        focal_avatar=focal_avatar,
        peer_avatars=peer_avatars,
        rng=rng,
        probability=float(config.target_probability),
        magnitude=int(config.action_magnitude),
    )
    return SampledAction(
        contribution=contribution,
        punished_avatar=punished_avatar,
        rewarded_avatar=rewarded_avatar,
    )
