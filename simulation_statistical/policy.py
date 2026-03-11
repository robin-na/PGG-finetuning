from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping

from .common import as_bool
from .algorithmic_latent.simulator.runtime import (
    AlgorithmicLatentPolicyConfig,
    AlgorithmicLatentPolicyRuntime,
)
from .cluster_plus_policy import (
    ArchetypeClusterPlusPolicyConfig,
    ArchetypeClusterPlusPolicyRuntime,
)
from .gpu_sequence_policy import (
    GpuSequenceArchetypePolicyConfig,
    GpuSequenceArchetypePolicyRuntime,
)
from .history_conditioned_policy import (
    HistoryConditionedArchetypePolicyConfig,
    HistoryConditionedArchetypePolicyRuntime,
)
from .structured_sequence_policy import (
    ExactSequenceArchetypePolicyConfig,
    ExactSequenceArchetypePolicyRuntime,
)
from .trained_policy import ArchetypeClusterPolicyConfig, ArchetypeClusterPolicyRuntime


@dataclass(frozen=True)
class SampledAction:
    contribution: int
    punished_avatar: Dict[str, int]
    rewarded_avatar: Dict[str, int]


@dataclass(frozen=True)
class RandomBaselinePolicyConfig:
    target_probability: float = 0.10
    action_magnitude: int = 1


@dataclass(frozen=True)
class PolicyStrategyBundle:
    name: str
    random_config: RandomBaselinePolicyConfig | None = None
    trained_config: ArchetypeClusterPolicyConfig | None = None
    trained_runtime: ArchetypeClusterPolicyRuntime | None = None
    history_config: HistoryConditionedArchetypePolicyConfig | None = None
    history_runtime: HistoryConditionedArchetypePolicyRuntime | None = None
    sequence_config: Any | None = None
    sequence_runtime: Any | None = None


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


def build_policy_strategy(
    *,
    strategy_name: str,
    target_probability: float = 0.10,
    action_magnitude: int = 1,
    archetype_artifacts_root: str | None = None,
    rebuild_cluster_behavior_model: bool = False,
) -> PolicyStrategyBundle:
    normalized = str(strategy_name or "random_baseline").strip().lower()
    if normalized == "random_baseline":
        return PolicyStrategyBundle(
            name="random_baseline",
            random_config=RandomBaselinePolicyConfig(
                target_probability=float(target_probability),
                action_magnitude=int(action_magnitude),
            ),
        )
    if normalized in {"archetype_cluster", "trained_archetype_cluster"}:
        trained_config = ArchetypeClusterPolicyConfig(
            artifacts_root=archetype_artifacts_root,
            rebuild_behavior_model=bool(rebuild_cluster_behavior_model),
            cluster_source="env_model",
        )
        runtime = ArchetypeClusterPolicyRuntime.from_config(trained_config)
        return PolicyStrategyBundle(
            name="archetype_cluster",
            trained_config=trained_config,
            trained_runtime=runtime,
        )
    if normalized in {"archetype_cluster_oracle_treatment", "oracle_treatment_archetype_cluster"}:
        trained_config = ArchetypeClusterPolicyConfig(
            artifacts_root=archetype_artifacts_root,
            rebuild_behavior_model=bool(rebuild_cluster_behavior_model),
            cluster_source="val_treatment_oracle",
        )
        runtime = ArchetypeClusterPolicyRuntime.from_config(trained_config)
        return PolicyStrategyBundle(
            name="archetype_cluster_oracle_treatment",
            trained_config=trained_config,
            trained_runtime=runtime,
        )
    if normalized in {"archetype_cluster_plus", "cluster_plus_archetype", "archetype_plus"}:
        sequence_config = ArchetypeClusterPlusPolicyConfig(
            artifacts_root=archetype_artifacts_root,
            rebuild_behavior_model=bool(rebuild_cluster_behavior_model),
            cluster_source="env_model",
        )
        runtime = ArchetypeClusterPlusPolicyRuntime.from_config(sequence_config)
        return PolicyStrategyBundle(
            name="archetype_cluster_plus",
            sequence_config=sequence_config,
            sequence_runtime=runtime,
        )
    if normalized in {
        "archetype_cluster_plus_oracle_treatment",
        "cluster_plus_oracle_treatment",
        "oracle_treatment_archetype_cluster_plus",
    }:
        sequence_config = ArchetypeClusterPlusPolicyConfig(
            artifacts_root=archetype_artifacts_root,
            rebuild_behavior_model=bool(rebuild_cluster_behavior_model),
            cluster_source="val_treatment_oracle",
        )
        runtime = ArchetypeClusterPlusPolicyRuntime.from_config(sequence_config)
        return PolicyStrategyBundle(
            name="archetype_cluster_plus_oracle_treatment",
            sequence_config=sequence_config,
            sequence_runtime=runtime,
        )
    if normalized in {"history_archetype", "history_conditioned_archetype"}:
        history_config = HistoryConditionedArchetypePolicyConfig(
            artifacts_root=archetype_artifacts_root,
            rebuild_model=bool(rebuild_cluster_behavior_model),
        )
        runtime = HistoryConditionedArchetypePolicyRuntime.from_config(history_config)
        return PolicyStrategyBundle(
            name="history_archetype",
            history_config=history_config,
            history_runtime=runtime,
        )
    if normalized in {"exact_sequence_archetype", "sequence_archetype", "exact_history_archetype"}:
        sequence_config = ExactSequenceArchetypePolicyConfig(
            artifacts_root=archetype_artifacts_root,
            rebuild_model=bool(rebuild_cluster_behavior_model),
            use_cluster=True,
            cluster_source="env_model",
        )
        runtime = ExactSequenceArchetypePolicyRuntime.from_config(sequence_config)
        return PolicyStrategyBundle(
            name="exact_sequence_archetype",
            sequence_config=sequence_config,
            sequence_runtime=runtime,
        )
    if normalized in {"exact_sequence_oracle_treatment", "sequence_oracle_treatment", "oracle_treatment_sequence"}:
        sequence_config = ExactSequenceArchetypePolicyConfig(
            artifacts_root=archetype_artifacts_root,
            rebuild_model=bool(rebuild_cluster_behavior_model),
            use_cluster=True,
            cluster_source="val_treatment_oracle",
        )
        runtime = ExactSequenceArchetypePolicyRuntime.from_config(sequence_config)
        return PolicyStrategyBundle(
            name="exact_sequence_oracle_treatment",
            sequence_config=sequence_config,
            sequence_runtime=runtime,
        )
    if normalized in {"exact_sequence_history_only", "sequence_history_only", "history_only_sequence"}:
        sequence_config = ExactSequenceArchetypePolicyConfig(
            artifacts_root=archetype_artifacts_root,
            rebuild_model=bool(rebuild_cluster_behavior_model),
            use_cluster=False,
        )
        runtime = ExactSequenceArchetypePolicyRuntime.from_config(sequence_config)
        return PolicyStrategyBundle(
            name="exact_sequence_history_only",
            sequence_config=sequence_config,
            sequence_runtime=runtime,
        )
    if normalized in {"gpu_sequence_archetype", "gpu_sequence", "torch_sequence_archetype"}:
        sequence_config = GpuSequenceArchetypePolicyConfig(
            artifacts_root=archetype_artifacts_root,
            rebuild_model=bool(rebuild_cluster_behavior_model),
        )
        runtime = GpuSequenceArchetypePolicyRuntime.from_config(sequence_config)
        return PolicyStrategyBundle(
            name="gpu_sequence_archetype",
            sequence_config=sequence_config,
            sequence_runtime=runtime,
        )
    if normalized in {"algorithmic_latent_family", "algorithmic_family", "family_latent"}:
        sequence_config = AlgorithmicLatentPolicyConfig(
            artifacts_root=archetype_artifacts_root,
        )
        runtime = AlgorithmicLatentPolicyRuntime.from_config(sequence_config)
        return PolicyStrategyBundle(
            name="algorithmic_latent_family",
            sequence_config=sequence_config,
            sequence_runtime=runtime,
        )
    raise ValueError(
        "Unsupported strategy "
        f"'{strategy_name}'. Allowed values: random_baseline, archetype_cluster, archetype_cluster_oracle_treatment, archetype_cluster_plus, archetype_cluster_plus_oracle_treatment, history_archetype, exact_sequence_archetype, exact_sequence_oracle_treatment, exact_sequence_history_only, gpu_sequence_archetype, algorithmic_latent_family."
    )
