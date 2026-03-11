from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AlgorithmFamilySpec:
    name: str
    description: str
    contribution_features: tuple[str, ...]
    action_features: tuple[str, ...]


COMMON_CONTRIBUTION_FEATURES: tuple[str, ...] = (
    "round_phase_visible_code",
    "roundIndex",
    "history_rounds_observed",
    "history_available",
    "rounds_remaining_visible",
    "n_players_current_round",
    "n_peers_current_round",
    "CONFIG_playerCount",
    "CONFIG_numRounds",
    "CONFIG_showNRounds",
    "CONFIG_endowment",
    "CONFIG_multiplier",
    "CONFIG_MPCR_adjusted",
    "CONFIG_allOrNothing",
    "CONFIG_chat",
    "CONFIG_punishmentExists",
    "CONFIG_punishmentCost",
    "CONFIG_punishmentMagnitude",
    "CONFIG_rewardExists",
    "CONFIG_rewardCost",
    "CONFIG_rewardMagnitude",
    "CONFIG_showOtherSummaries",
    "CONFIG_showPunishmentId",
    "CONFIG_showRewardId",
    "expected_norm_visible",
    "expected_norm_visible_bin5",
    "own_history_mean_contribution_rate",
    "own_history_mean_contribution_bin5",
    "own_history_mode_contribution_bin5",
    "peer_history_mean_contribution_rate",
    "peer_history_mean_contribution_bin5",
    "peer_history_mean_peer_std_rate",
    "peer_history_mean_zero_count",
    "peer_history_mean_full_count",
    "cumulative_punish_received_units",
    "cumulative_reward_received_units",
    "cumulative_punish_given_units",
    "cumulative_reward_given_units",
)


COMMON_ACTION_FEATURES: tuple[str, ...] = (
    "round_phase_visible_code",
    "roundIndex",
    "history_rounds_observed",
    "history_available",
    "rounds_remaining_visible",
    "n_players_current_round",
    "n_peers_current_round",
    "CONFIG_playerCount",
    "CONFIG_numRounds",
    "CONFIG_showNRounds",
    "CONFIG_endowment",
    "CONFIG_multiplier",
    "CONFIG_MPCR_adjusted",
    "CONFIG_allOrNothing",
    "CONFIG_chat",
    "CONFIG_punishmentExists",
    "CONFIG_punishmentCost",
    "CONFIG_punishmentMagnitude",
    "CONFIG_rewardExists",
    "CONFIG_rewardCost",
    "CONFIG_rewardMagnitude",
    "CONFIG_showOtherSummaries",
    "CONFIG_showPunishmentId",
    "CONFIG_showRewardId",
    "own_current_contribution_rate",
    "peer_current_mean_contribution_rate",
    "peer_current_std_contribution_rate",
    "peer_current_min_contribution_rate",
    "peer_current_max_contribution_rate",
    "n_peers_zero_current",
    "n_peers_full_current",
    "n_peers_below_expected_current",
    "n_peers_above_expected_current",
    "expected_norm_visible",
    "expected_norm_visible_bin5",
    "own_history_mean_contribution_rate",
    "own_history_mean_contribution_bin5",
    "own_history_mode_contribution_bin5",
    "peer_history_mean_contribution_rate",
    "peer_history_mean_contribution_bin5",
    "peer_history_mean_peer_std_rate",
    "peer_history_mean_zero_count",
    "peer_history_mean_full_count",
    "cumulative_punish_received_units",
    "cumulative_reward_received_units",
    "cumulative_punish_given_units",
    "cumulative_reward_given_units",
    "target_current_contribution_rate",
    "target_minus_peer_mean_current",
    "target_minus_expected_norm_visible",
    "target_current_rank_among_peers",
)


FAMILY_LIBRARY: tuple[AlgorithmFamilySpec, ...] = (
    AlgorithmFamilySpec(
        name="unconditional_cooperator",
        description="High baseline contribution with weak responsiveness to peer history.",
        contribution_features=COMMON_CONTRIBUTION_FEATURES,
        action_features=COMMON_ACTION_FEATURES + (
            "target_minus_expected_norm_visible",
            "target_current_rank_among_peers",
            "peer_current_mean_contribution_rate",
        ),
    ),
    AlgorithmFamilySpec(
        name="unconditional_defector",
        description="Low baseline contribution with minimal sanctioning or reward behavior.",
        contribution_features=COMMON_CONTRIBUTION_FEATURES,
        action_features=COMMON_ACTION_FEATURES,
    ),
    AlgorithmFamilySpec(
        name="conditional_cooperator",
        description="Contribution responds to previous peer cooperation and own prior behavior.",
        contribution_features=COMMON_CONTRIBUTION_FEATURES
        + (
            "own_prev_contribution_rate",
            "own_prev_contribution_bin5",
            "peer_prev_mean_contribution_rate",
            "peer_prev_mean_contribution_bin5",
            "peer_prev_std_contribution_rate",
        ),
        action_features=COMMON_ACTION_FEATURES
        + (
            "peer_prev_mean_contribution_rate",
            "peer_prev_mean_contribution_bin5",
            "target_minus_expected_norm_visible",
            "target_current_rank_among_peers",
        ),
    ),
    AlgorithmFamilySpec(
        name="generous_conditional_cooperator",
        description="Conditional cooperation with stronger reward sensitivity and forgiveness.",
        contribution_features=COMMON_CONTRIBUTION_FEATURES
        + (
            "own_prev_contribution_rate",
            "peer_prev_mean_contribution_rate",
            "peer_prev_std_contribution_rate",
            "rewarded_prev_any",
            "reward_received_prev_units",
            "visible_prev_peer_mean_rewards",
            "visible_prev_peer_mean_payoff",
            "visible_history_peer_mean_rewards",
            "visible_history_peer_mean_payoff",
        ),
        action_features=COMMON_ACTION_FEATURES
        + (
            "peer_prev_mean_contribution_rate",
            "rewarded_prev_any",
            "reward_received_prev_units",
            "target_minus_expected_norm_visible",
            "target_current_rank_among_peers",
            "reward_id_visible",
            "target_rewarded_focal_history_visible_count",
            "focal_rewarded_target_history_visible_count",
        ),
    ),
    AlgorithmFamilySpec(
        name="endgame_defector",
        description="More cooperative early, but contribution drops when the known horizon closes.",
        contribution_features=COMMON_CONTRIBUTION_FEATURES
        + (
            "own_prev_contribution_rate",
            "peer_prev_mean_contribution_rate",
            "rounds_remaining_visible",
        ),
        action_features=COMMON_ACTION_FEATURES
        + (
            "target_minus_expected_norm_visible",
            "rounds_remaining_visible",
        ),
    ),
    AlgorithmFamilySpec(
        name="retaliatory_punisher",
        description="Punishes below-norm peers and visible prior aggressors.",
        contribution_features=COMMON_CONTRIBUTION_FEATURES
        + (
            "own_prev_contribution_rate",
            "peer_prev_mean_contribution_rate",
            "punished_prev_any",
            "punish_received_prev_units",
        ),
        action_features=COMMON_ACTION_FEATURES
        + (
            "punished_prev_any",
            "punish_received_prev_units",
            "target_minus_expected_norm_visible",
            "target_punished_focal_prev_visible",
            "focal_punished_target_prev_visible",
            "target_punished_focal_history_visible_count",
            "focal_punished_target_history_visible_count",
            "punishment_id_visible",
        ),
    ),
    AlgorithmFamilySpec(
        name="norm_enforcer",
        description="Uses sanctioning mainly to enforce contribution norms rather than retaliate.",
        contribution_features=COMMON_CONTRIBUTION_FEATURES
        + (
            "own_prev_contribution_rate",
            "peer_prev_mean_contribution_rate",
        ),
        action_features=COMMON_ACTION_FEATURES
        + (
            "expected_norm_visible",
            "target_minus_expected_norm_visible",
            "target_minus_peer_mean_current",
            "target_current_rank_among_peers",
            "own_current_contribution_rate",
        ),
    ),
    AlgorithmFamilySpec(
        name="reward_oriented_cooperator",
        description="Contributes relatively highly and reinforces above-norm peers with reward.",
        contribution_features=COMMON_CONTRIBUTION_FEATURES
        + (
            "own_prev_contribution_rate",
            "peer_prev_mean_contribution_rate",
            "rewarded_prev_any",
            "reward_received_prev_units",
            "visible_prev_peer_mean_rewards",
            "visible_history_peer_mean_rewards",
        ),
        action_features=COMMON_ACTION_FEATURES
        + (
            "rewarded_prev_any",
            "reward_received_prev_units",
            "target_minus_expected_norm_visible",
            "target_current_rank_among_peers",
            "target_rewarded_focal_prev_visible",
            "focal_rewarded_target_prev_visible",
            "target_rewarded_focal_history_visible_count",
            "focal_rewarded_target_history_visible_count",
            "reward_id_visible",
        ),
    ),
)
