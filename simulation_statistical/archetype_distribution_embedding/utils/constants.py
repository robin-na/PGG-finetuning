from __future__ import annotations

REQUIRED_CONFIG_COLUMNS = [
    "CONFIG_playerCount",
    "CONFIG_numRounds",
    "CONFIG_showNRounds",
    "CONFIG_allOrNothing",
    "CONFIG_chat",
    "CONFIG_defaultContribProp",
    "CONFIG_punishmentExists",
    "CONFIG_punishmentCost",
    "CONFIG_punishmentTech",
    "CONFIG_rewardExists",
    "CONFIG_rewardCost",
    "CONFIG_rewardTech",
    "CONFIG_showOtherSummaries",
    "CONFIG_showPunishmentId",
    "CONFIG_showRewardId",
    "CONFIG_MPCR",
]

CANONICAL_TAGS = [
    "CONTRIBUTION",
    "COMMUNICATION",
    "PUNISHMENT",
    "RESPONSE_TO_END_GAME",
    "RESPONSE_TO_OTHERS_OUTCOME",
    "RESPONSE_TO_PUNISHER",
    "RESPONSE_TO_REWARDER",
    "REWARD",
]

GMM_CLUSTER_GRID_DEFAULT = [6, 8, 10, 12, 15, 20]
EMBEDDING_MODEL_DEFAULT = "text-embedding-3-large"
PCA_COMPONENTS_DEFAULT = 50

MERGE_UNMATCHED_THRESHOLD = 0.01
TAG_PARSE_LOSS_THRESHOLD = 0.05
EPSILON = 1e-8

ENV_BOOLEAN_COLUMNS = [
    "CONFIG_showNRounds",
    "CONFIG_allOrNothing",
    "CONFIG_chat",
    "CONFIG_punishmentExists",
    "CONFIG_rewardExists",
    "CONFIG_showOtherSummaries",
    "CONFIG_showPunishmentId",
    "CONFIG_showRewardId",
]

ENV_CATEGORICAL_COLUMNS = [
    "CONFIG_punishmentTech",
    "CONFIG_rewardTech",
]

ENV_NUMERIC_COLUMNS = [
    "CONFIG_playerCount",
    "CONFIG_numRounds",
    "CONFIG_defaultContribProp",
    "CONFIG_punishmentCost",
    "CONFIG_rewardCost",
    "CONFIG_MPCR",
]

PUNISHMENT_FEATURE_COLUMNS = [
    "CONFIG_punishmentExists",
    "CONFIG_punishmentCost",
    "CONFIG_punishmentTech",
]

REWARD_FEATURE_COLUMNS = [
    "CONFIG_rewardExists",
    "CONFIG_rewardCost",
    "CONFIG_rewardTech",
]
