"""
Configuration system for Public Goods Game simulation.

This module defines the PGGConfig dataclass that holds all 14 design parameters
for configuring a PGG experiment.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class PGGConfig:
    """Configuration for a single PGG experiment.

    This class encapsulates all 14 design parameters that can be varied to test
    context sensitivity in LLM agents playing public goods games.

    Design Parameters:
    ------------------
    1. Game Structure:
       - group_size: Number of players (2-20)
       - game_length: Number of rounds (1-30)
       - horizon_knowledge: Whether players know total rounds ("known"/"unknown")

    2. Economic Parameters:
       - endowment: Coins per player per round
       - mpcr: Marginal Per Capita Return (0.06-0.7)
       - multiplier: Auto-calculated as mpcr * group_size

    3. Contribution Mechanism:
       - contribution_type: "variable" (any amount) or "all_or_nothing"
       - contribution_framing: "opt_in" (contribute from private) or "opt_out" (withdraw from public)

    4. Social Information:
       - communication_enabled: Whether chat is allowed
       - peer_outcome_visibility: Whether detailed peer outcomes are shown
       - actor_anonymity: Whether punishment/reward actor IDs are hidden

    5. Incentive Mechanisms:
       - punishment_enabled: Whether punishment is available (mutually exclusive with reward)
       - peer_incentive_cost: Unified cost per unit for punishment/reward (1-4 coins)
       - punishment_impact: Deduction to target per unit (1-4 coins)
       - reward_enabled: Whether rewards are available (mutually exclusive with punishment)
       - reward_impact: Addition to target per unit (0.5-1.5 coins)
    """

    # ===== Game Structure =====
    group_size: int = 4  # Number of players (2-20)
    game_length: int = 1  # Number of rounds (1-30)
    endowment: int = 20  # Coins per player per round
    horizon_knowledge: Literal["known", "unknown"] = "known"  # Show total rounds?

    # ===== Economic Parameters =====
    mpcr: float = 0.4  # Marginal Per Capita Return (0.06-0.7)

    # ===== Contribution Mechanism =====
    contribution_type: Literal["variable", "all_or_nothing"] = "variable"
    contribution_framing: Literal["opt_in", "opt_out"] = "opt_in"

    # ===== Social Information =====
    communication_enabled: bool = False  # Chat allowed?
    peer_outcome_visibility: bool = True  # Show detailed peer outcomes?
    actor_anonymity: bool = False  # Hide punishment/reward actor IDs?

    # ===== Peer Incentive Mechanisms =====
    # Note: punishment_enabled and reward_enabled are mutually exclusive
    punishment_enabled: bool = False
    reward_enabled: bool = False

    peer_incentive_cost: int = 1  # Unified cost per unit (1-4 coins)
    punishment_impact: int = 3  # Deduction to target per unit (1-4 coins)
    reward_impact: float = 1.0  # Addition to target per unit (0.5-1.5 coins)

    # ===== LLM Settings =====
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.5
    edsl_iterations: int = 1  # Number of iterations for EDSL surveys (reasoning refinement)

    @property
    def multiplier(self) -> float:
        """Calculate the public fund multiplier.

        In PGG, contributions are multiplied by (MPCR * N) where N is group size.
        The multiplied amount is then split equally among all players.

        Returns:
            float: The multiplier for total contributions
        """
        return self.mpcr * self.group_size

    def __post_init__(self):
        """Validate configuration parameters after initialization.

        Raises:
            ValueError: If any parameter is out of valid range or constraints are violated
        """
        # Game structure validation
        if not (2 <= self.group_size <= 20):
            raise ValueError(f"group_size must be 2-20, got {self.group_size}")
        if not (1 <= self.game_length <= 30):
            raise ValueError(f"game_length must be 1-30, got {self.game_length}")
        if self.endowment <= 0:
            raise ValueError(f"endowment must be positive, got {self.endowment}")

        # Economic parameter validation
        if not (0.06 <= self.mpcr <= 0.7):
            raise ValueError(f"mpcr must be 0.06-0.7, got {self.mpcr}")

        # Mutual exclusivity: punishment and reward cannot both be enabled
        if self.punishment_enabled and self.reward_enabled:
            raise ValueError(
                "punishment_enabled and reward_enabled cannot both be True. "
                "Only one peer incentive mechanism allowed per game."
            )

        # Validate peer_incentive_cost when any incentive mechanism is enabled
        if self.punishment_enabled or self.reward_enabled:
            if not (1 <= self.peer_incentive_cost <= 4):
                raise ValueError(
                    f"peer_incentive_cost must be 1-4, got {self.peer_incentive_cost}"
                )

        # Punishment impact validation
        if self.punishment_enabled:
            if not (1 <= self.punishment_impact <= 4):
                raise ValueError(
                    f"punishment_impact must be 1-4, got {self.punishment_impact}"
                )

        # Reward impact validation
        if self.reward_enabled:
            if not (0.5 <= self.reward_impact <= 1.5):
                raise ValueError(
                    f"reward_impact must be 0.5-1.5, got {self.reward_impact}"
                )


# ===== Example Configurations =====

# Baseline: Standard PGG with punishment
CONFIG_BASELINE = PGGConfig(
    group_size=4,
    game_length=10,
    mpcr=0.4,
    contribution_framing="opt_in",
    punishment_enabled=True,
    peer_incentive_cost=2,
    punishment_impact=3,
    peer_outcome_visibility=True
)

# Test opt-out framing
CONFIG_OPT_OUT = PGGConfig(
    group_size=4,
    game_length=10,
    mpcr=0.4,
    contribution_framing="opt_out",  # Key difference
    punishment_enabled=True,
    peer_incentive_cost=2,
    punishment_impact=3
)

# Test communication effect
CONFIG_COMMUNICATION = PGGConfig(
    group_size=4,
    game_length=10,
    mpcr=0.4,
    communication_enabled=True,  # Key difference
    punishment_enabled=True
)

# Test anonymity effect
CONFIG_ANONYMOUS = PGGConfig(
    group_size=4,
    game_length=10,
    mpcr=0.4,
    punishment_enabled=True,
    actor_anonymity=True  # Hide punisher identity
)

# Test horizon knowledge effect
CONFIG_UNKNOWN_HORIZON = PGGConfig(
    group_size=4,
    game_length=10,
    mpcr=0.4,
    horizon_knowledge="unknown",  # Don't reveal total rounds
    punishment_enabled=True
)

# Rewards-only condition
CONFIG_REWARDS = PGGConfig(
    group_size=4,
    game_length=10,
    mpcr=0.4,
    punishment_enabled=False,
    reward_enabled=True,
    peer_incentive_cost=2,
    reward_impact=1.5
)

# All-or-nothing contribution
CONFIG_ALL_OR_NOTHING = PGGConfig(
    group_size=4,
    game_length=10,
    mpcr=0.4,
    contribution_type="all_or_nothing",  # Key difference
    punishment_enabled=True
)

# Hidden visibility (no peer outcome information)
CONFIG_HIDDEN_OUTCOMES = PGGConfig(
    group_size=4,
    game_length=10,
    mpcr=0.4,
    peer_outcome_visibility=False,  # Key difference
    punishment_enabled=True
)
