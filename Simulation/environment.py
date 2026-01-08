"""
Public Goods Game environment and game state management.

This module implements the core game mechanics including:
- Contribution validation
- Payoff calculation
- Punishment/reward redistribution
- Round state tracking
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
from config import PGGConfig


class EfficiencyCalculator:
    """Calculate normalized efficiency with proper baseline tracking.

    Normalized efficiency measures actual group earnings relative to theoretical
    cooperation/defection baselines:

        efficiency = (E_actual - E_defect) / (E_cooperate - E_defect)

    where:
        - E_actual = sum of all payoffs across all rounds
        - E_defect = baseline if no one contributes (Nash equilibrium)
        - E_cooperate = baseline if everyone contributes max (Pareto optimal)

    Must be calculated round-by-round to handle player dropouts properly.
    """

    def __init__(self, config: PGGConfig):
        """Initialize efficiency calculator.

        Args:
            config: Game configuration
        """
        self.config = config
        self.total_actual = 0.0
        self.total_max_possible = 0.0
        self.total_min_possible = 0.0

    def update_round(self, round_state: RoundState, active_player_count: int = None):
        """Update efficiency baselines for a single round.

        Critical: Must be called every round to handle dropouts properly.

        Args:
            round_state: The completed round state
            active_player_count: Number of active players (defaults to all if None)
        """
        if active_player_count is None:
            active_player_count = len(round_state.payoffs)

        # Actual earnings this round (sum of all payoffs)
        round_actual = sum(round_state.payoffs.values())
        self.total_actual += round_actual

        # Defect baseline: Everyone keeps endowment, no public good
        # Each person gets: endowment (no contributions, no public fund)
        round_min = active_player_count * self.config.endowment
        self.total_min_possible += round_min

        # Cooperate baseline: Everyone contributes max, gets back multiplier
        # Each person gets: endowment * multiplier (their share of the pot)
        # Total contributions = N * endowment
        # Public fund = (N * endowment) * multiplier
        # Per person share = (N * endowment * multiplier) / N = endowment * multiplier
        round_max = active_player_count * self.config.endowment * self.config.multiplier
        self.total_max_possible += round_max

    def get_efficiency(self) -> float:
        """Calculate final normalized efficiency.

        Returns:
            float: Efficiency in range [0, 1] (can be negative or > 1)
                - 1.0 = Perfect cooperation (Pareto optimal)
                - 0.0 = Nash equilibrium (all defect)
                - < 0 = Worse than defection (excessive punishment)
                - > 1 = Rewards generated surplus
        """
        denominator = self.total_max_possible - self.total_min_possible
        if denominator == 0:
            return 0.0

        numerator = self.total_actual - self.total_min_possible
        return numerator / denominator

    def reset(self):
        """Reset all counters to zero."""
        self.total_actual = 0.0
        self.total_max_possible = 0.0
        self.total_min_possible = 0.0


@dataclass
class RoundState:
    """Complete state for a single round of the game.

    This dataclass stores all information about what happened in a round:
    - Player contributions
    - Chat messages (if communication enabled)
    - Public fund and payoffs
    - Punishment and reward actions
    - Cumulative wallets
    """
    round_num: int
    contributions: Dict[str, int]  # agent_id -> contribution amount
    chat_messages: List[Dict[str, str]]  # [{"agent_id": ..., "avatar_name": ..., "message": ...}]
    public_fund: float  # Total after multiplication
    payoffs: Dict[str, float]  # agent_id -> round payoff (before redistribution)
    punishments: Dict[Tuple[str, str], int]  # (punisher_id, target_id) -> units
    rewards: Dict[Tuple[str, str], int]  # (rewarder_id, target_id) -> units
    wallets: Dict[str, float]  # agent_id -> cumulative wallet


class PGGEnvironment:
    """Public Goods Game environment managing game state and mechanics.

    This class enforces game rules, calculates payoffs, and tracks the history
    of rounds. It implements the mathematical core of the PGG:

    Payoff formula (per round):
        contribution_payoff = endowment - contribution + (public_fund / N)
        where public_fund = sum(contributions) * multiplier
        and multiplier = MPCR * group_size

    Redistribution (optional):
        punisher_cost = units * punishment_cost
        target_deduction = units * punishment_impact
        rewarder_cost = units * reward_cost
        target_addition = units * reward_impact
    """

    def __init__(self, config: PGGConfig):
        """Initialize the game environment.

        Args:
            config: Game configuration with all design parameters
        """
        self.config = config
        self.round_history: List[RoundState] = []
        self.current_round: int = 0
        self.efficiency_calculator = EfficiencyCalculator(config)

    def validate_contribution(self, amount: int, agent_id: str = None) -> int:
        """Validate and adjust contribution amount based on game rules.

        Args:
            amount: Proposed contribution amount
            agent_id: Agent making the contribution (for logging)

        Returns:
            int: Valid contribution amount (clamped and adjusted)
        """
        # Clamp to valid range [0, endowment]
        clamped = max(0, min(amount, self.config.endowment))

        # Apply contribution type constraint
        if self.config.contribution_type == "all_or_nothing":
            # Force to 0 or endowment (whichever is closer)
            if clamped < self.config.endowment / 2:
                result = 0
            else:
                result = self.config.endowment

            if result != amount:
                print(f"  Note: Adjusted contribution from {amount} to {result} (all-or-nothing constraint)")
        else:
            result = clamped
            if result != amount:
                print(f"  Note: Clamped contribution from {amount} to {result}")

        return result

    def calculate_payoffs(self, contributions: Dict[str, int]) -> Dict[str, float]:
        """Calculate payoffs for all agents based on contributions.

        Implements the standard PGG payoff formula:
            payoff = (endowment - contribution) + (public_fund / group_size)

        Args:
            contributions: Dictionary mapping agent_id to contribution amount

        Returns:
            Dict mapping agent_id to round payoff (before any redistribution)
        """
        # Calculate public fund
        total_contributions = sum(contributions.values())
        public_fund = total_contributions * self.config.multiplier
        per_person_share = public_fund / len(contributions)

        # Calculate individual payoffs
        payoffs = {}
        for agent_id, contrib in contributions.items():
            # Payoff = what you kept + your share of the public fund
            kept = self.config.endowment - contrib
            payoff = kept + per_person_share
            payoffs[agent_id] = round(payoff, 2)

        return payoffs

    def apply_redistribution(
        self,
        punishments: Dict[Tuple[str, str], int],
        rewards: Dict[Tuple[str, str], int],
        current_payoffs: Dict[str, float]
    ) -> Dict[str, float]:
        """Apply punishment and reward redistribution to payoffs.

        Modifies payoffs based on punishment/reward actions:
        - Punisher pays punishment_cost per unit
        - Target loses punishment_impact per unit
        - Rewarder pays reward_cost per unit
        - Target gains reward_impact per unit

        Args:
            punishments: Dict of (punisher_id, target_id) -> units
            rewards: Dict of (rewarder_id, target_id) -> units
            current_payoffs: Current payoffs before redistribution

        Returns:
            Dict mapping agent_id to redistribution adjustment (can be negative)
        """
        adjustments = {agent_id: 0.0 for agent_id in current_payoffs.keys()}

        # Apply punishments
        for (punisher_id, target_id), units in punishments.items():
            if units > 0:
                # Punisher pays cost
                cost = units * self.config.punishment_cost
                adjustments[punisher_id] -= cost

                # Target loses impact
                impact = units * self.config.punishment_impact
                adjustments[target_id] -= impact

        # Apply rewards
        for (rewarder_id, target_id), units in rewards.items():
            if units > 0:
                # Rewarder pays cost
                cost = units * self.config.reward_cost
                adjustments[rewarder_id] -= cost

                # Target gains impact
                impact = units * self.config.reward_impact
                adjustments[target_id] += impact

        # Round to 2 decimal places
        adjustments = {agent_id: round(adj, 2) for agent_id, adj in adjustments.items()}

        return adjustments

    def get_cumulative_wallets(self, agent_ids: List[str]) -> Dict[str, float]:
        """Calculate cumulative wallets from all previous rounds.

        Args:
            agent_ids: List of agent IDs

        Returns:
            Dict mapping agent_id to cumulative earnings
        """
        wallets = {agent_id: 0.0 for agent_id in agent_ids}

        for round_state in self.round_history:
            for agent_id in agent_ids:
                wallets[agent_id] += round_state.payoffs.get(agent_id, 0)

        return wallets

    def create_round_state(
        self,
        round_num: int,
        contributions: Dict[str, int],
        chat_messages: List[Dict[str, str]],
        payoffs: Dict[str, float],
        punishments: Dict[Tuple[str, str], int],
        rewards: Dict[Tuple[str, str], int]
    ) -> RoundState:
        """Create a RoundState object with all round information.

        Args:
            round_num: Round number
            contributions: Contribution amounts
            chat_messages: Chat messages from this round
            payoffs: Final payoffs (after redistribution)
            punishments: Punishment actions
            rewards: Reward actions

        Returns:
            RoundState object
        """
        # Calculate public fund
        total_contributions = sum(contributions.values())
        public_fund = total_contributions * self.config.multiplier

        # Calculate cumulative wallets
        wallets = self.get_cumulative_wallets(list(contributions.keys()))
        for agent_id, payoff in payoffs.items():
            wallets[agent_id] += payoff

        return RoundState(
            round_num=round_num,
            contributions=contributions,
            chat_messages=chat_messages,
            public_fund=public_fund,
            payoffs=payoffs,
            punishments=punishments,
            rewards=rewards,
            wallets=wallets
        )

    def add_round_to_history(self, round_state: RoundState):
        """Add a completed round to the game history.

        Args:
            round_state: The completed round state
        """
        self.round_history.append(round_state)
        self.current_round = round_state.round_num
        # Update efficiency calculator
        self.efficiency_calculator.update_round(round_state)

    def get_agent_history(self, agent_id: str) -> List[Dict]:
        """Get the history of rounds from a specific agent's perspective.

        Args:
            agent_id: The agent whose perspective to get

        Returns:
            List of dicts with round information
        """
        history = []
        for round_state in self.round_history:
            history.append({
                "round_num": round_state.round_num,
                "my_contribution": round_state.contributions.get(agent_id, 0),
                "my_payoff": round_state.payoffs.get(agent_id, 0),
                "my_wallet": round_state.wallets.get(agent_id, 0),
                "public_fund": round_state.public_fund,
                "all_contributions": round_state.contributions,
                "chat_messages": round_state.chat_messages
            })
        return history


# ===== Testing / Demo =====
if __name__ == "__main__":
    from config import PGGConfig

    print("Testing PGG Environment")
    print("=" * 60)

    # Create a simple config
    config = PGGConfig(
        group_size=4,
        game_length=3,
        endowment=20,
        mpcr=0.4,
        punishment_enabled=True,
        punishment_cost=2,
        punishment_impact=3
    )

    print(f"Config: {config.group_size} players, multiplier={config.multiplier}")

    # Initialize environment
    env = PGGEnvironment(config)

    # Simulate Round 1
    print("\n--- Round 1 ---")
    contributions = {
        "agent_0": 10,
        "agent_1": 15,
        "agent_2": 20,
        "agent_3": 5
    }
    print(f"Contributions: {contributions}")

    payoffs = env.calculate_payoffs(contributions)
    print(f"Payoffs (before redistribution): {payoffs}")

    # Add some punishment
    punishments = {
        ("agent_1", "agent_3"): 2  # agent_1 punishes agent_3 with 2 units
    }
    adjustments = env.apply_redistribution(punishments, {}, payoffs)
    print(f"Redistribution adjustments: {adjustments}")

    final_payoffs = {agent_id: payoffs[agent_id] + adjustments[agent_id]
                     for agent_id in payoffs.keys()}
    print(f"Final payoffs: {final_payoffs}")

    # Create round state
    round_state = env.create_round_state(
        round_num=1,
        contributions=contributions,
        chat_messages=[],
        payoffs=final_payoffs,
        punishments=punishments,
        rewards={}
    )
    env.add_round_to_history(round_state)

    print(f"\nWallets after Round 1: {round_state.wallets}")
