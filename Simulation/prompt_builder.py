"""
Prompt builder for constructing context-sensitive LLM prompts.

This module creates prompts that vary based on game configuration parameters,
implementing the "Agent Perception" layer that determines what LLMs see.

Key features:
- Scenario descriptions explaining game rules
- Contribution prompts with framing (opt-in vs opt-out)
- Redistribution prompts with visibility filtering
- Chat prompts (if communication enabled)
- Historical context from previous rounds
"""

from typing import List, Dict
from config import PGGConfig
from environment import RoundState


# ===== Helper Functions (ported from generate_prompts.py) =====

def pluralize(quantity, singular="coin", plural="coins"):
    """Return singular or plural form based on quantity.

    Args:
        quantity: The number to check
        singular: Singular form of the word
        plural: Plural form of the word

    Returns:
        str: Singular if quantity==1, else plural
    """
    return singular if float(quantity) == 1.0 else plural


def coins_to_words(value):
    """Translate a value to words with correct plurality.

    Args:
        value: Number of coins (int or float)

    Returns:
        str: Formatted string like "5 coins" or "1 coin"
    """
    if isinstance(value, (int, float)):
        if float(value).is_integer():
            value_str = f"{int(value)} {pluralize(value)}"
        else:
            value_str = f"{value} {pluralize(value)}"
    else:
        value_str = f"{value} {pluralize(value)}"
    return value_str


# ===== Main Prompt Builder =====

class PromptBuilder:
    """Builds context-sensitive prompts based on game configuration and state.

    This class implements the mapping from game configuration parameters to
    LLM prompt text, creating different "worlds" for agents to perceive.
    """

    def __init__(self, config: PGGConfig):
        """Initialize prompt builder with configuration.

        Args:
            config: Game configuration determining prompt structure
        """
        self.config = config

    def build_scenario_description(self) -> str:
        """Generate game rules description.

        This creates the initial game explanation that all agents see,
        explaining rules, MPCR, contribution framing, and punishment/reward.

        Returns:
            str: Formatted scenario description
        """
        lines = []

        # Basic game info
        lines.append(
            f"In this multi-player online public goods game, you will be in a group of "
            f"{self.config.group_size} players. "
            "You will refer to each other by their avatar (e.g., 'DOG', 'CHICKEN')."
        )

        lines.append(
            f"Each person is given {coins_to_words(self.config.endowment)} at the start of each round."
        )

        lines.append(
            "There will be a public fund that you can choose to contribute to—you will not be "
            "able to see others' contributions before making your own. "
            f"After everyone has contributed, the amount in the public fund will be multiplied by {self.config.multiplier}."
        )

        lines.append(
            "This amount is then evenly divided among the group as the payoff. "
            "You get to keep the payoff in addition to whatever you have left of your private funds."
        )

        lines.append("")  # Blank line

        # Contribution framing
        if self.config.contribution_framing == "opt_out":
            lines.append(
                f"You start each round with all {coins_to_words(self.config.endowment)} in the public fund "
                "and can choose to withdraw these to your private fund. The remaining coins in the public fund will be your contribution."
            )
        else:  # opt-in
            lines.append(
                f"You start each round with all {coins_to_words(self.config.endowment)} in your private fund "
                "and can contribute by moving these to the public fund."
            )

        # Contribution type
        if self.config.contribution_type == "all_or_nothing":
            lines.append(
                f"You can choose to either contribute all of your {coins_to_words(self.config.endowment)} or nothing."
            )
        else:
            lines.append(
                f"You can choose to contribute any integer amount from 0 up to your entire endowment of {coins_to_words(self.config.endowment)}."
            )

        # Communication
        if self.config.communication_enabled:
            lines.append("You can communicate with other players.")
        else:
            lines.append("You cannot communicate with other players.")

        lines.append("")  # Blank line

        # Punishment/Reward
        lines.append("After contribution and redistribution, you will see how much each player contributed to the public fund.")

        if self.config.punishment_enabled:
            lines.append(
                f"After seeing each player's contributions, players can impose deductions on each other. "
                f"Per unit deduction, the punisher spends {coins_to_words(self.config.peer_incentive_cost)}, "
                f"causing the punished player to lose {coins_to_words(self.config.punishment_impact)}."
            )

        if self.config.reward_enabled:
            lines.append(
                f"After seeing each player's contributions, players can reward each other. "
                f"Per unit reward, the rewarder spends {coins_to_words(self.config.peer_incentive_cost)} and grants "
                f"{coins_to_words(self.config.reward_impact)} to the rewarded player."
            )

        return "\n".join(lines)

    def build_round_header(self, round_num: int) -> str:
        """Build round header with optional horizon information.

        Args:
            round_num: Current round number

        Returns:
            str: Formatted round header
        """
        if self.config.horizon_knowledge == "known":
            return f"## Round {round_num} of {self.config.game_length}:"
        else:
            return f"## Round {round_num}:"

    def build_chat_prompt(
        self,
        agent_id: str,
        agent_name: str,
        round_num: int,
        chat_history: List[Dict[str, str]]
    ) -> str:
        """Build prompt for chat message decision.

        Args:
            agent_id: ID of the agent
            agent_name: Avatar name of the agent
            round_num: Current round number
            chat_history: Previous messages in this round

        Returns:
            str: Formatted chat prompt
        """
        lines = []

        lines.append(self.build_round_header(round_num))
        lines.append("### Chat Stage")

        if chat_history:
            lines.append("Previous messages:")
            for msg in chat_history:
                if msg["agent_id"] != agent_id:
                    lines.append(f"{msg['avatar_name']}: \"{msg['message']}\"")

        lines.append("\nYou can send a message to discuss strategy with other players.")
        lines.append("")
        lines.append("**Output format (required):**")
        lines.append("<REASONING>")
        lines.append("Your strategic thinking here... Keep it brief. Limit to 50 words")
        lines.append("</REASONING>")
        lines.append("<MESSAGE>")
        lines.append("Your message to other players (or 'nothing' to stay silent)")
        lines.append("</MESSAGE>")

        return "\n".join(lines)

    def build_contribution_prompt(
        self,
        agent_id: str,
        agent_name: str,
        round_num: int,
        history: List[RoundState],
        chat_messages: List[Dict[str, str]],
        agent_names: Dict[str, str] = None
    ) -> str:
        """Build prompt for contribution decision.

        This is the core prompt that varies based on framing (opt-in vs opt-out)
        and includes game history.

        Args:
            agent_id: ID of the agent
            agent_name: Avatar name of the agent
            round_num: Current round number
            history: List of previous round states
            chat_messages: Chat messages from current round
            agent_names: Mapping of agent_id to avatar_name (for history formatting)

        Returns:
            str: Formatted contribution prompt
        """
        lines = []

        # Include scenario description at the start (first round only in full context)
        if round_num == 1:
            lines.append(self.build_scenario_description())
            lines.append("\n# GAME STARTS\n")

        # Add history summary
        if history:
            for round_state in history:
                lines.append(self._format_round_summary(agent_id, agent_name, round_state, agent_names))

        # Current round header
        lines.append(self.build_round_header(round_num))

        # Chat stage (if enabled and there are messages)
        if self.config.communication_enabled and chat_messages:
            lines.append("### Chat Stage")
            for msg in chat_messages:
                lines.append(f"{msg['avatar_name']}: \"{msg['message']}\"")
            lines.append("")

        # Contribution stage
        lines.append("### Contribution Stage: Decide how much to contribute.")

        if self.config.contribution_framing == "opt_out":
            lines.append(
                f"You have {self.config.endowment} coins in the public fund. "
                f"How much do you withdraw to your private fund?"
            )
            if self.config.contribution_type == "all_or_nothing":
                lines.append(f"You must withdraw either ALL ({self.config.endowment}) or NOTHING (0).")
            else:
                lines.append(f"You can withdraw any amount from 0 to {self.config.endowment}.")
        else:  # opt-in
            lines.append(
                f"You have {self.config.endowment} coins in your private fund. "
                f"How much do you move to the public fund?"
            )
            if self.config.contribution_type == "all_or_nothing":
                lines.append(f"You must contribute either ALL ({self.config.endowment}) or NOTHING (0).")
            else:
                lines.append(f"You can contribute any amount from 0 to {self.config.endowment}.")

        # Add structured output format
        lines.append("")
        lines.append("**Output format (required):**")
        lines.append("<REASONING>")
        lines.append("Your strategic thinking and reasoning here...")
        lines.append("</REASONING>")
        lines.append("<CONTRIBUTE>")
        lines.append("A single integer number only (no text)")
        lines.append("</CONTRIBUTE>")

        return "\n".join(lines)

    def build_redistribution_prompt(
        self,
        agent_id: str,
        agent_name: str,
        round_num: int,
        contributions: Dict[str, int],
        other_agents: List[Dict[str, str]],  # [{"agent_id": ..., "avatar_name": ...}]
        current_wallet: float = None
    ) -> str:
        """Build prompt for punishment/reward decisions (EDSL CheckBox format).

        This prompt is designed for EDSL QuestionCheckBox where each selection = 1 unit.
        The max_selections parameter enforces budget constraint.

        Args:
            agent_id: ID of the focal agent
            agent_name: Avatar name of the focal agent
            round_num: Current round number
            contributions: Contributions from all agents this round
            other_agents: List of other agents (dicts with agent_id and avatar_name)
            current_wallet: Current wallet balance after contribution stage

        Returns:
            str: Formatted redistribution prompt for CheckBox question
        """
        lines = []

        lines.append("### Redistribution Stage")
        lines.append("")

        # Show ALL contributions (full visibility in Phase 2)
        lines.append("**All contributions are now revealed:**")
        my_contrib = contributions.get(agent_id, 0)
        lines.append(f"- YOU ({agent_name}) contributed {my_contrib} coins")
        for other in other_agents:
            contrib = contributions.get(other["agent_id"], 0)
            lines.append(f"- {other['avatar_name']} contributed {contrib} coins")
        lines.append("")

        # Explain mechanics with unified cost
        if self.config.punishment_enabled:
            lines.append(
                f"You can now punish other players. "
                f"Each punishment costs you {self.config.peer_incentive_cost} coin(s) "
                f"and deducts {self.config.punishment_impact} coin(s) from the target."
            )
            action_verb = "punish"
            action_noun = "punishment"
        elif self.config.reward_enabled:
            lines.append(
                f"You can now reward other players. "
                f"Each reward costs you {self.config.peer_incentive_cost} coin(s) "
                f"and grants {self.config.reward_impact} coin(s) to the target."
            )
            action_verb = "reward"
            action_noun = "reward"
        else:
            action_verb = "select"
            action_noun = "action"

        # Budget constraint
        if current_wallet is not None:
            max_actions = int(current_wallet / self.config.peer_incentive_cost)
            lines.append("")
            lines.append(f"**Budget:** You have {current_wallet:.2f} coins.")
            lines.append(
                f"You can {action_verb} up to {max_actions} player(s) "
                f"(at {self.config.peer_incentive_cost} coins per {action_noun})."
            )

        # CheckBox selection format
        lines.append("")
        lines.append(f"**Select which players you want to {action_verb}:**")
        lines.append("(You can select multiple players, or none)")
        lines.append("")
        for other in other_agents:
            lines.append(f"☐ {other['avatar_name']}")

        return "\n".join(lines)

    def _format_round_summary(
        self,
        agent_id: str,
        agent_name: str,
        round_state: RoundState,
        agent_names: Dict[str, str] = None
    ) -> str:
        """Format a round summary for inclusion in history.

        Args:
            agent_id: Focal agent ID
            agent_name: Focal agent avatar name
            round_state: The round state to summarize
            agent_names: Mapping of agent_id to avatar_name (optional)

        Returns:
            str: Formatted round summary
        """
        if agent_names is None:
            agent_names = {}

        lines = []

        lines.append(self.build_round_header(round_state.round_num))

        # Contribution stage
        my_contrib = round_state.contributions.get(agent_id, 0)
        lines.append(f"### Contribution Stage")
        lines.append(f"You contributed {my_contrib} {pluralize(my_contrib)}.")

        # Redistribution stage
        total_contrib = sum(round_state.contributions.values())
        lines.append(f"### Redistribution Stage")
        lines.append(
            f"Total group contribution is {total_contrib} {pluralize(total_contrib)}, "
            f"multiplied by {self.config.multiplier} => {round_state.public_fund} {pluralize(round_state.public_fund)} "
            f"in the public fund => divided and redistributed to each player."
        )

        # Outcome stage (visibility-dependent)
        if self.config.peer_outcome_visibility:
            lines.append("### Outcome Stage")
            for other_id, other_contrib in round_state.contributions.items():
                if other_id != agent_id:
                    other_name = agent_names.get(other_id, "Another player")
                    lines.append(f"{other_name} contributed {other_contrib} {pluralize(other_contrib)}.")

        # Punishment/Reward feedback (anonymity-dependent)
        if self.config.punishment_enabled or self.config.reward_enabled:
            # Calculate what this agent received
            punishments_received = []
            rewards_received = []

            for (actor_id, target_id), units in round_state.punishments.items():
                if target_id == agent_id and units > 0:
                    actor_name = agent_names.get(actor_id, "Someone")
                    punishments_received.append((actor_id, actor_name, units))

            for (actor_id, target_id), units in round_state.rewards.items():
                if target_id == agent_id and units > 0:
                    actor_name = agent_names.get(actor_id, "Someone")
                    rewards_received.append((actor_id, actor_name, units))

            # Display punishment feedback
            if punishments_received:
                if self.config.actor_anonymity:
                    # Anonymous: don't reveal who punished
                    total_punishment_units = sum(units for _, _, units in punishments_received)
                    total_impact = total_punishment_units * self.config.punishment_impact
                    lines.append(
                        f"You received {total_punishment_units} {pluralize(total_punishment_units, 'punishment unit', 'punishment units')}, "
                        f"deducting {coins_to_words(total_impact)}."
                    )
                else:
                    # Revealed: show who punished
                    for actor_id, actor_name, units in punishments_received:
                        impact = units * self.config.punishment_impact
                        lines.append(
                            f"{actor_name} punished you with {units} {pluralize(units, 'unit', 'units')}, "
                            f"deducting {coins_to_words(impact)}."
                        )

            # Display reward feedback
            if rewards_received:
                if self.config.actor_anonymity:
                    # Anonymous: don't reveal who rewarded
                    total_reward_units = sum(units for _, _, units in rewards_received)
                    total_impact = total_reward_units * self.config.reward_impact
                    lines.append(
                        f"You received {total_reward_units} {pluralize(total_reward_units, 'reward unit', 'reward units')}, "
                        f"granting you {coins_to_words(total_impact)}."
                    )
                else:
                    # Revealed: show who rewarded
                    for actor_id, actor_name, units in rewards_received:
                        impact = units * self.config.reward_impact
                        lines.append(
                            f"{actor_name} rewarded you with {units} {pluralize(units, 'unit', 'units')}, "
                            f"granting you {coins_to_words(impact)}."
                        )

        # Round summary
        my_payoff = round_state.payoffs.get(agent_id, 0)
        lines.append(f"### Round Summary")
        lines.append(f"Your round gains/losses: {'+' if my_payoff >= 0 else ''}{my_payoff} {pluralize(my_payoff)}")
        lines.append("")

        return "\n".join(lines)
