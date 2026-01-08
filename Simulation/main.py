"""
Main experiment runner for Public Goods Game LLM agent simulation.

This script orchestrates the complete simulation:
1. Initializes configuration, environment, agents, and logger
2. Runs the game loop through all rounds
3. Handles chat, contribution, and redistribution stages
4. Logs all data for analysis

Usage:
    python main.py

The script includes several pre-configured experiments testing different
parameters (framing, communication, anonymity, etc.)
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
api_key = os.getenv('OPENAI_API_KEY')


from config import PGGConfig, CONFIG_BASELINE, CONFIG_OPT_OUT, CONFIG_COMMUNICATION
from environment import PGGEnvironment, RoundState
from agent import LLMAgent, create_agents
from prompt_builder import PromptBuilder
from llm_client import LLMClient
from logger import ExperimentLogger
from typing import List, Dict


def run_single_game(
    config: PGGConfig,
    game_id: str,
    logger: ExperimentLogger,
    verbose: bool = True
) -> None:
    """Run one complete game with the given configuration.

    This is the main game loop that executes all stages of each round:
    1. Chat stage (if communication enabled)
    2. Contribution stage
    3. Payoff calculation
    4. Redistribution stage (if punishment/reward enabled)
    5. Logging

    Args:
        config: Game configuration
        game_id: Unique identifier for this game
        logger: Logger for saving data
        verbose: Whether to print progress messages
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Starting Game: {game_id}")
        print(f"{'='*60}")
        print(f"Configuration:")
        print(f"  Group size: {config.group_size}")
        print(f"  Rounds: {config.game_length}")
        print(f"  Framing: {config.contribution_framing}")
        print(f"  Communication: {config.communication_enabled}")
        print(f"  Punishment: {config.punishment_enabled}")
        print(f"  Reward: {config.reward_enabled}")
        print(f"{'='*60}\n")

    # Initialize components
    env = PGGEnvironment(config)
    llm_client = LLMClient(model=config.llm_model, temperature=config.llm_temperature)
    agents = create_agents(config, llm_client)
    prompt_builder = PromptBuilder(config)

    # Create agent name mapping
    agent_names = {agent.agent_id: agent.avatar_name for agent in agents}

    # Game loop
    for round_num in range(1, config.game_length + 1):
        if verbose:
            print(f"\n{'='*40}")
            print(f"Round {round_num}/{config.game_length}")
            print(f"{'='*40}\n")

        # ===== Stage 1: Chat (if enabled) =====
        chat_messages = []
        if config.communication_enabled:
            if verbose:
                print("Stage 1: Chat")
            for agent in agents:
                chat_prompt = prompt_builder.build_chat_prompt(
                    agent.agent_id,
                    agent.avatar_name,
                    round_num,
                    chat_messages
                )

                # Save prompt if verbose
                if verbose:
                    logger.save_prompt(game_id, round_num, agent.agent_id, "chat", chat_prompt)

                message, raw_response = agent.get_chat_message(chat_prompt)

                # Log raw response
                logger.log_raw_response(
                    game_id, round_num, agent.agent_id, agent.avatar_name,
                    "chat", raw_response, message
                )

                if message:  # Only add non-empty messages
                    chat_messages.append({
                        "agent_id": agent.agent_id,
                        "avatar_name": agent.avatar_name,
                        "message": message
                    })
                    # Log chat message to CSV
                    logger.log_chat_message(
                        game_id, round_num, agent.agent_id, agent.avatar_name, message
                    )
                    if verbose:
                        print(f"  {agent.avatar_name}: \"{message}\"")

            if not chat_messages and verbose:
                print("  (No messages)")
            if verbose:
                print()

        # ===== Stage 2: Contribution =====
        if verbose:
            print("Stage 2: Contribution")

        contributions = {}
        for agent in agents:
            contrib_prompt = prompt_builder.build_contribution_prompt(
                agent.agent_id,
                agent.avatar_name,
                round_num,
                env.round_history,
                chat_messages,
                agent_names  # Pass agent names for history formatting
            )

            # Save prompt if verbose
            if verbose:
                logger.save_prompt(game_id, round_num, agent.agent_id, "contribution", contrib_prompt)

            amount, raw_response = agent.get_contribution_decision(contrib_prompt)

            # Handle opt-out framing conversion
            if config.contribution_framing == "opt_out":
                # LLM output is withdrawal amount, convert to contribution
                amount = config.endowment - amount

            # Validate
            amount = env.validate_contribution(amount, agent.agent_id)
            contributions[agent.agent_id] = amount

            # Log raw response
            logger.log_raw_response(
                game_id, round_num, agent.agent_id, agent.avatar_name,
                "contribution", raw_response, str(amount)
            )

            if verbose:
                print(f"  {agent.avatar_name}: {amount} coins")

        if verbose:
            print()

        # ===== Stage 3: Calculate Payoffs =====
        base_payoffs = env.calculate_payoffs(contributions)

        if verbose:
            total_contrib = sum(contributions.values())
            public_fund = total_contrib * config.multiplier
            print(f"Total contributions: {total_contrib} coins")
            print(f"Public fund (after multiplication): {public_fund} coins")
            print(f"Per-person share: {public_fund / config.group_size} coins\n")

        # ===== Stage 4: Redistribution (if enabled) =====
        punishments = {}
        rewards = {}

        if config.punishment_enabled or config.reward_enabled:
            if verbose:
                print("Stage 4: Redistribution")

            for agent in agents:
                # Calculate current wallet balance (after contribution stage)
                current_wallet = base_payoffs[agent.agent_id]

                # Build list of other agents
                other_agents = [
                    {"agent_id": a.agent_id, "avatar_name": a.avatar_name}
                    for a in agents if a.agent_id != agent.agent_id
                ]

                # Build prompt with budget constraint
                redist_prompt = prompt_builder.build_redistribution_prompt(
                    agent.agent_id,
                    agent.avatar_name,
                    round_num,
                    contributions,
                    other_agents,
                    current_wallet=current_wallet  # NEW: pass wallet balance
                )

                # Save prompt if verbose
                if verbose:
                    logger.save_prompt(game_id, round_num, agent.agent_id, "redistribution", redist_prompt)

                amounts_decided, raw_response = agent.get_redistribution_decision(
                    redist_prompt,
                    len(other_agents)
                )

                # Log raw response
                logger.log_raw_response(
                    game_id, round_num, agent.agent_id, agent.avatar_name,
                    "redistribution", raw_response, str(amounts_decided)
                )

                # Calculate total cost
                total_cost = 0
                if config.punishment_enabled:
                    total_cost += sum(amounts_decided) * config.punishment_cost
                if config.reward_enabled:
                    total_cost += sum(amounts_decided) * config.reward_cost

                # Apply proportional scaling if over budget (Solution 2)
                was_scaled = False
                amounts_actual = amounts_decided.copy()

                if total_cost > current_wallet:
                    if total_cost > 0:
                        scaling_factor = current_wallet / total_cost
                        # Scale down and floor to integers
                        amounts_actual = [int(amt * scaling_factor) for amt in amounts_decided]
                        was_scaled = True
                        if verbose:
                            print(f"  {agent.avatar_name}: Budget exceeded ({total_cost:.2f} > {current_wallet:.2f}), scaled by {scaling_factor:.3f}")

                # Map amounts to target agents and log details
                for idx, target_agent in enumerate(other_agents):
                    units_decided = amounts_decided[idx]
                    units_actual = amounts_actual[idx]

                    if units_decided > 0 or units_actual > 0:  # Log even if scaled to 0
                        target_id = target_agent["agent_id"]
                        target_name = target_agent["avatar_name"]

                        if config.punishment_enabled:
                            key = (agent.agent_id, target_id)
                            if units_actual > 0:
                                punishments[key] = units_actual

                            # Log punishment detail
                            logger.log_redistribution_detail(
                                game_id, round_num,
                                agent.agent_id, agent.avatar_name,
                                target_id, target_name,
                                "punishment",
                                units_decided, units_actual, was_scaled,
                                units_actual * config.punishment_cost,
                                units_actual * config.punishment_impact
                            )

                            if verbose and units_actual > 0:
                                scaled_marker = " (scaled)" if was_scaled and units_decided != units_actual else ""
                                print(f"  {agent.avatar_name} → {target_name}: {units_actual} punishment units{scaled_marker}")

                        if config.reward_enabled:
                            key = (agent.agent_id, target_id)
                            if units_actual > 0:
                                rewards[key] = units_actual

                            # Log reward detail
                            logger.log_redistribution_detail(
                                game_id, round_num,
                                agent.agent_id, agent.avatar_name,
                                target_id, target_name,
                                "reward",
                                units_decided, units_actual, was_scaled,
                                units_actual * config.reward_cost,
                                units_actual * config.reward_impact
                            )

                            if verbose and units_actual > 0:
                                scaled_marker = " (scaled)" if was_scaled and units_decided != units_actual else ""
                                print(f"  {agent.avatar_name} → {target_name}: {units_actual} reward units{scaled_marker}")

            if not punishments and not rewards and verbose:
                print("  (No redistribution actions)")
            if verbose:
                print()

        # Apply redistribution to payoffs
        adjustments = env.apply_redistribution(punishments, rewards, base_payoffs)
        final_payoffs = {
            agent_id: base_payoffs[agent_id] + adjustments[agent_id]
            for agent_id in base_payoffs.keys()
        }

        # ===== Create and store round state =====
        round_state = env.create_round_state(
            round_num=round_num,
            contributions=contributions,
            chat_messages=chat_messages,
            payoffs=final_payoffs,
            punishments=punishments,
            rewards=rewards
        )

        env.add_round_to_history(round_state)

        # ===== Log round =====
        logger.log_round(game_id, round_state, agent_names)

        # ===== Print round summary =====
        if verbose:
            print("Round Summary:")
            for agent in agents:
                payoff = final_payoffs[agent.agent_id]
                wallet = round_state.wallets[agent.agent_id]
                print(f"  {agent.avatar_name}: Round payoff = {payoff:+.1f}, Total wallet = {wallet:.1f}")

    # ===== Game complete =====
    if verbose:
        print(f"\n{'='*60}")
        print(f"Game {game_id} Complete!")
        print(f"{'='*60}")
        print("Final Standings:")
        sorted_agents = sorted(agents, key=lambda a: round_state.wallets[a.agent_id], reverse=True)
        for i, agent in enumerate(sorted_agents, 1):
            wallet = round_state.wallets[agent.agent_id]
            print(f"  {i}. {agent.avatar_name}: {wallet:.1f} coins")

        # Print LLM usage
        llm_client.print_usage_summary()


def run_experiment(
    experiment_name: str,
    config: PGGConfig,
    num_games: int = 1,
    verbose: bool = True
):
    """Run multiple games with the same configuration.

    Args:
        experiment_name: Name for this experiment (used in output directory)
        config: Game configuration
        num_games: Number of games to run
        verbose: Whether to print progress
    """
    if verbose:
        print(f"\n{'#'*70}")
        print(f"# EXPERIMENT: {experiment_name}")
        print(f"{'#'*70}\n")

    # Create logger
    with ExperimentLogger(experiment_name, config) as logger:
        for game_idx in range(num_games):
            game_id = f"{experiment_name}_game{game_idx+1}"
            run_single_game(config, game_id, logger, verbose=verbose)

        if verbose:
            print(f"\n{'#'*70}")
            print(f"# Experiment '{experiment_name}' complete!")
            print(f"# Data saved to: {logger.base_dir}")
            print(f"{'#'*70}\n")


# ===== Main Entry Point =====

def main():
    """Main entry point for running experiments.

    This runs several pre-configured experiments to demonstrate different
    design parameters and test context sensitivity.
    """
    print("\n" + "="*70)
    print("  Public Goods Game LLM Agent Simulation")
    print("  Testing Context Sensitivity with 14 Design Parameters")
    print("="*70)

    # Ask user which experiment to run
    experiments = {
        "1": ("baseline", CONFIG_BASELINE, "Baseline PGG with punishment"),
        "2": ("optout_framing", CONFIG_OPT_OUT, "Test opt-out framing effect"),
        "3": ("communication", CONFIG_COMMUNICATION, "Test communication effect"),
        "4": ("custom", None, "Custom configuration (manual setup)")
    }

    print("\nAvailable experiments:")
    for key, (name, _, description) in experiments.items():
        print(f"  {key}. {description}")

    choice = input("\nSelect experiment (1-4) or 'all' to run all: ").strip()

    if choice == "all":
        # Run all predefined experiments
        for key in ["1", "2", "3"]:
            name, config, desc = experiments[key]
            run_experiment(name, config, num_games=1, verbose=True)
    elif choice in experiments:
        name, config, desc = experiments[choice]
        if config is None:
            print("\nCustom configuration not yet implemented. Please edit main.py to add your config.")
            return
        run_experiment(name, config, num_games=1, verbose=True)
    else:
        print("Invalid choice. Exiting.")
        return


if __name__ == "__main__":
    # For quick testing, you can uncomment one of these:

    # 1. Baseline experiment (small scale for testing)
    # test_config = PGGConfig(
    #     group_size=3,
    #     game_length=3,
    #     mpcr=0.4,
    #     punishment_enabled=True
    # )
    # run_experiment("quick_test", test_config, num_games=1, verbose=True)

    # 2. Run the interactive menu
    main()

    # 3. Or define your own experiment here:
    # custom_config = PGGConfig(
    #     group_size=4,
    #     game_length=5,
    #     mpcr=0.5,
    #     contribution_framing="opt_out",
    #     communication_enabled=True,
    #     punishment_enabled=True,
    #     peer_outcome_visibility=False  # Hidden outcomes
    # )
    # run_experiment("my_custom_experiment", custom_config, num_games=1)
