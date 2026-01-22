"""
Main experiment runner for Public Goods Game LLM agent simulation with EDSL.

This script orchestrates the complete simulation using EDSL parallel execution:
1. Initializes configuration, environment, agents, and logger
2. Runs the game loop through all rounds with two-phase structure:
   - Phase 1: Contribution (all agents decide simultaneously)
   - Phase 2: Redistribution (all agents decide simultaneously after seeing contributions)
3. Logs all data for analysis

Usage:
    python main.py

The script includes several pre-configured experiments testing different
parameters (framing, punishment, rewards, etc.)
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
api_key = os.getenv('OPENAI_API_KEY')


from config import PGGConfig, CONFIG_BASELINE, CONFIG_OPT_OUT, CONFIG_REWARDS
from environment import PGGEnvironment, RoundState
from agent import create_pgg_agents
from prompt_builder import PromptBuilder
from edsl_client import EDSLGameClient
from logger import ExperimentLogger
from typing import List, Dict


def run_single_game(
    config: PGGConfig,
    game_id: str,
    logger: ExperimentLogger,
    verbose: bool = True
) -> None:
    """Run one complete game with EDSL parallel execution and two-phase structure.

    This is the main game loop that executes all stages of each round:
    1. Phase 1: Contribution (parallel EDSL execution)
    2. Payoff calculation
    3. Phase 2: Redistribution (parallel EDSL execution, if enabled)
    4. Logging

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
        print(f"  Punishment: {config.punishment_enabled}")
        print(f"  Reward: {config.reward_enabled}")
        if config.punishment_enabled or config.reward_enabled:
            print(f"  Peer incentive cost: {config.peer_incentive_cost} coins")
        print(f"{'='*60}\n")

    # Initialize components
    env = PGGEnvironment(config)
    edsl_client = EDSLGameClient(model=config.llm_model, temperature=config.llm_temperature)
    agents = create_pgg_agents(config)
    prompt_builder = PromptBuilder(config)

    # Create agent name mapping
    agent_names = {agent.agent_id: agent.avatar_name for agent in agents}

    # Game loop
    for round_num in range(1, config.game_length + 1):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Round {round_num}/{config.game_length}")
            print(f"{'='*60}\n")

        # ===== PHASE 1: CONTRIBUTION (PARALLEL EDSL EXECUTION) =====
        if verbose:
            print("Phase 1: Contribution")
            print("All agents decide simultaneously...\n")

        # Build contribution contexts for all agents
        contribution_contexts = []
        for agent in agents:
            prompt = prompt_builder.build_contribution_prompt(
                agent.agent_id,
                agent.avatar_name,
                round_num,
                env.round_history,
                [],  # No chat messages (communication_enabled always false)
                agent_names
            )
            contribution_contexts.append(
                agent.prepare_contribution_context(prompt)
            )

        # Execute contributions in parallel via EDSL with two-stage reasoning
        print(f"\n[DEBUG] Running contribution survey with {len(contribution_contexts)} agents, iterations={config.edsl_iterations}")
        all_iterations_data = edsl_client.run_contribution_survey(
            contribution_contexts,
            config.endowment,
            iterations=config.edsl_iterations
        )
        print(f"[DEBUG] Got {len(all_iterations_data)} iteration records")

        # Extract final contributions (last iteration per agent) for game logic
        contributions = {}
        agent_iterations = {}  # Group iterations by agent

        for record in all_iterations_data:
            agent_id = record['agent_id']
            if agent_id not in agent_iterations:
                agent_iterations[agent_id] = []
            agent_iterations[agent_id].append(record)

        # Get last iteration for each agent
        for agent_id, iterations in agent_iterations.items():
            last_iteration = max(iterations, key=lambda x: x['iteration'])
            contributions[agent_id] = last_iteration['decision']

        print(f"[DEBUG] Final contributions (last iteration): {contributions}")

        # Log ALL iterations to raw_responses.csv
        total_logged = 0
        for record in all_iterations_data:
            logger.log_raw_response(
                game_id=game_id,
                round_num=round_num,
                agent_id=record['agent_id'],
                avatar_name=agent_names[record['agent_id']],
                prompt_type=f"contribution_iter{record['iteration']}",
                raw_response=record['reasoning'],
                parsed_result=str(record['decision'])
            )
            total_logged += 1
        print(f"[DEBUG] Logged {total_logged} contribution iteration records")

        # Handle opt-out framing conversion
        if config.contribution_framing == "opt_out":
            contributions = {
                aid: config.endowment - amt
                for aid, amt in contributions.items()
            }

        # Validate contributions
        contributions = {
            aid: env.validate_contribution(amt, aid)
            for aid, amt in contributions.items()
        }

        if verbose:
            print("Contributions:")
            for agent in agents:
                print(f"  {agent.avatar_name}: {contributions[agent.agent_id]} coins")
            print()

        # ===== Calculate Base Payoffs =====
        base_payoffs = env.calculate_payoffs(contributions)

        if verbose:
            total_contrib = sum(contributions.values())
            public_fund = total_contrib * config.multiplier
            print(f"Public Fund Calculation:")
            print(f"  Total contributions: {total_contrib} coins")
            print(f"  After multiplication (×{config.multiplier:.2f}): {public_fund:.2f} coins")
            print(f"  Per-person share: {public_fund / config.group_size:.2f} coins")
            print()

        # ===== PHASE 2: REDISTRIBUTION (PARALLEL EDSL EXECUTION) =====
        punishments = {}
        rewards = {}

        if config.punishment_enabled or config.reward_enabled:
            if verbose:
                print("Phase 2: Redistribution")
                print("All contributions are now visible to all players.\n")

            # Build redistribution contexts for all agents
            redistribution_contexts = []
            for agent in agents:
                current_wallet = base_payoffs[agent.agent_id]

                # Build list of other agents
                other_agents = [
                    {"agent_id": a.agent_id, "avatar_name": a.avatar_name}
                    for a in agents if a.agent_id != agent.agent_id
                ]

                # Build prompt with full visibility and budget constraint
                prompt = prompt_builder.build_redistribution_prompt(
                    agent.agent_id,
                    agent.avatar_name,
                    round_num,
                    contributions,  # Show all contributions
                    other_agents,
                    current_wallet=current_wallet
                )

                redistribution_contexts.append(
                    agent.prepare_redistribution_context(
                        prompt,
                        other_agents,
                        current_wallet
                    )
                )

            # Execute redistribution in parallel via EDSL with two-stage reasoning
            redist_type = "punishment" if config.punishment_enabled else "reward"
            print(f"\n[DEBUG] Running redistribution survey with {len(redistribution_contexts)} agents, type={redist_type}, iterations={config.edsl_iterations}")
            redistribution_results, redistribution_iterations = edsl_client.run_redistribution_survey(
                redistribution_contexts,
                redist_type,
                config,
                iterations=config.edsl_iterations
            )
            print(f"[DEBUG] Got redistribution results: {redistribution_results}")
            print(f"[DEBUG] Got {len(redistribution_iterations)} redistribution iteration records")

            # Log ALL iterations to raw_responses.csv
            total_redist_logged = 0
            for record in redistribution_iterations:
                # Format selected names as string
                selected_str = ", ".join(record['selected_names']) if record['selected_names'] else "(none)"

                logger.log_raw_response(
                    game_id=game_id,
                    round_num=round_num,
                    agent_id=record['agent_id'],
                    avatar_name=agent_names[record['agent_id']],
                    prompt_type=f"{redist_type}_iter{record['iteration']}",
                    raw_response=record['reasoning'],
                    parsed_result=selected_str
                )
                total_redist_logged += 1
            print(f"[DEBUG] Logged {total_redist_logged} {redist_type} iteration records")

            # Assign to correct dict
            if config.punishment_enabled:
                punishments = redistribution_results
                if verbose and punishments:
                    print("Punishment actions:")
                    for (actor_id, target_id), units in punishments.items():
                        actor_name = agent_names[actor_id]
                        target_name = agent_names[target_id]
                        cost = units * config.peer_incentive_cost
                        impact = units * config.punishment_impact
                        print(f"  {actor_name} → {target_name}: {units} unit(s) (cost: {cost}, impact: {impact})")

                # Log detailed redistribution info
                for (actor_id, target_id), units in punishments.items():
                    logger.log_redistribution_detail(
                        game_id=game_id,
                        round_num=round_num,
                        actor_id=actor_id,
                        actor_name=agent_names[actor_id],
                        target_id=target_id,
                        target_name=agent_names[target_id],
                        redist_type="punishment",
                        units_decided=units,
                        units_actual=units,
                        was_scaled=False,
                        cost=units * config.peer_incentive_cost,
                        impact=units * config.punishment_impact
                    )
                print(f"[DEBUG] Logged {len(punishments)} punishment details")
            else:
                rewards = redistribution_results
                if verbose and rewards:
                    print("Reward actions:")
                    for (actor_id, target_id), units in rewards.items():
                        actor_name = agent_names[actor_id]
                        target_name = agent_names[target_id]
                        cost = units * config.peer_incentive_cost
                        impact = units * config.reward_impact
                        print(f"  {actor_name} → {target_name}: {units} unit(s) (cost: {cost}, impact: {impact})")

                # Log detailed redistribution info
                for (actor_id, target_id), units in rewards.items():
                    logger.log_redistribution_detail(
                        game_id=game_id,
                        round_num=round_num,
                        actor_id=actor_id,
                        actor_name=agent_names[actor_id],
                        target_id=target_id,
                        target_name=agent_names[target_id],
                        redist_type="reward",
                        units_decided=units,
                        units_actual=units,
                        was_scaled=False,
                        cost=units * config.peer_incentive_cost,
                        impact=units * config.reward_impact
                    )
                print(f"[DEBUG] Logged {len(rewards)} reward details")

            if not punishments and not rewards and verbose:
                print("  (No punishment/reward actions)")
            if verbose:
                print()

        # Apply punishment/reward adjustments to payoffs
        adjustments = env.apply_redistribution(punishments, rewards, base_payoffs)
        final_payoffs = {
            agent_id: base_payoffs[agent_id] + adjustments[agent_id]
            for agent_id in base_payoffs.keys()
        }

        # ===== Create and store round state =====
        round_state = env.create_round_state(
            round_num=round_num,
            contributions=contributions,
            chat_messages=[],  # No chat in EDSL version
            payoffs=final_payoffs,
            punishments=punishments,
            rewards=rewards
        )

        print(f"\n[DEBUG] Created round state for round {round_num}:")
        print(f"  Contributions: {round_state.contributions}")
        print(f"  Payoffs: {round_state.payoffs}")
        print(f"  Wallets: {round_state.wallets}")
        print(f"  Punishments: {round_state.punishments}")
        print(f"  Rewards: {round_state.rewards}")

        env.add_round_to_history(round_state)

        # ===== Log round =====
        print(f"\n[DEBUG] Calling logger.log_round with game_id={game_id}, round={round_state.round_num}, {len(agent_names)} agents")
        logger.log_round(game_id, round_state, agent_names)
        print(f"[DEBUG] logger.log_round completed successfully")

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

        # Print EDSL usage
        edsl_client.print_usage_summary()


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
    print("  Public Goods Game LLM Agent Simulation (EDSL Version)")
    print("  Testing Context Sensitivity with 14 Design Parameters")
    print("="*70)

    # Ask user which experiment to run
    experiments = {
        "1": ("baseline", CONFIG_BASELINE, "Baseline PGG with punishment"),
        "2": ("optout_framing", CONFIG_OPT_OUT, "Test opt-out framing effect"),
        "3": ("rewards", CONFIG_REWARDS, "Test reward mechanism"),
        "4": ("custom", None, "Custom configuration (manual setup)")
    }

    print("\nAvailable experiments:")
    for key, (name, _, description) in experiments.items():
        print(f"  {key}. {description}")

    choice = input("\nSelect experiment (1-4) or 'all' to run all: ").strip()

    if choice == "all":
        # Run all predefined experiments
        for key in ["1", "2", "3"]:
            name, config, _ = experiments[key]
            run_experiment(name, config, num_games=1, verbose=True)
    elif choice in experiments:
        name, config, _ = experiments[choice]
        if config is None:
            print("\nCustom configuration not yet implemented. Please edit main.py to add your config.")
            return
        run_experiment(name, config, num_games=1, verbose=True)
    else:
        print("Invalid choice. Exiting.")
        return


if __name__ == "__main__":
    # For quick testing, you can uncomment one of these:

    # 1. Quick test (small scale for testing)
    # test_config = PGGConfig(
    #     group_size=3,
    #     game_length=2,
    #     mpcr=0.4,
    #     punishment_enabled=True,
    #     peer_incentive_cost=2
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
    #     punishment_enabled=True,
    #     peer_incentive_cost=2,
    #     punishment_impact=3
    # )
    # run_experiment("my_custom_experiment", custom_config, num_games=1)
