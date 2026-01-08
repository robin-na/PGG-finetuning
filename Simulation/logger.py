"""
Experiment logging system for PGG simulations.

This module provides logging functionality to save:
- Game data to CSV (matching existing data format)
- Configuration to JSON
- Prompts for reproducibility
"""

import csv
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict
from config import PGGConfig
from environment import RoundState


class ExperimentLogger:
    """Logger for PGG experiments.

    Manages file output including:
    - CSV file with round-by-round data
    - JSON configuration file
    - Individual prompt files (optional)
    """

    def __init__(self, experiment_id: str, config: PGGConfig, output_dir: str = "experiments"):
        """Initialize experiment logger.

        Args:
            experiment_id: Unique identifier for this experiment
            config: Game configuration
            output_dir: Base directory for experiment outputs

        Creates directory structure:
            experiments/
                {experiment_id}/
                    config.json
                    game_log.csv
                    prompts/  (optional)
        """
        self.experiment_id = experiment_id
        self.config = config
        self.base_dir = Path(output_dir) / experiment_id
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Generate config hash for tracking
        self.config_hash = self._hash_config(config)

        # Save configuration
        self._save_config()

        # Initialize CSV file
        self._init_csv()

    def _hash_config(self, config: PGGConfig) -> str:
        """Generate hash of configuration for tracking.

        Args:
            config: Configuration to hash

        Returns:
            str: First 8 characters of SHA256 hash
        """
        # Convert config to sorted JSON string for consistent hashing
        config_str = json.dumps(config.__dict__, sort_keys=True)
        hash_obj = hashlib.sha256(config_str.encode())
        return hash_obj.hexdigest()[:8]

    def _save_config(self):
        """Save configuration to JSON file."""
        config_data = {
            "experiment_id": self.experiment_id,
            "config_hash": self.config_hash,
            "timestamp": datetime.now().isoformat(),
            "config": {
                # Game structure
                "group_size": self.config.group_size,
                "game_length": self.config.game_length,
                "endowment": self.config.endowment,
                "horizon_knowledge": self.config.horizon_knowledge,
                # Economic
                "mpcr": self.config.mpcr,
                "multiplier": self.config.multiplier,
                # Contribution
                "contribution_type": self.config.contribution_type,
                "contribution_framing": self.config.contribution_framing,
                # Social information
                "communication_enabled": self.config.communication_enabled,
                "peer_outcome_visibility": self.config.peer_outcome_visibility,
                "actor_anonymity": self.config.actor_anonymity,
                # Punishment
                "punishment_enabled": self.config.punishment_enabled,
                "punishment_cost": self.config.punishment_cost,
                "punishment_impact": self.config.punishment_impact,
                # Reward
                "reward_enabled": self.config.reward_enabled,
                "reward_cost": self.config.reward_cost,
                "reward_impact": self.config.reward_impact,
                # LLM
                "llm_model": self.config.llm_model,
                "llm_temperature": self.config.llm_temperature
            }
        }

        with open(self.base_dir / "config.json", "w") as f:
            json.dump(config_data, f, indent=2)

    def _init_csv(self):
        """Initialize CSV files with headers."""
        # Main game log
        self.csv_file = open(self.base_dir / "game_log.csv", "w", newline='')
        self.csv_writer = csv.DictWriter(
            self.csv_file,
            fieldnames=[
                "experiment_id",
                "config_hash",
                "game_id",
                "round",
                "agent_id",
                "avatar_name",
                "contribution",
                "punishments_sent",
                "punishments_received",
                "rewards_sent",
                "rewards_received",
                "round_payoff",
                "cumulative_wallet",
                "public_fund",
                "timestamp"
            ]
        )
        self.csv_writer.writeheader()

        # Chat messages log
        self.chat_file = open(self.base_dir / "chat_messages.csv", "w", newline='')
        self.chat_writer = csv.DictWriter(
            self.chat_file,
            fieldnames=[
                "experiment_id",
                "config_hash",
                "game_id",
                "round",
                "agent_id",
                "avatar_name",
                "message",
                "timestamp"
            ]
        )
        self.chat_writer.writeheader()

        # Raw LLM responses log
        self.response_file = open(self.base_dir / "raw_responses.csv", "w", newline='')
        self.response_writer = csv.DictWriter(
            self.response_file,
            fieldnames=[
                "experiment_id",
                "config_hash",
                "game_id",
                "round",
                "agent_id",
                "avatar_name",
                "prompt_type",
                "raw_response",
                "parsed_result",
                "timestamp"
            ]
        )
        self.response_writer.writeheader()

        # Redistribution details log (NEW)
        self.redistribution_file = open(self.base_dir / "redistribution_details.csv", "w", newline='')
        self.redistribution_writer = csv.DictWriter(
            self.redistribution_file,
            fieldnames=[
                "experiment_id",
                "config_hash",
                "game_id",
                "round",
                "actor_id",
                "actor_name",
                "target_id",
                "target_name",
                "type",
                "units_decided",
                "units_actual",
                "was_scaled",
                "cost",
                "impact",
                "timestamp"
            ]
        )
        self.redistribution_writer.writeheader()

    def log_round(
        self,
        game_id: str,
        round_state: RoundState,
        agent_names: Dict[str, str]  # agent_id -> avatar_name
    ):
        """Log a completed round to CSV.

        Args:
            game_id: Identifier for this game
            round_state: The completed round state
            agent_names: Mapping from agent_id to avatar_name
        """
        # Calculate punishment/reward stats per agent
        punishments_sent = {agent_id: 0 for agent_id in round_state.contributions.keys()}
        punishments_received = {agent_id: 0 for agent_id in round_state.contributions.keys()}
        rewards_sent = {agent_id: 0 for agent_id in round_state.contributions.keys()}
        rewards_received = {agent_id: 0 for agent_id in round_state.contributions.keys()}

        # Count punishments
        for (punisher_id, target_id), units in round_state.punishments.items():
            punishments_sent[punisher_id] += units
            punishments_received[target_id] += units

        # Count rewards
        for (rewarder_id, target_id), units in round_state.rewards.items():
            rewards_sent[rewarder_id] += units
            rewards_received[target_id] += units

        # Write row for each agent
        for agent_id in round_state.contributions.keys():
            self.csv_writer.writerow({
                "experiment_id": self.experiment_id,
                "config_hash": self.config_hash,
                "game_id": game_id,
                "round": round_state.round_num,
                "agent_id": agent_id,
                "avatar_name": agent_names.get(agent_id, "UNKNOWN"),
                "contribution": round_state.contributions.get(agent_id, 0),
                "punishments_sent": punishments_sent.get(agent_id, 0),
                "punishments_received": punishments_received.get(agent_id, 0),
                "rewards_sent": rewards_sent.get(agent_id, 0),
                "rewards_received": rewards_received.get(agent_id, 0),
                "round_payoff": round_state.payoffs.get(agent_id, 0),
                "cumulative_wallet": round_state.wallets.get(agent_id, 0),
                "public_fund": round_state.public_fund,
                "timestamp": datetime.now().isoformat()
            })

        # Flush to disk
        self.csv_file.flush()

    def log_chat_message(
        self,
        game_id: str,
        round_num: int,
        agent_id: str,
        avatar_name: str,
        message: str
    ):
        """Log a chat message to CSV.

        Args:
            game_id: Identifier for this game
            round_num: Round number
            agent_id: Agent identifier
            avatar_name: Agent's avatar name
            message: The chat message
        """
        self.chat_writer.writerow({
            "experiment_id": self.experiment_id,
            "config_hash": self.config_hash,
            "game_id": game_id,
            "round": round_num,
            "agent_id": agent_id,
            "avatar_name": avatar_name,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
        self.chat_file.flush()

    def log_raw_response(
        self,
        game_id: str,
        round_num: int,
        agent_id: str,
        avatar_name: str,
        prompt_type: str,
        raw_response: str,
        parsed_result: str
    ):
        """Log a raw LLM response to CSV.

        Args:
            game_id: Identifier for this game
            round_num: Round number
            agent_id: Agent identifier
            avatar_name: Agent's avatar name
            prompt_type: Type of prompt (contribution/chat/redistribution)
            raw_response: The raw LLM response text
            parsed_result: The parsed/extracted value
        """
        self.response_writer.writerow({
            "experiment_id": self.experiment_id,
            "config_hash": self.config_hash,
            "game_id": game_id,
            "round": round_num,
            "agent_id": agent_id,
            "avatar_name": avatar_name,
            "prompt_type": prompt_type,
            "raw_response": raw_response,
            "parsed_result": parsed_result,
            "timestamp": datetime.now().isoformat()
        })
        self.response_file.flush()

    def log_redistribution_detail(
        self,
        game_id: str,
        round_num: int,
        actor_id: str,
        actor_name: str,
        target_id: str,
        target_name: str,
        redist_type: str,  # "punishment" or "reward"
        units_decided: int,
        units_actual: int,
        was_scaled: bool,
        cost: float,
        impact: float
    ):
        """Log a single redistribution action detail to CSV.

        Args:
            game_id: Identifier for this game
            round_num: Round number
            actor_id: Agent performing the action
            actor_name: Actor's avatar name
            target_id: Target agent
            target_name: Target's avatar name
            redist_type: Type of action ("punishment" or "reward")
            units_decided: Units originally decided by LLM
            units_actual: Units actually applied (after scaling)
            was_scaled: Whether scaling was applied
            cost: Total cost to actor
            impact: Total impact on target
        """
        self.redistribution_writer.writerow({
            "experiment_id": self.experiment_id,
            "config_hash": self.config_hash,
            "game_id": game_id,
            "round": round_num,
            "actor_id": actor_id,
            "actor_name": actor_name,
            "target_id": target_id,
            "target_name": target_name,
            "type": redist_type,
            "units_decided": units_decided,
            "units_actual": units_actual,
            "was_scaled": was_scaled,
            "cost": cost,
            "impact": impact,
            "timestamp": datetime.now().isoformat()
        })
        self.redistribution_file.flush()

    def save_prompt(
        self,
        game_id: str,
        round_num: int,
        agent_id: str,
        prompt_type: str,
        prompt: str
    ):
        """Save a prompt to file for reproducibility.

        Args:
            game_id: Game identifier
            round_num: Round number
            agent_id: Agent identifier
            prompt_type: Type of prompt (e.g., "contribution", "redistribution", "chat")
            prompt: The full prompt text
        """
        prompt_dir = self.base_dir / "prompts"
        prompt_dir.mkdir(exist_ok=True)

        filename = f"{game_id}_r{round_num}_{agent_id}_{prompt_type}.txt"
        with open(prompt_dir / filename, "w") as f:
            f.write(prompt)

    def close(self):
        """Close log files."""
        if hasattr(self, 'csv_file') and self.csv_file:
            self.csv_file.close()
        if hasattr(self, 'chat_file') and self.chat_file:
            self.chat_file.close()
        if hasattr(self, 'response_file') and self.response_file:
            self.response_file.close()
        if hasattr(self, 'redistribution_file') and self.redistribution_file:
            self.redistribution_file.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# ===== Testing / Demo =====
if __name__ == "__main__":
    from config import PGGConfig
    from environment import PGGEnvironment, RoundState

    print("Testing ExperimentLogger")
    print("=" * 60)

    # Create a test config
    config = PGGConfig(
        group_size=3,
        game_length=2,
        mpcr=0.4,
        punishment_enabled=True
    )

    # Create logger
    with ExperimentLogger("test_experiment", config, output_dir="test_output") as logger:
        print(f"Created logger for experiment: {logger.experiment_id}")
        print(f"Output directory: {logger.base_dir}")
        print(f"Config hash: {logger.config_hash}")

        # Create a mock round state
        round_state = RoundState(
            round_num=1,
            contributions={"agent_0": 10, "agent_1": 15, "agent_2": 20},
            chat_messages=[],
            public_fund=54.0,  # 45 * 1.2
            payoffs={"agent_0": 28.0, "agent_1": 23.0, "agent_2": 18.0},
            punishments={("agent_1", "agent_2"): 1},
            rewards={},
            wallets={"agent_0": 28.0, "agent_1": 23.0, "agent_2": 18.0}
        )

        agent_names = {
            "agent_0": "DOG",
            "agent_1": "CAT",
            "agent_2": "BIRD"
        }

        # Log the round
        logger.log_round("game_001", round_state, agent_names)
        print("\nLogged round 1")

        # Save a test prompt
        test_prompt = "You have 20 coins. How much do you contribute?"
        logger.save_prompt("game_001", 1, "agent_0", "contribution", test_prompt)
        print("Saved test prompt")

    print(f"\nCheck output at: {Path('test_output/test_experiment').absolute()}")
