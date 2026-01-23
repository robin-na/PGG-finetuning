"""
EDSL client for parallel PGG agent execution.

This module provides EDSL-based parallel execution of LLM agent decisions,
replacing direct OpenAI API calls with survey-based approach.
"""

import os
from edsl import (
    QuestionNumerical,
    QuestionCheckBox,
    QuestionFreeText,
    QuestionMultipleChoice,
    Survey,
    Model,
    Agent as EDSLAgent,
    AgentList
)
from typing import List, Dict, Tuple, TYPE_CHECKING
import time

# Disable EDSL remote inference - use local API keys
#os.environ['EDSL_RUN_MODE'] = 'local'

if TYPE_CHECKING:
    from config import PGGConfig


class EDSLGameClient:
    """Client for running EDSL surveys with parallel agent execution.

    This class handles:
    - Parallel contribution decisions using QuestionNumerical
    - Parallel redistribution decisions using QuestionCheckBox
    - Budget constraints via max_selections parameter
    - Usage tracking and cost estimation
    """

    def __init__(self, model: str = "gpt-4o", temperature: float = 0.5):
        """Initialize EDSL client.

        Args:
            model: OpenAI model name (e.g., "gpt-4o", "gpt-4")
            temperature: Sampling temperature (0.0 = deterministic, 2.0 = very random)
        """
        self.model_name = model
        self.temperature = temperature
        self.edsl_model = Model(model, temperature=temperature)
        self.total_surveys = 0
        self.total_questions = 0
        self.start_time = time.time()

    def run_contribution_survey(
        self,
        agent_contexts: List[Dict],
        max_value: int,
        iterations: int = 10
    ) -> Dict[str, int]:
        """Execute contribution decisions with two-stage reasoning.

        Stage 1: FreeText reasoning question
        Stage 2: MultipleChoice actual contribution decision
        Uses targeted memory so Stage 2 can see Stage 1 answer.
        Runs n iterations and returns results.

        Args:
            agent_contexts: List of dicts with keys:
                - "agent_id": str - unique agent identifier
                - "avatar_name": str - agent's name (e.g., "Alice")
                - "prompt": str - full contribution prompt for this agent
            max_value: Maximum contribution amount (typically endowment)
            iterations: Number of iterations to run (default 10)

        Returns:
            Dict mapping agent_id to contribution amount (int)

        Example:
            >>> contexts = [
            ...     {"agent_id": "agent_0", "avatar_name": "Alice", "prompt": "You have 20 coins..."},
            ...     {"agent_id": "agent_1", "avatar_name": "Bob", "prompt": "You have 20 coins..."}
            ... ]
            >>> contributions = client.run_contribution_survey(contexts, 20, iterations=10)
            >>> print(contributions)
            {'agent_0': 15, 'agent_1': 10}
        """
        # Step 1: Create EDSL agents with names and traits
        edsl_agents_list = []
        agent_id_list = []

        for ctx in agent_contexts:
            agent_id = ctx["agent_id"]
            avatar_name = ctx["avatar_name"]
            prompt_text = ctx["prompt"]
            agent_id_list.append(agent_id)

            # Create agent with name as direct parameter (not in traits)
            agent = EDSLAgent(
                name=avatar_name,  # Name as direct parameter
                traits={
                    "agent_id": agent_id,
                    "prompt": prompt_text
                }
            )
            edsl_agents_list.append(agent)

        # Create AgentList
        edsl_agents = AgentList(edsl_agents_list)

        # Step 2: Build two-stage survey
        # Question 1: Reasoning (FreeText)
        q1 = QuestionFreeText(
            question_name="contribution_reasoning",
            question_text=(
                "{{prompt}}\n\n"
                "Before you decide, reason through the situation: "
                "What factors should you consider? "
                "How might others respond? "
                "What is your strategy?"
            )
        )

        # Question 2: Actual decision (MultipleChoice)
        contribution_options = [str(v) for v in range(max_value + 1)]
        q2 = QuestionMultipleChoice(
            question_name="contribution_amount",
            question_text=(
                f"Based on your reasoning above, what amount will you contribute? "
                f"Choose a number from 0 to {max_value}."
            ),
            question_options=contribution_options
        )

        # Create survey with targeted memory (q2 can see q1's answer)
        survey = Survey([q1, q2]).add_targeted_memory(q2, q1)

        # Step 3: Execute with iterations
        print(f"[DEBUG EDSL] Running contribution survey with {len(agent_id_list)} agents, {iterations} iterations")
        results = survey.by(edsl_agents).by(self.edsl_model).run(n=iterations)
        print(f"[DEBUG EDSL] Survey completed, got results object")

        # Step 4: Parse results using pandas
        # Convert to DataFrame for easy extraction
        print(f"[DEBUG EDSL] Converting results to pandas...")
        df = results.to_pandas(remove_prefix=True)
        print(f"[DEBUG EDSL] DataFrame shape: {df.shape}")
        print(f"[DEBUG EDSL] DataFrame columns: {df.columns.tolist()}")
        print(f"[DEBUG EDSL] First few rows:\n{df.head()}")

        # Extract ALL iterations for each agent (not just last one)
        # Return full DataFrame for iteration-level logging
        all_iterations_data = []

        for agent_id in agent_id_list:
            # Filter rows for this agent
            agent_rows = df[df['agent_id'] == agent_id]
            print(f"[DEBUG EDSL] Agent {agent_id}: found {len(agent_rows)} rows (iterations)")

            if not agent_rows.empty:
                # Store ALL iterations for this agent
                for _, row in agent_rows.iterrows():
                    iteration_num = int(row.get('iteration', 0))
                    amount_str = str(row['contribution_amount'])

                    all_iterations_data.append({
                        'agent_id': agent_id,
                        'iteration': iteration_num,
                        'reasoning': str(row.get('contribution_reasoning', '')),
                        'decision': int(amount_str)
                    })

                print(f"[DEBUG EDSL] Agent {agent_id}: stored {len(agent_rows)} iterations")
            else:
                # Fallback if no results
                all_iterations_data.append({
                    'agent_id': agent_id,
                    'iteration': 0,
                    'reasoning': '[NO RESULTS]',
                    'decision': max_value // 2
                })
                print(f"[DEBUG EDSL] Agent {agent_id}: NO RESULTS, using fallback")

        self.total_surveys += 1  # Only 1 survey executed
        self.total_questions += len(agent_contexts) * 2  # 2 questions per agent

        print(f"[DEBUG EDSL] Total iteration records: {len(all_iterations_data)}")

        # Return all iterations data for logging
        return all_iterations_data

    def run_redistribution_survey(
        self,
        agent_contexts: List[Dict],
        redist_type: str,
        config: 'PGGConfig',
        iterations: int = 10
    ) -> Dict[Tuple[str, str], int]:
        """Execute redistribution decisions with two-stage reasoning.

        Stage 1: FreeText reasoning question
        Stage 2: CheckBox actual selection decision
        Uses targeted memory so Stage 2 can see Stage 1 answer.
        Runs n iterations and returns results.

        Args:
            agent_contexts: List of dicts with keys:
                - "agent_id": str - unique agent identifier
                - "avatar_name": str - agent's name (e.g., "Alice")
                - "prompt": str - full redistribution prompt
                - "other_agents": List[Dict] - other agents with "agent_id" and "avatar_name"
                - "max_units": int - maximum units this agent can afford
            redist_type: Either "punishment" or "reward"
            config: PGGConfig instance for accessing parameters
            iterations: Number of iterations to run (default 10)

        Returns:
            Dict mapping (actor_id, target_id) to units (always 1 per selection)

        Example:
            >>> contexts = [{
            ...     "agent_id": "agent_0",
            ...     "avatar_name": "Alice",
            ...     "prompt": "Alice contributed 20, Bob contributed 0...",
            ...     "other_agents": [{"agent_id": "agent_1", "avatar_name": "Bob"}],
            ...     "max_units": 3
            ... }]
            >>> results = client.run_redistribution_survey(contexts, "punishment", config, iterations=10)
            >>> print(results)
            {('agent_0', 'agent_1'): 1}
        """
        if not agent_contexts:
            return {}

        # Filter agents with budget
        valid_contexts = [ctx for ctx in agent_contexts if ctx["max_units"] > 0 and ctx["other_agents"]]
        if not valid_contexts:
            return {}

        action_verb = "punish" if redist_type == "punishment" else "reward"
        all_results = {}
        all_iterations_data = []  # Store all iterations for logging

        # Process each agent separately (due to different player lists and budgets)
        for i, ctx in enumerate(valid_contexts):
            agent_id = ctx["agent_id"]
            avatar_name = ctx["avatar_name"]
            other_agents = ctx["other_agents"]
            max_units = ctx["max_units"]
            prompt_text = ctx["prompt"]

            player_options = [a["avatar_name"] for a in other_agents]

            # IMPORTANT: max_selections cannot exceed number of available options
            # Cap max_units at the number of available players
            effective_max_units = min(max_units, len(player_options))

            # Create agent-specific questions
            # Question 1: Reasoning (FreeText)
            q1 = QuestionFreeText(
                question_name=f"{redist_type}_reasoning_{i}",
                question_text=(
                    f"{prompt_text}\n\n"
                    f"Before you decide, reason through: "
                    f"Who do you think deserves to be {action_verb}ed? Why?"
                )
            )

            # Question 2: Actual selection (CheckBox)
            q2 = QuestionCheckBox(
                question_name=f"{redist_type}_targets_{i}",
                question_text=(
                    f"Based on your reasoning above, select which players to {action_verb}:"
                ),
                question_options=player_options,
                max_selections=effective_max_units  # Use capped value
            )

            # Create survey with targeted memory (q2 can see q1's answer)
            survey_agent = Survey([q1, q2]).add_targeted_memory(q2, q1)

            # Create single agent with name as direct parameter
            edsl_agent = EDSLAgent(
                name=avatar_name,  # Name as direct parameter
                traits={
                    "agent_id": agent_id
                }
            )

            # Execute with iterations
            print(f"[DEBUG EDSL] Running {redist_type} survey for agent {agent_id} ({avatar_name}), iterations={iterations}")
            results = survey_agent.by(edsl_agent).by(self.edsl_model).run(n=iterations)
            print(f"[DEBUG EDSL] Survey completed for agent {agent_id}")

            # Parse selections using pandas
            print(f"[DEBUG EDSL] Converting to pandas...")
            df = results.to_pandas(remove_prefix=True)
            print(f"[DEBUG EDSL] DataFrame shape: {df.shape}, columns: {df.columns.tolist()}")

            if not df.empty:
                # Store ALL iterations for this agent
                for _, row in df.iterrows():
                    iteration_num = int(row.get('iteration', 0))
                    selected_names = row[f"{redist_type}_targets_{i}"]
                    reasoning_text = str(row.get(f"{redist_type}_reasoning_{i}", ''))

                    # Convert to list if it's not already
                    if isinstance(selected_names, str):
                        selected_names = [selected_names]
                    elif not isinstance(selected_names, list):
                        selected_names = []

                    # Store iteration data
                    all_iterations_data.append({
                        'agent_id': agent_id,
                        'iteration': iteration_num,
                        'reasoning': reasoning_text,
                        'selected_names': selected_names.copy(),
                        'other_agents': other_agents
                    })

                # Get the LAST iteration's answer for game logic
                last_row = df.iloc[-1]
                selected_names_final = last_row[f"{redist_type}_targets_{i}"]

                # Convert to list if it's not already
                if isinstance(selected_names_final, str):
                    selected_names_final = [selected_names_final]
                elif not isinstance(selected_names_final, list):
                    selected_names_final = []

                # Map names to agent IDs (for game logic)
                # Each selection = 1 unit of punishment/reward
                for other in other_agents:
                    if other["avatar_name"] in selected_names_final:
                        key = (agent_id, other["agent_id"])
                        all_results[key] = 1  # 1 unit per selection

        self.total_surveys += len(valid_contexts)
        self.total_questions += len(valid_contexts) * 2  # 2 questions per agent

        return all_results, all_iterations_data

    def print_usage_summary(self):
        """Print summary of API usage and estimated costs."""
        elapsed_time = time.time() - self.start_time

        print(f"\n{'='*60}")
        print("EDSL API Usage Summary")
        print(f"{'='*60}")
        print(f"Model: {self.model_name}")
        print(f"Total surveys executed: {self.total_surveys}")
        print(f"Total questions processed: {self.total_questions}")
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        print(f"{'='*60}\n")
