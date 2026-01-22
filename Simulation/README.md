# Public Goods Game LLM Agent Simulation

A fully configurable simulation system for testing how LLM agents behave in Public Goods Games under different rule configurations. This implementation tests **Context Sensitivity** - how the same LLM responds differently based on game framing, information visibility, and incentive structures.

## Overview

This simulation implements all 14 design parameters from the paper, allowing systematic testing of:
- **Contribution framing** (opt-in vs opt-out)
- **Information visibility** (what agents can see about others)
- **Incentive mechanisms** (punishment and rewards)
- **Communication** (chat between agents)
- **Economic parameters** (MPCR, group size, game length)

## Features

✅ **Complete Parameter Coverage**: All 14 design parameters implemented
✅ **Context-Sensitive Prompts**: Different prompts based on configuration
✅ **Robust Parsing**: Handles varied LLM output formats
✅ **Comprehensive Logging**: CSV output matching existing data format
✅ **Cost Tracking**: Monitor OpenAI API usage and costs
✅ **Modular Architecture**: Easy to extend and modify

## Installation

### Prerequisites

- Python 3.8+
- OpenAI API key

### Setup

```bash
# Navigate to simulation directory
cd Simulation

# Install dependencies
pip install openai

# Set your OpenAI API key
export OPENAI_API_KEY='your-api-key-here'
```

## Quick Start

### Basic Usage

```bash
python main.py
```

This will present an interactive menu:
```
Available experiments:
  1. Baseline PGG with punishment
  2. Test opt-out framing effect
  3. Test communication effect
  4. Custom configuration (manual setup)

Select experiment (1-4) or 'all' to run all:
```

### Running a Specific Experiment

```python
from config import PGGConfig
from main import run_experiment

# Define configuration
config = PGGConfig(
    group_size=4,
    game_length=10,
    mpcr=0.4,
    contribution_framing="opt_out",  # Key parameter
    punishment_enabled=True
)

# Run experiment
run_experiment("my_experiment", config, num_games=1)
```

## Configuration Parameters

### 1. Game Structure

```python
group_size: int = 4          # Number of players (2-20)
game_length: int = 10        # Number of rounds (1-30)
endowment: int = 20          # Coins per player per round
horizon_knowledge: "known" | "unknown"  # Show total rounds?
```

### 2. Economic Parameters

```python
mpcr: float = 0.4           # Marginal Per Capita Return (0.06-0.7)
multiplier: float           # Auto-calculated: mpcr * group_size
```

### 3. Contribution Mechanism

```python
contribution_type: "variable" | "all_or_nothing"
contribution_framing: "opt_in" | "opt_out"
```

**Framing Effect:**
- **opt-in**: "You have 20 coins in your private fund. How much do you move to the public fund?"
- **opt-out**: "You have 20 coins in the public fund. How much do you withdraw to your private fund?"

### 4. Social Information

```python
communication_enabled: bool = False      # Chat allowed?
peer_outcome_visibility: bool = True     # Show detailed peer outcomes?
actor_anonymity: bool = False           # Hide punishment/reward sources?
```

### 5. Incentive Mechanisms

```python
# Punishment
punishment_enabled: bool = False
punishment_cost: int = 1        # Cost to punisher per unit (1-4)
punishment_impact: int = 3      # Deduction to target per unit (1-4)

# Reward
reward_enabled: bool = False
reward_cost: int = 1           # Cost to rewarder per unit (1-4)
reward_impact: float = 1.0     # Addition to target per unit (0.5-1.5)
```

### 6. LLM Settings

```python
llm_model: str = "gpt-4"
llm_temperature: float = 1.0
```

#### Backend Selection

By default, the simulation uses OpenAI. To switch providers, set `LLM_BACKEND` or enable
auto routing with `LLM_BACKEND_STRATEGY=auto` (or `LLM_BACKEND=auto`):

```bash
# OpenAI (default)
export OPENAI_API_KEY="your-api-key"

# Auto routing (OpenAI for gpt-4/gpt-4o/gpt-3.5 prefixes, otherwise vLLM if healthy, else HF)
export LLM_BACKEND_STRATEGY="auto"
export OPENAI_API_KEY="your-api-key"
export VLLM_BASE_URL="http://localhost:8000/v1"
export HF_MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="your-hf-token"

# vLLM (OpenAI-compatible server)
export LLM_BACKEND="vllm"
export VLLM_BASE_URL="http://localhost:8000/v1"
# Optional: VLLM_API_KEY (if your server requires it)

# Hugging Face Inference API
export LLM_BACKEND="hf"
export HF_MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="your-hf-token"
# Optional: local PEFT adapter path (requires transformers + peft)
export HF_PEFT_PATH="/path/to/adapter"
```

The `LLMClient` API is unchanged; you can continue using `LLMClient(model=..., temperature=...)`. For Hugging Face, `model` defaults to `HF_MODEL_ID` unless you pass one explicitly. For vLLM, `model` is forwarded to the server (e.g., the model name you started vLLM with).

## Architecture

### Module Overview

```
Simulation/
├── main.py              # Experiment runner and game loop
├── config.py            # Configuration dataclass (14 parameters)
├── environment.py       # Game engine (payoffs, redistribution)
├── agent.py             # LLM agent class
├── prompt_builder.py    # Context-sensitive prompt construction
├── llm_client.py       # OpenAI API wrapper
├── response_parser.py   # Parse LLM outputs
└── logger.py           # CSV/JSON logging
```

### Data Flow

```
PGGConfig → PGGEnvironment → PromptBuilder → LLMAgent → LLMClient (GPT-4)
                ↓                                          ↓
           GameState ← ResponseParser ← LLM Response ← OpenAI API
                ↓
           Logger (CSV)
```

## Example Experiments

### 1. Test Framing Effect

```python
from config import PGGConfig
from main import run_experiment

# Opt-in framing
config_optin = PGGConfig(
    group_size=4,
    game_length=10,
    contribution_framing="opt_in",
    punishment_enabled=True
)
run_experiment("optin_test", config_optin)

# Opt-out framing (mathematically equivalent, different prompt)
config_optout = PGGConfig(
    group_size=4,
    game_length=10,
    contribution_framing="opt_out",
    punishment_enabled=True
)
run_experiment("optout_test", config_optout)
```

**Hypothesis**: Agents may contribute differently due to loss aversion framing effects.

### 2. Test Communication Effect

```python
config_chat = PGGConfig(
    group_size=4,
    game_length=10,
    communication_enabled=True,  # Enable chat
    punishment_enabled=True
)
run_experiment("communication_test", config_chat)
```

**Hypothesis**: Communication enables coordination, increasing contributions.

### 3. Test Anonymity Effect

```python
config_revealed = PGGConfig(
    punishment_enabled=True,
    actor_anonymity=False  # Show who punished
)

config_anonymous = PGGConfig(
    punishment_enabled=True,
    actor_anonymity=True   # Hide punisher identity
)
```

**Hypothesis**: Revealed identity may reduce punishment (fear of retaliation).

### 4. Test Horizon Effect

```python
config_known = PGGConfig(
    horizon_knowledge="known"    # "Round 5 of 10"
)

config_unknown = PGGConfig(
    horizon_knowledge="unknown"  # "Round 5"
)
```

**Hypothesis**: Unknown horizon may increase cooperation (no end-game defection).

## Output Data

### Directory Structure

```
experiments/
    {experiment_name}/
        config.json          # Full configuration
        game_log.csv        # Round-by-round data
        raw_responses.csv   # Raw LLM outputs with prompt_type labels
        chat_messages.csv   # Per-round chat messages
        redistribution_details.csv  # Punishment/reward action details
        prompts/            # Individual prompts (optional)
            {game_id}_r{round}_{agent_id}_{type}.txt
```

### Raw Responses Prompt Types

`raw_responses.csv` includes a `prompt_type` column with values such as `chat`, `contribution`,
and `punishment_reward`. The punishment/reward prompt type was previously logged as
`redistribution`; update any readers that filter on `redistribution` to also accept
`punishment_reward` when working with newer logs.

### Validation Config Manifest

The validation experiment configs are collected into a single manifest at:

```
data/validation_configs.json
```

Each entry uses the normalized schema below:

```json
{
  "experiment_id": "VALIDATION_1_C",
  "config": {
    "group_size": 3,
    "game_length": 16,
    "endowment": 20,
    "horizon_knowledge": "known",
    "mpcr": 0.67,
    "multiplier": 2,
    "contribution_type": "variable",
    "contribution_framing": "opt_in",
    "communication_enabled": false,
    "peer_outcome_visibility": true,
    "actor_anonymity": true,
    "punishment_enabled": false,
    "punishment_cost": 1,
    "punishment_impact": 2,
    "reward_enabled": false,
    "reward_cost": 1,
    "reward_impact": 1,
    "llm_model": "gpt-4o",
    "llm_temperature": 1.0
  }
}
```

Generate or refresh the manifest with:

```bash
python scripts/export_validation_configs.py
```

Run the full validation batch directly from the manifest (each `experiment_id`
becomes its output directory under `experiments/`):

```bash
python run_validation_experiments.py --manifest-path data/validation_configs.json
```

To reduce console output while running the batch:

```bash
python run_validation_experiments.py --manifest-path data/validation_configs.json --quiet
```

### CSV Format

```csv
experiment_id,config_hash,game_id,round,agent_id,avatar_name,contribution,
punishments_sent,punishments_received,rewards_sent,rewards_received,
round_payoff,cumulative_wallet,public_fund,timestamp
```

This format matches the existing data structure from `player-rounds.csv`.

## Cost Estimation

**Approximate costs per experiment:**
- 4 agents × 10 rounds × 2-3 API calls per agent per round ≈ 80-120 calls
- Using GPT-4: ~$0.045 per 1K tokens
- Typical prompt: ~500 tokens, response: ~50 tokens
- **Estimated cost per experiment: $2-5 USD**

For open-source backends (vLLM or Hugging Face), the client does not assign any per-token cost (cost remains $0 in the usage summary).

To track costs in real-time:
```python
# After running experiment, the LLM client prints usage summary
# Example output:
# ==================================================
# LLM API Usage Summary
# ==================================================
# Model: gpt-4
# Total API calls: 84
# Total tokens used: 42,150
# Estimated cost: $1.90 USD
# ==================================================
```

## Advanced Usage

### Custom Experiment Suite

```python
from config import PGGConfig
from main import run_experiment

# Define parameter sweep
mpcr_values = [0.2, 0.4, 0.6]
framing_types = ["opt_in", "opt_out"]

for mpcr in mpcr_values:
    for framing in framing_types:
        config = PGGConfig(
            group_size=4,
            game_length=10,
            mpcr=mpcr,
            contribution_framing=framing,
            punishment_enabled=True
        )

        exp_name = f"mpcr{mpcr}_framing{framing}"
        run_experiment(exp_name, config, num_games=3)  # Run 3 replications
```

### Accessing Agent Memory

```python
# After running a game, you can inspect agent decisions
from agent import create_agents
from llm_client import LLMClient
from config import PGGConfig

config = PGGConfig(group_size=4)
client = LLMClient()
agents = create_agents(config, client)

# After simulation...
for agent in agents:
    print(agent.get_memory_summary())
```

## Validation

### Testing Payoff Calculations

```bash
# Run environment tests
python environment.py
```

Expected output:
```
Testing PGG Environment
============================================================
Config: 4 players, multiplier=1.6
--- Round 1 ---
Contributions: {'agent_0': 10, 'agent_1': 15, 'agent_2': 20, 'agent_3': 5}
Payoffs (before redistribution): {...}
```

### Testing Response Parsing

```bash
# Run parser tests
python response_parser.py
```

## Troubleshooting

### Issue: "OpenAI API key not found"

**Solution:**
```bash
export OPENAI_API_KEY='your-key-here'
```

### Issue: "Rate limit exceeded"

**Solution:** The system has built-in exponential backoff. If persistent:
1. Reduce `group_size` or `game_length`
2. Add delays between experiments
3. Use GPT-3.5-turbo instead of GPT-4

### Issue: "Could not parse response"

**Solution:** The system logs warnings and uses fallback defaults. Check:
- `experiments/{name}/prompts/` for actual prompts sent
- Agent memory for raw LLM responses
- Consider adjusting prompt format or using temperature=0.7 for more consistent outputs

## Extending the System

### Adding New Parameters

1. **Add to PGGConfig** (`config.py`):
```python
@dataclass
class PGGConfig:
    my_new_parameter: bool = False
```

2. **Implement engine logic** (`environment.py`):
```python
def calculate_payoffs(self, contributions):
    if self.config.my_new_parameter:
        # New logic
```

3. **Implement agent perception** (`prompt_builder.py`):
```python
def build_scenario_description(self):
    if self.config.my_new_parameter:
        lines.append("New rule explanation...")
```

### Using Different LLMs

```python
# Currently supports OpenAI API
# To add Anthropic Claude or other providers:
# 1. Extend LLMClient class
# 2. Add new API integration
# 3. Update config.llm_model parameter
```

## References

- Baseline GRU model: `../baseline_gru_set.py`
- Prompt generation patterns: `../alsobay2025publicGoodsGame/generate_prompts.py`
- Experimental configurations: `../data/exp_config_files/validation.yaml`

## Citation

If you use this simulation system, please cite:
```
[Your paper citation here]
```

## License

[Your license here]

## Contact

For questions or issues, please [open an issue on GitHub / contact information].
