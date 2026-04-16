# Pipeline Overview

The chip-bargain benchmark now uses a hybrid structure:

- dataset adapter: [`../datasets/chip_bargain.py`](../datasets/chip_bargain.py)
- prompt builder: [`../prompts/chip_bargain.py`](../prompts/chip_bargain.py)
- dedicated batch builder: [`./build_batch_inputs.py`](./build_batch_inputs.py)
- chip-bargain-specific Twin assignment: [`./profile_sampling/sample_twin_personas_for_chip_bargain.py`](./profile_sampling/sample_twin_personas_for_chip_bargain.py)
- parse/eval stack:
  - [`./common.py`](./common.py)
  - [`./parse_outputs.py`](./parse_outputs.py)
  - [`./evaluate_outputs.py`](./evaluate_outputs.py)
  - [`./analyze_vs_human_treatments.py`](./analyze_vs_human_treatments.py)
  - [`./compare_models_with_noise_ceiling.py`](./compare_models_with_noise_ceiling.py)

Current pipeline status:

1. load raw chip-bargain JSON game logs
2. convert each cohort-stage game into one canonical benchmark record
3. optionally attach one sampled Twin persona per player for the unadjusted Twin variant
4. build a prompt that asks for the full 9-turn game trajectory
5. write batch inputs and metadata
6. parse model outputs back into structured bargaining trajectories
7. re-simulate both human and generated trajectories through the same state update logic
8. score paper-aligned outcome distributions within treatment cell

Current limitation:

- only `baseline` and `twin_sampled_unadjusted_seed_0` are enabled
- adjusted demographic variants remain disabled because standard demographic fields are not currently available in the checked chip-bargain source files
- the raw data do not always preserve one fixed proposer order across all rounds, while the current prompt text still describes a repeated fixed order
