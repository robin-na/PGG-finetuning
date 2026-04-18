---
license: cc-by-nc-sa-4.0
task_categories:
- multiple-choice
language:
- en
size_categories:
- 1K<n<10K
tags:
- llm
- human behavior
- simulation
- benchmark
- social science
- behavioral science
- decision-making
- self-assessment
- judgment
- problem-solving
configs:
- config_name: default
  data_files:
  - split: SimBenchPop
    path: "SimBenchPop.csv"
  - split: SimBenchGrouped
    path: "SimBenchGrouped.csv"

---

# SimBench: A Large-Scale Benchmark for Simulating Human Behavior

**SimBench** is the first large-scale benchmark designed to evaluate how well Large Language Models (LLMs) can simulate group-level human behaviors across diverse settings and tasks. As LLM-based simulations gain traction in social and behavioral sciences, SimBench offers a robust tool to understand their capabilities and limitations.

**Paper:** https://arxiv.org/abs/2510.17516

**Project Website:** http://simbench.tiancheng.hu/

**GitHub Repository:** https://github.com/pitehu/SimBench_release

Simulations of human behavior based on LLMs have the potential to revolutionize the social and behavioral sciences, *if and only if* they faithfully reflect real human behaviors. Prior work across many disciplines has evaluated the simulation capabilities of specific LLMs in specific experimental settings, but often produced disparate results. To move towards a more robust understanding, SimBench compiles 20 datasets in a unified format, measuring diverse types of behavior (e.g., decision-making vs. self-assessment) across hundreds of thousands of diverse participants (e.g., from different parts of the world).

The benchmark is designed to help answer fundamental questions regarding when, how, and why LLM simulations succeed or fail.

## Dataset Description

Simulations of human behavior based on LLMs have the potential to revolutionize the social and behavioral sciences, *if and only if* they faithfully reflect real human behaviors. Prior work across many disciplines has evaluated the simulation capabilities of specific LLMs in specific experimental settings, but often produced disparate results. To move towards a more robust understanding, SimBench compiles 20 datasets in a unified format, measuring diverse types of behavior (e.g., decision-making vs. self-assessment) across hundreds of thousands of diverse participants (e.g., from different parts of the world).

The benchmark is designed to help answer fundamental questions regarding when, how, and why LLM simulations succeed or fail.

### Key Features:
*   **20 Diverse Datasets:** Covering a wide range of human behaviors including decision-making, self-assessment, judgment, and problem-solving.
*   **Global Participant Diversity:** Data from participants across at least 130 countries, representing various cultural and socioeconomic backgrounds.
*   **Unified Format:** All datasets are processed into a consistent structure, facilitating easy use and comparison.
*   **Group-Level Focus:** Evaluates simulation of aggregated human response distributions.
*   **Permissively Licensed:** Enabling broad accessibility and use.

### Dataset Splits

SimBench provides two main splits for evaluation (available as configurations):

1.  **`SimBenchPop` (Population-level Simulation):**
    *   **Content:** Covers questions from all 20 datasets (7,167 test cases).
    *   **Grouping:** Persona prompts are based on the general population of each source dataset.
    *   **Purpose:** Measures the ability of LLMs to simulate responses of broad and diverse human populations.

2.  **`SimBenchGrouped` (Demographically-Grouped Simulation):**
    *   **Content:** Focuses on 5 large-scale survey datasets (AfroBarometer, ESS, ISSP, LatinoBarometro, OpinionQA), with questions selected for significant variation across demographic groups (6,343 test cases).
    *   **Grouping:** Persona prompts specify particular participant sociodemographics (e.g., age, gender, ideology).
    *   **Purpose:** Measures the ability of LLMs to simulate responses from specific participant groups.


## Data Fields

Each instance in the dataset contains the following primary fields:

*   `dataset_name` (string): The name of the original dataset (e.g., "OpinionQA", "WisdomOfCrowds").
*   `group_prompt_template` (string): A template string for constructing the persona/grouping prompt. This template may contain placeholders (e.g., `{age_group}`).
    *   For `SimBenchPop`, this template often represents a default population (e.g., "You are an Amazon Mechanical Turk worker from the United States.").
    *   For `SimBenchGrouped`, this template is designed to incorporate specific demographic attributes.
*   `group_prompt_variable_map` (dict): A dictionary mapping placeholder variables in `group_prompt_template` to their specific values for the instance.
    *   For `SimBenchPop`, this is often an empty dictionary (`{}`) if the template is self-contained.
    *   For `SimBenchGrouped`, this contains the demographic attributes and their values (e.g., `{"age_group": "30-49", "country": "Kenya"}`).
    *   The final persona prompt is constructed by formatting `group_prompt_template` with `group_prompt_variable_map`.
*   `input_template` (string): The text of the question presented to participants/LLMs. This is typically the question stem.
*   `human_answer` (dict): A dictionary representing the aggregated human response distribution for the given question and group.
    *   Keys are the option labels (e.g., "A", "B", "1", "2").
    *   Values are the proportions of human respondents who chose that option (e.g., `{"A": 0.25, "B": 0.50, ...}`).
*   `group_size` (int): The number of human respondents contributing to the `human_answer` distribution for this specific instance.
*   `auxiliary` (dict): A dictionary containing additional metadata from the original dataset. Contents vary by dataset but may include:
    *   `task_id` or `question_id_original`: Original identifier for the question/task.
    *   `correct_answer`: The correct option label, if applicable (e.g., for problem-solving tasks).
    *   Other dataset-specific information.


## How to Use

You can load SimBench using the `datasets` library. Since the data is stored in `.pkl` files, ensure you have a local copy of the dataset loading script (typically `SimBench.py` or the name of your dataset on the Hub) in your working directory or Python path if `datasets` cannot automatically find it for this format.

```python
from datasets import load_dataset

# Load the 'default' configuration which contains both SimBenchPop and SimBenchGrouped
simbench_data_dict = load_dataset("pitehu/SimBench", name="default")

# Access SimBenchPop
simbench_pop = simbench_data_dict["SimBenchPop"]

# Access SimBenchGrouped
simbench_grouped = simbench_data_dict["SimBenchGrouped"]

# Example: Accessing the first instance in SimBenchPop
instance = simbench_pop[0]
print(instance)


