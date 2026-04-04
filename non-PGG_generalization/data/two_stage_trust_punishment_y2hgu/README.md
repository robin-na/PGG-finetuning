# Two-Stage Trust and Punishment/Helping Dataset (OSF: y2hgu)

## Source
- **OSF Project**: [https://osf.io/y2hgu/](https://osf.io/y2hgu/)
- **Paper**: "Deliberating Cost and Impact: Trustworthiness Signals in Punishment and Helping" (Communications Psychology)
- **Recruitment Platform**: Prolific (UK, English fluent, approval rate = 100%, opted into "studies that may involve deception")

## Study Overview

This dataset examines whether **the process of making prosocial decisions** (not just the outcome) serves as a trustworthiness signal. Across 5 experiments, participants play a two-stage game:

- **Stage 1**: Player A decides whether to help or punish a third party, and may optionally **deliberate** (check cost/impact information) before deciding.
- **Stage 2**: Player A and Player B play a trust game. In the "observable" condition, Player B can see Player A's deliberation process (whether they checked, or how fast they decided).

The key hypothesis: appearing "uncalculating" (not deliberating) when being prosocial signals genuine motivation, making Player A seem more trustworthy.

## Experimental Design

### The 5 Experiments

| Experiment | File | Context | What "deliberation" means |
|-----------|------|---------|---------------------------|
| E1 | `helpcostcheckE1.csv` | Helping | Checking the **personal cost** of helping |
| E2 | `puncostcheckE2.csv` | Punishment | Checking the **personal cost** of punishing |
| E3 | `puntimeE3a.csv` + `puntimeE3b.csv` | Punishment | **Decision speed** (fast vs. slow) |
| E4 | `helpimpactcheckE4.csv` | Helping | Checking the **impact on target** of helping |
| E5 | `punimpactcheckE5.csv` | Punishment | Checking the **impact on target** of punishing |

### Between-Subjects Conditions (2x2 for E1/E2/E4/E5)

| Condition Code | Role | Process Visibility |
|---------------|------|-------------------|
| 3 | Player A | Observable (B will see A's deliberation) |
| 4 | Player A | Hidden (B will NOT see A's deliberation) |
| 5 | Player B | Observable (B CAN see how A deliberated) |
| 6 | Player B | Hidden (B CANNOT see how A deliberated) |

### Trust Game (Stage 2)
- Player B has 10 pence and decides how much (0–10) to send to Player A
- The sent amount is **tripled**
- Player A returns a percentage (0–100%) of the tripled amount to Player B
- Player B uses a **strategy method**: makes conditional decisions for each scenario (A helped/punished or not × deliberated or not)

## Data Files

### Column Descriptions (E1/E2/E4/E5 files, semicolon-delimited)

| Column | Description | Values |
|--------|-------------|--------|
| `PID` | Participant ID | string |
| `Duration` | Time taken (seconds) | integer |
| `A1comp1`–`A1comp4` | Stage 1 comprehension checks (Player A) | 0/1 |
| `A2comp1`–`A2comp3` | Stage 2 comprehension checks (Player A) | 0/1 |
| `A2comp4OB` | Comprehension check, observable condition | 0/1 |
| `A2comp4HID` | Comprehension check, hidden condition | 0/1 |
| `checkObs` | Did A check cost/impact in observable condition? | 0/1 |
| `checkHid` | Did A check cost/impact in hidden condition? | 0/1 |
| `calcHelp` / `calcPun` | A helped/punished AFTER checking | 0/1 |
| `uncalcHelp` / `uncalcPun` | A helped/punished WITHOUT checking | 0/1 |
| `return` | A's return in trust game (% of tripled amount) | 0–100 |
| `helpUncalc` / `punUncalc` | Pence B sends when A acted without checking | 0–10 |
| `helpCalc` / `punCalc` | Pence B sends when A acted after checking | 0–10 |
| `noUncalc` | Pence B sends when A did NOT act, without checking | 0–10 |
| `noCalc` | Pence B sends when A did NOT act, after checking | 0–10 |
| `helpedHidB` / `punHidB` | Pence B sends in hidden condition when A acted | 0–10 |
| `noHelpdHidB` / `noPunHidB` | Pence B sends in hidden condition when A did NOT act | 0–10 |
| `gender` | Gender | 1=male, 2=female, 3/4=other |
| `age` | Age in years | integer |
| `Condition` | Treatment condition | 3/4/5/6 |
| `RecordedDate` | Date of participation | datetime |

### E3 Files (Decision Time Experiment)
- **`puntimeE3a.csv`** (Player A, N=1,306): Contains `decisionT` (decision time in seconds), `punishing` (0/1), `return` (0–100%)
- **`puntimeE3b.csv`** (Player B, N=~1,306): Contains conditional trust decisions: `punFast`, `punSlow`, `noFast`, `noSlow`, `punHid`, `noPunHid` (all 0–10 pence)

### Supporting Files
- `R Code/`: Analysis scripts for each experiment + cross-experiment comparisons
- `Laboratory Log.docx`: Experiment administration log
- `*.docx`: Paper manuscript and supplementary information

## Demographics
- **Age**: Yes (continuous)
- **Gender**: Yes (1=male, 2=female, 3/4=other)
- **Education**: Not collected

## Sample
- Each experiment: ~2,600 participants (~1,300 Player A–B pairs)
- 5 experiments total
- UK residents, English fluent, Prolific approval rate = 100%

---

## LLM Simulation Prompts

This study requires two separate prompts — one for Player A and one for Player B — matched within each experiment.

### Player A Prompt (Help Cost Check variant — E1)

```
You are participating in an economic decision-making study with two stages.

**Stage 1: Helping Decision**
Another participant (Player C) has been unfairly treated in a prior interaction. You have the option to help Player C, which would restore some of their lost earnings.

Before deciding, you can choose to CHECK the personal cost of helping, or you can decide WITHOUT checking.

{If observable condition:}
Important: Player B (whom you will interact with in Stage 2) will be able to see whether or not you checked the cost before deciding.

{If hidden condition:}
Note: Player B (whom you will interact with in Stage 2) will NOT see whether you checked the cost.

Question 1: Do you want to CHECK the personal cost of helping before deciding?
Please answer: YES or NO

Question 2: Do you want to HELP Player C?
Please answer: YES or NO

**Stage 2: Trust Game**
You now play a trust game with Player B. Player B will send you some amount (0–10 pence), which is tripled. You then decide what percentage of the tripled amount to return to Player B.

Question 3: What percentage of the tripled amount will you return to Player B?
Please answer with a number from 0 to 100.
```

### Player B Prompt (Observable Condition, Strategy Method)

```
You are participating in an economic decision-making study.

In a prior stage, Player A had the option to help another participant who was treated unfairly. Before deciding, Player A could choose to check or not check the personal cost of helping.

You will now play a trust game with Player A. You have 10 pence. You can send any amount (0–10 pence) to Player A. Whatever you send will be TRIPLED. Player A will then return some percentage of the tripled amount to you.

Please indicate how much you would send (0–10 pence) in each of the following scenarios:

1. Player A HELPED without checking the cost: ___
2. Player A HELPED after checking the cost: ___
3. Player A DID NOT HELP without checking the cost: ___
4. Player A DID NOT HELP after checking the cost: ___

Please respond with four integers, each between 0 and 10.
```

### Adaptation for Other Experiments
- **E2/E5 (Punishment)**: Replace "help" with "punish a norm violator" and adjust framing
- **E3 (Decision Time)**: Replace "checked/did not check" with "decided quickly/decided slowly"
- **E4 (Impact Check)**: Replace "check the personal cost" with "check the impact on the target"

### Persona Prefix for Demographic Conditioning
```
You are a {age}-year-old {gender} living in the United Kingdom.
```
