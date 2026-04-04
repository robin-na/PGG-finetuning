# Multi-Game Battery with LLM Delegation Dataset (OSF: fvk2c)

## Source
- **OSF Project**: [https://osf.io/fvk2c/](https://osf.io/fvk2c/)
- **Paper**: "Adverse Reactions to the Use of Large Language Models in Social Interactions" by Dvorak, Stumpf, Fehrler & Fischbacher (PNAS Nexus, 2025)
- **Recruitment Platform**: Prolific (sample described as "close to UK population" in age/gender/education)
- **Ethics**: University of Konstanz / GfEW IRB approval

## Study Overview

This dataset investigates how people react when their interaction partner in economic games **delegates decisions to an AI (ChatGPT)**. Participants play **5 classic economic games** across 7 roles, with a between-subjects manipulation of how AI delegation is introduced (transparent vs. opaque, voluntary vs. random, personalized vs. unpersonalized).

A separate Turing-test experiment evaluates whether human raters can distinguish AI-generated decisions/justifications from human ones.

## Experimental Design

### The 5 Economic Games

#### 1. Ultimatum Game (UG) — 2 Roles
- **Proposer**: Has 10 ECU, offers 0–10 ECU to the Responder
- **Responder**: States minimum acceptable offer (0–10 ECU, strategy method)
- If offer >= threshold: split is realized. Otherwise both get 0.

#### 2. Trust Game (TG) — 2 Roles
- Both start with 5 ECU
- **Sender**: Decides whether to transfer 2 ECU (binary: trust/no trust). Transfer is **tripled** to 6 ECU.
- **Receiver**: Decides how much of the 6 ECU to return (0–6 ECU, strategy method)

#### 3. Prisoner's Dilemma (PD) — Symmetric
| | Partner: A (Cooperate) | Partner: B (Defect) |
|---|---|---|
| **You: A (Cooperate)** | 5, 5 | 1, 8 |
| **You: B (Defect)** | 8, 1 | 3, 3 |

#### 4. Stag Hunt (SH) — Symmetric
| | Partner: X (Cooperate) | Partner: Y (Defect) |
|---|---|---|
| **You: X (Cooperate)** | 8, 8 | 1, 5 |
| **You: Y (Defect)** | 5, 1 | 4, 4 |

#### 5. Coordination Game (C) — Symmetric
- Both players choose from: Mercury, Venus, Earth, Mars, Saturn
- Same choice: 5 ECU each; Different choice: 2 ECU each
- "Earth" is the focal point

### 6 Treatment Conditions (3 × 2 Between-Subjects)

#### Dimension 1: Delegation Type
| Treatment | Description |
|-----------|-------------|
| **TransparentRandom (TR)** | One player is randomly replaced by AI; both players know |
| **TransparentDelegation (TD)** | Players can voluntarily delegate to AI; opponent knows if delegated |
| **OpaqueDelegation (OD)** | Players can voluntarily delegate to AI; opponent does NOT know |

#### Dimension 2: AI Personalization
| Treatment | Description |
|-----------|-------------|
| **Personalized (P)** | AI prompted with participant's 7 binary personality traits |
| **Unpersonalized (U)** | AI prompted generically ("Pretend you are a person") |

#### Treatment Codes
| Code | Full Name |
|------|-----------|
| TRP | Transparent Random + Personalized |
| TRU | Transparent Random + Unpersonalized |
| TDP | Transparent Delegation + Personalized |
| TDU | Transparent Delegation + Unpersonalized |
| ODP | Opaque Delegation + Personalized |
| ODU | Opaque Delegation + Unpersonalized |

### 7 Binary Personality Traits (for AI Personalization)
1. Intuition vs. Thoughtfulness
2. Introversion vs. Extraversion
3. Fairness vs. Efficiency
4. Chaos vs. Boredom
5. Selfishness vs. Altruism
6. Novelty vs. Reliability
7. Truth vs. Harmony

### Payment
- All payoffs in ECU (1 ECU = £0.50)
- Base participation fee: £2
- One game scenario randomly selected for bonus payment
- Randomly matched with another participant in the same treatment

## Data Files

### `Package/data/MainDataRawClean.csv` (Main Experiment, ~27,920 rows)
Panel structure: multiple rows per participant (one per scenario within their treatment).

**Key columns** (see `DICTIONARY.txt` for full list):

| Column | Description | Values |
|--------|-------------|--------|
| `SubjectID` | Participant ID | integer |
| `Treatment` | Treatment group | "OpaqueDelegation", "TransparentDelegation", "TransparentRandom" |
| `PersonalizedTreatment` | AI personalization | 0/1 |
| `Proposer` | UG offer | 0–10 |
| `Responder` | UG min. acceptance | 0–10 |
| `Sender` | TG trust decision | 0/1 |
| `Receiver` | TG return amount | 0–6 |
| `PD` | PD cooperation | 0/1 |
| `SH` | SH cooperation | 0/1 |
| `C` | Coordination choice | -2 to 2 (Mercury to Saturn) |
| `*_Delegation` | Delegation decision for each role | 0/1 |
| `Age` | Age in years | integer |
| `Gender` | Gender | "Man", "Woman", "Non-binary", "Self-description" |
| `Education` | Highest qualification | string |
| `KnowledgeChatGPT` | Knows ChatGPT | "Yes"/"No" |
| `UsageChatGPT` | ChatGPT usage frequency | categorical |
| `Q_AITrustworthy` | AI trustworthiness rating | 5-point Likert |
| `Q_DelegationAppropriate` | Delegation appropriateness | 5-point Likert |
| `personality_string` | 7-digit binary string | e.g., "2112222" |

### `Package/data/TuringDataRawClean.csv` (Turing Test, ~30,177 rows)
Raters evaluate pairs of human vs. AI statements per game situation.

| Column | Description |
|--------|-------------|
| `RatingIsAI` | Correctly identified AI? (0/1) |
| `Cert` | Confidence (0–4) |
| `Situation` | Game role ("UG_P", "TG_S", "PD", etc.) |
| `AI_JUS` / `Human_JUS` | Justification texts shown |

### `Package/data/AIData.csv`
All ChatGPT decisions across 128 personalized + 64 unpersonalized personality types for 10 game roles.

### `Package/data/{ODP,ODN,T}_statements/`
Turing test stimulus pairs (human and AI statements with justifications) per game.

### `Package/DICTIONARY.txt`
Full variable dictionary for all data files.

### `Package/*.R`
- `Main.R`: Master script
- `Config.R`: Package dependencies and settings
- `Data.R`: Variable construction and data cleaning (507 lines)
- `Analysis.R`: Statistical analyses

## Demographics Available
- **Age**: Yes
- **Gender**: Yes (Man/Woman/Non-binary/Self-description)
- **Education**: Yes (highest qualification)
- **AI familiarity**: ChatGPT knowledge and usage frequency
- **Personality**: 7 binary trait dimensions
- **AI attitudes**: 9 Likert-scale items on AI trust, delegation, equality

## Sample
- Main experiment: 2,947 participants after exclusions (~490 per treatment)
- Turing test: 655 raters
- Recruited via Prolific; demographics close to UK population

---

## LLM Simulation Prompts

### Prompt Template for Each Game

The following prompts replicate the 5 economic games. For the full study, each LLM agent should complete all 7 role-decisions.

#### 1. Ultimatum Game — Proposer

```
You are participating in an economic decision-making study.

**Ultimatum Game (Proposer)**
You have 10 ECU (Experimental Currency Units). You must propose how to split this amount between yourself and another participant.

- You offer some amount (0–10 ECU) to the other participant and keep the rest.
- The other participant has set a minimum acceptable amount. If your offer is at least that amount, the split goes through. If your offer is below their minimum, BOTH of you receive 0 ECU.

1 ECU = £0.50. One game will be randomly selected to determine your real bonus payment.

How much do you offer to the other participant?
Please respond with a single integer from 0 to 10.
```

#### 2. Ultimatum Game — Responder

```
You are participating in an economic decision-making study.

**Ultimatum Game (Responder)**
Another participant has 10 ECU and will propose a split. You must set your minimum acceptable offer.

- If their offer meets or exceeds your minimum, the split goes through (you receive their offer, they keep the rest).
- If their offer is below your minimum, BOTH of you receive 0 ECU.

You are setting your minimum BEFORE seeing the actual offer.

What is your minimum acceptable offer?
Please respond with a single integer from 0 to 10.
```

#### 3. Trust Game — Sender

```
You are participating in an economic decision-making study.

**Trust Game (Sender)**
Both you and another participant start with 5 ECU. You can choose to transfer 2 ECU to the other participant. If you transfer:

- Your 2 ECU will be TRIPLED to 6 ECU and given to the other participant.
- The other participant then decides how much of the 6 ECU to return to you.

If you do NOT transfer, both of you keep your 5 ECU.

Do you want to transfer 2 ECU?
Please respond: YES or NO.
```

#### 4. Trust Game — Receiver

```
You are participating in an economic decision-making study.

**Trust Game (Receiver)**
Both you and another participant start with 5 ECU. The other participant may transfer 2 ECU to you. If they do, their 2 ECU is TRIPLED to 6 ECU.

If they transfer, you receive the 6 ECU (on top of your 5 ECU) and decide how much of the 6 ECU to return to them.

Assuming the other participant transfers, how much of the 6 ECU would you return?
Please respond with a single integer from 0 to 6.
```

#### 5. Prisoner's Dilemma

```
You are participating in an economic decision-making study.

**Prisoner's Dilemma**
You and another participant each choose A or B. Payoffs (in ECU) depend on both choices:

- Both choose A: You get 5, they get 5
- You choose A, they choose B: You get 1, they get 8
- You choose B, they choose A: You get 8, they get 1
- Both choose B: You get 3, they get 3

What do you choose: A or B?
Please respond with a single letter: A or B.
```

#### 6. Stag Hunt

```
You are participating in an economic decision-making study.

**Stag Hunt**
You and another participant each choose X or Y. Payoffs (in ECU):

- Both choose X: You get 8, they get 8
- You choose X, they choose Y: You get 1, they get 5
- You choose Y, they choose X: You get 5, they get 1
- Both choose Y: You get 4, they get 4

What do you choose: X or Y?
Please respond with a single letter: X or Y.
```

#### 7. Coordination Game

```
You are participating in an economic decision-making study.

**Coordination Game**
You and another participant each independently choose one planet: Mercury, Venus, Earth, Mars, or Saturn.

- If you BOTH choose the same planet: You each earn 5 ECU.
- If you choose DIFFERENT planets: You each earn 2 ECU.

Which planet do you choose?
Please respond with one word: Mercury, Venus, Earth, Mars, or Saturn.
```

### Delegation Decision Prompt (for TD/OD treatments)

```
Before making your decision in this game, you have the option to DELEGATE your decision to an AI assistant (ChatGPT).

{If personalized:}
The AI has been configured with your personality profile and will try to make decisions that match your preferences.

{If unpersonalized:}
The AI will make a generic decision on your behalf.

{If transparent:}
Note: The other participant will know whether you delegated to AI or made the decision yourself.

{If opaque:}
Note: The other participant will NOT know whether you delegated to AI or made the decision yourself.

Do you want to delegate this decision to the AI?
Please respond: YES or NO.
```

### Persona Prefix for Demographic Conditioning

```
You are a {age}-year-old {gender} living in the United Kingdom. Your highest education level is {education}. {If knows ChatGPT:} You are familiar with ChatGPT and have used it {usage_frequency}. Your personality profile: you prefer {trait1} over {alt1}, {trait2} over {alt2}, ... (7 traits).
```
