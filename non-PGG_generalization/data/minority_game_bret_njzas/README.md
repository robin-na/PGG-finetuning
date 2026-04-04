# Minority Game with BRET Risk Elicitation Dataset (OSF: njzas)

## Source
- **OSF Project**: [https://osf.io/njzas/](https://osf.io/njzas/)
- **Experiment Code**: GitHub (herding-game, oTree)
- **Analysis Code**: GitHub (herding-analysis)
- **Recruitment Platform**: Prolific (UK residents, opted into "studies that may involve deception")
- **Ethics**: ETH Zurich Ethics Commission (2022-N-37)

## Study Overview

This dataset captures behavior in a **repeated minority game** disguised as a "bonus game" (a weak Prisoner's Dilemma with a bonus mechanism that rewards minority-strategy choices). Participants play **11 rounds** against **20 scripted bots** (though participants in the deception condition believe they are playing with real people). The study also includes a **BRET (Bomb Risk Elicitation Task)** measuring individual risk preferences.

The core research question is whether participants exhibit **herding behavior** (following the majority) or **contrarian/minority-seeking behavior**, and how this relates to risk preferences.

## Experimental Design

### The "Bonus Game" (Minority Game)

Each round, participants choose **A** or **B**. The game is framed as a group interaction with 20 other players, but in reality the "other players" are entirely simulated — the participant plays a **solo sequential decision against a deterministic pot schedule**. The pot (reward pool) at each round depends **only on the previous round's choice**, making it a Markov decision process.

#### Deterministic Pot Schedule

| Round | Pot if prev = A | Pot if prev = B |
|-------|----------------|-----------------|
| 1 | 0 (always) | 0 (always) |
| 2 | 84 | 20 |
| 3 | 88 | 40 |
| 4 | 92 | 60 |
| 5 | 96 | 80 |
| 6 | **100** | **100** |
| 7 | 104 | 120 |
| 8 | 108 | 140 |
| 9 | 112 | 160 |
| 10 | 116 | 180 |
| 11 | 116 | 180 |

**Critical crossover at Round 6**: Both choices yield pot = 100. Before Round 6, choosing A leads to a higher pot next round. After Round 6, choosing B leads to a higher pot. The optimal strategy is therefore `AAAAABBBBB*` (expected pot = 124.0 per paying round).

After each round, participants see the current pot value and simulated information about what "other players" chose. A 5-second artificial wait simulates other players deciding.

### 11 Rounds
- Participants make one A/B decision per round
- Between rounds, they observe the pot value and simulated group choice distribution
- One round from rounds 2–11 is randomly selected for payment (Round 1 always has pot = 0)

### Deception Treatment
- `in_deception = 1`: Participant believes they are playing with real humans (actually bots)
- `in_deception = 0`: Non-deception condition (participant may know about bots)
- Post-experiment debriefing reveals the deception

### BRET (Bomb Risk Elicitation Task)
After the bonus game, participants complete a BRET:
- A grid of 100 boxes, one containing a hidden bomb
- Participants choose how many boxes to collect (0–100)
- Each collected box earns points, BUT if the bomb is in a collected box, all earnings are lost
- `boxes_collected` serves as a continuous risk preference measure (more boxes = more risk-seeking)

### Payment
- `session.config.real_world_currency_per_point = 0.005` (1 point = £0.005)
- `session.config.participation_fee = £0.60`
- One round from the bonus game is randomly selected for payment
- Average total ~£9.57/hour

## Data Files

### `experiment_data/all_apps_wide-2022-08-31.csv` (Main Data, N=2,500 rows)

| Column | Description | Values |
|--------|-------------|--------|
| `participant.code` | Unique participant ID | string |
| `participant.finished` | Completed the study? | 0/1 |
| `participant.in_deception` | Deception treatment | 0/1 |
| `participant.payoff` | Total payoff in points | numeric |
| `introduction.1.player.q1`–`q5` | Comprehension check answers | varies |
| `bonus_game.{1-11}.player.decision` | Round decision | "A" or "B" |
| `bonus_game.{1-11}.player.time_spent` | Decision time (seconds) | numeric |
| `bonus_game.{1-11}.player.payoff` | Payoff from this round (if selected) | numeric |
| `bonus_game.{1-11}.player.potential_payoff` | Cumulative potential payoff | numeric |
| `bonus_game.{1-11}.player.in_deception` | Deception status for this round | 0/1 |
| `bret.1.player.boxes_collected` | Number of BRET boxes opened (risk measure) | 0–100 |
| `bret.1.player.bomb` | Whether bomb was in collected boxes | 0/1 |
| `bret.1.player.bomb_row` | Bomb location (row) | integer |
| `bret.1.player.bomb_col` | Bomb location (column) | integer |
| `bret.1.player.pay_this_round` | Whether BRET was selected for payment | 0/1 |
| `bret.1.player.round_result` | BRET payoff (boxes if no bomb, 0 if bomb) | numeric |
| `debrief.1.player.debrief` | Self-reported strategy | "only_c", "minority", "random", "majority", "sophisticated", "only_d" |
| `debrief.1.player.debrief2` | Perceived strategy of others | same categories as above |

### `experiment_data/prolific_export_62fcafdbdaec84519e0c272b.csv` (Demographics, N=1,999 rows)

Rich demographic data from Prolific:

| Column | Description |
|--------|-------------|
| `Participant id` | Prolific participant ID |
| `Age` | Age in years |
| `Sex` | Male/Female |
| `Ethnicity simplified` | Ethnic group |
| `Country of birth` | Country of birth |
| `Country of residence` | Country of residence |
| `Nationality` | Nationality |
| `Language` | Primary language |
| `Student status` | Student/Non-student |
| `Employment status` | Employment category |
| `Highest education level completed` | Education level |
| `Charitable giving` | Charitable giving behavior |
| `Negotiation experience` | Negotiation experience level |
| `Total approvals` | Prolific approval count |
| `Total rejections` | Prolific rejection count |
| `Approval rate` | Prolific approval rate |
| `Deception` | Opted into deception studies |

**Note**: The two files must be joined via participant ID (Prolific PID in session data linked to `Participant id` in prolific export).

### `experiment_example/`
- `game_sequence_screenshots.pdf`: Screenshots of the experimental interface
- `bret.mov`: Video of BRET task interface
- `waiting_for_players.mov`: Video of waiting room interface

## Demographics Available
- **Age**: Yes
- **Sex**: Yes
- **Ethnicity**: Yes
- **Education**: Yes
- **Employment**: Yes
- **Student status**: Yes
- **Country of birth/residence**: Yes
- **Charitable giving, Negotiation experience**: Yes
- **Prolific quality metrics**: Total approvals, rejections, approval rate

## Sample
- 2,500 rows total; 2,003 finished participants
- Deception condition: 1,011 finished (in_deception=1), 992 finished (in_deception=0)
- UK residents, Prolific platform
- Demographics: mean age 39.1 (range 18–84), 50/50 male/female, 88% White

## Observed Behavioral Patterns
- A-rate (cooperation) starts at ~71% in Round 1, declines to ~51% by Round 11
- Self-reported strategies: 41% "only cooperate", 14% "minority", 13% "random", 12% "majority", 10% "sophisticated", 9% "only defect"
- BRET boxes collected: mean=49.6, median=50 (risk-neutral benchmark = 50)
- Deception has negligible effect on behavior (~0.3% A-rate difference)

---

## LLM Simulation Prompts

### Minority Game (Bonus Game) — Per-Round Prompt

The original study frames this as a group interaction. To replicate the participant's experience faithfully, use the group framing (Version A). To test behavior under transparent mechanics, use the solo framing (Version B).

**Version A: Group Framing (as participants experienced it)**

For **Round 1** (no prior information):

```
You are participating in an economic decision-making study. You are in a group with 20 other players.

Each round, you and the other players simultaneously choose either A or B. Your earnings depend on what you and others choose:

- You earn a base amount plus a bonus for each opponent who chose the SAME action as you
- However, the bonus per person is HIGHER when fewer opponents chose your action
- This means choosing the less popular option (the "minority" choice) can be more profitable overall

Your total payoff from one randomly selected round will be converted to real money (1 point = £0.005).

This is Round 1. You have no information about what others have chosen in previous rounds.

What do you choose: A or B?

Please respond with a single letter: A or B.
```

For **Rounds 2–11** (with history):

```
You are in a group with 20 other players. Each round, you simultaneously choose A or B. The minority choice yields higher total payoff.

Here is what happened in previous rounds:
{For each previous round:}
- Round {N}: {count_A} players chose A, {count_B} players chose B. You chose {your_choice}. The pot was {pot} points.

This is Round {current_round}.

What do you choose: A or B?

Please respond with a single letter: A or B.
```

**Version B: Solo Framing (transparent mechanics)**

```
You are participating in a sequential decision task with 11 rounds. Each round you choose A or B.

The reward pot for each round depends on your PREVIOUS round's choice:
- If you chose A last round, the pot this round is approximately {80 + 4*round} points
- If you chose B last round, the pot this round is approximately {20*round} points
- These converge at Round 6 (pot=100 either way), then B becomes more profitable

One round (from rounds 2–11) will be randomly selected, and you earn that round's pot.
1 point = £0.005.

This is Round {current_round}. {If round > 1: Last round you chose {prev_choice} and the pot was {pot}.}

What do you choose: A or B?

Please respond with a single letter: A or B.
```

### BRET (Bomb Risk Elicitation Task)

```
You now have the opportunity to earn additional money in a risk task.

There is a grid of 100 boxes. One box contains a hidden bomb — you do not know which one. You choose how many boxes to collect:

- Each collected box earns you points
- BUT if the bomb is in one of your collected boxes, you earn NOTHING from this task

The bomb is equally likely to be in any of the 100 boxes.

How many boxes do you want to collect? Please respond with a number from 0 to 100.
```

### Persona Prefix for Demographic Conditioning

```
You are a {age}-year-old {sex} living in the United Kingdom. Your ethnicity is {ethnicity}. Your highest education level is {education}. You are currently {employment_status}. {If student: You are a student.}
```
