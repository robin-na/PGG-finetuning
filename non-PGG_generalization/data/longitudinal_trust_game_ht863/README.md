# Longitudinal Repeated Trust Game Dataset (OSF: ht863)

## Source
- **OSF Project**: [https://osf.io/ht863/](https://osf.io/ht863/)
- **Registered Report**: [https://osf.io/fa697](https://osf.io/fa697)
- **Recruitment Platform**: Prolific (UK residents, English native speakers, approval rate >= 90%, >10 prior submissions)

## Study Overview

This dataset comes from a longitudinal trust game study in which the same participants completed **10 sessions over a 3-week period** (every other day). In each session, participants played **16 trust game trials** with virtual partners, yielding 160 total decisions per participant.

Unlike a standard binary trust game, this uses a **willingness-to-play (WTP) paradigm**: participants rate how much they want to play with a partner given the partner's historical cooperation probability and the stake required.

## Experimental Design

### Trust Game Trial Structure

On each trial, participants see:
1. **"In the past, your partner shared: [X]%"** — the partner's cooperation probability
2. **"Tokens you have to give: [Y]"** — the stake (cost to play)

The participant rates: **"How much do you want to play with this partner?"** on a 1–9 scale (1 = "Not at all", 9 = "Extremely").

If the participant plays:
- The stake tokens go to the partner and are **doubled**
- With probability X%, the partner shares and both receive equal payoffs
- With probability (1-X)%, the partner keeps everything

### 16 Trials Per Session (Full Factorial)

| Factor | Levels |
|--------|--------|
| **Cooperation Probability** | 80%, 75%, 70%, 65% |
| **Stake** | 1, 2, 4, 5 tokens |

This gives 4 x 4 = 16 unique trial types per session, fully crossed.

### Payment
- **Fixed**: £7.50 for completing all 10 sessions
- **Bonus**: One randomly selected trial per session determines bonus payment (1 token = £0.10)
- Payment settled at the end of the 3-week period

## Data Files

### `Data/matrix_mean.csv` (Primary Analysis File)
Long-format panel data: one row per participant-day (~154 participants x 10 days = 1,540 rows).

| Column | Description |
|--------|-------------|
| `PID` | Row index |
| `MeanTrustGame` | Mean WTP rating across 16 trials (1–9 scale) |
| `Day` | Session number (1–10) |
| `Questionnaire` | Mean of all 6 GSS trust items (Day 1 + Day 10) |
| `QuestionnaireD1` | Mean of 3 GSS trust items at Day 1 (0–10 scale) |
| `QuestionnaireD10` | Mean of 3 GSS trust items at Day 10 (0–10 scale) |
| `SumExtraHelp` | Sum of prosocial behavior indicators |
| `ExtraHelp` | Willingness to help review future study (0/1) |
| `DataSharing` | Consent to share data (0/1) |
| `ChildhoodResources` | 3-item childhood SES composite (1–7 scale) |
| `ChildhoodPredictability` | 3-item childhood predictability composite (1–7 scale) |
| `AdultResources` | 3-item current SES composite (1–7 scale) |
| `Age` | Age in years |

### `Data/Repeated_trust_game+-+day+{N}_*.csv` (Raw Qualtrics Files, N=1–10)
Raw session-level data with 3 Qualtrics header rows (column names, question text, import IDs).

Key columns:
- `{1..16}_Q38`: Per-trial WTP decision (1–9 scale)
- `{1..16}_Q25_*`: Per-trial response time metrics (First Click, Last Click, Page Submit, Click Count)
- `Q60`: GSS trust item 1 — "Can most people be trusted?" (0–10)
- `Q61`: GSS trust item 2 — "Would people take advantage of you?" (0–10)
- `Q62`: GSS trust item 3 — "Are people helpful or self-interested?" (0–10)
- `Q52`: Prolific PID (identifier)
- `totalGains`, `totalBonus`: Accumulated earnings
- `GamePlayed`, `Cooperated`: Which trial was randomly selected and its outcome
- `PROLIFIC_PID`: Prolific participant ID

### `Scripts/`
- `AnalysesMainRegisteredReport.R`: Main registered report analyses (multilevel models, trust trajectory)
- `AnalysesLCA.R`: Latent class analysis identifying 4 trust trajectory types

## Demographics Available
- **Age**: Yes (in matrix_mean.csv)
- **Gender**: Collected on Day 10 raw CSV (Q49: Male/Female/Non-binary), NOT in matrix_mean.csv
- **Education**: Not collected
- **SES**: Childhood resources, childhood predictability, adult resources (composite scales)

## Key Behavioral Variables
- Trust decisions (WTP 1–9) across 160 trials over 10 sessions
- GSS social trust questionnaire (Day 1 and Day 10)
- Real prosocial behavior (data sharing, volunteering to help)

## Sample
- ~275 recruited; ~154 completed all 10 sessions and passed comprehension checks
- UK residents, English native speakers, Prolific approval rate >= 90%

---

## LLM Simulation Prompt

The following prompt template can be used to simulate the trust game task with an LLM agent. One prompt should be issued per trial (16 per session, 10 sessions).

```
You are participating in an economic decision-making study. You will play a series of trust games with different partners.

In this game, both you and your partner start with 5 tokens each. You must decide whether to play a trust game with a partner.

Here is the information about this trial:
- In the past, your partner shared (cooperated) {cooperation_probability}% of the time.
- If you decide to play, you must give {stake} of your tokens to your partner. These tokens will be doubled. Your partner then decides whether to share the total equally or keep everything.
- If you decide NOT to play, both you and your partner keep your original 5 tokens.

Your decision will affect your real earnings: one trial from each session will be randomly selected, and your payoff from that trial will be converted to a cash bonus (1 token = £0.10).

On a scale from 1 to 9, how much do you want to play with this partner?
1 = "Not at all"
9 = "Extremely"

Please respond with a single integer from 1 to 9.
```

### Parameter Grid for Full Simulation
To replicate the full study, run the above prompt for all 16 combinations:

| cooperation_probability | stake |
|------------------------|-------|
| 80 | 1, 2, 4, 5 |
| 75 | 1, 2, 4, 5 |
| 70 | 1, 2, 4, 5 |
| 65 | 1, 2, 4, 5 |

Repeat this 16-trial block 10 times (simulating 10 sessions) to capture any trajectory effects. For persona-based simulation, prepend demographic context:

```
You are a {age}-year-old {gender} living in the United Kingdom. You grew up in a {childhood_resources_description} household. Your current financial situation is {adult_resources_description}.
```
