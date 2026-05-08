# Pairwise Comparison Prompt

## System Prompt

```text
You are comparing two target tasks and forecasting which one is more likely to benefit from an external persona prior from another human study.
```

## User Prompt Template

```text
You are given a source-study card for Twin-2K-500 and two target task cards.

Goal:
Decide which target task is more likely to benefit from transferring a Twin-based persona prior, relative to a direct baseline with no such prior.

Output format:
1. First write an Explanation section in plain text.
2. Then write a Final JSON section containing only valid JSON with this schema:
{
  "more_likely_positive": "TaskA" | "TaskB" | "Tie",
  "relative_confidence": 0.0,
  "task_a_label": "positive" | "negative" | "insignificant",
  "task_b_label": "positive" | "negative" | "insignificant"
}

Twin source-study information:
- Paper title: Database Report: Twin-2K-500: A Data Set for Building Digital Twins of over 2,000 People Based on Their Answers to over 500 Questions
- Abstract-style summary: The Twin-2K-500 study introduces a large, U.S.-based multiwave dataset designed for building digital twins of individual people. Participants answer more than 500 questions spanning demographics, personality, cognition, economic preferences, behavioral games, and open-ended self-description. The paper argues that this breadth makes it possible to test whether a model can use rich prior information about a person to predict their future responses on held-out tasks.
- Source population: N=2,058 U.S. participants who completed all four waves, recruited on Prolific from a U.S.-representative intake sample by age, sex, and ethnicity.
- Collection schedule: Four survey waves collected in early 2025. Waves 1-3 contain the persona information used to build digital twins; wave 4 repeats selected tasks to measure test-retest accuracy.
- Measurement scope:
  - 14 demographic questions
  - 19 personality tests spanning 26 constructs and 279 questions
  - 11 cognitive ability tests spanning 85 questions
  - 10 economic preference tests spanning 34 questions
  - 11 between-subject behavioral-economics experiments
  - 5 within-subject behavioral-economics experiments
  - 1 pricing study with 40 purchase decisions
- Information available about each Twin person:
  - A dry compressed summary of one Twin participant based on many earlier survey responses
  - Demographics, trait scores, cognitive summaries, economic-preference summaries, and selected open-ended content
  - No access to the held-out target-study answers at forecast time

Task A card:
[Insert Task A card here]

Task B card:
[Insert Task B card here]
```

## Suggested Use

- Use the global ranking prompt when you want a single-call ranking across all tasks.
- Use this pairwise prompt when you want more stable relative judgments and are willing to aggregate many pairwise calls into a final ranking.
