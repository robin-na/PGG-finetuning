# Global Ranking Prompt

## System Prompt

```text
You are forecasting whether an external persona prior from one human study will help or hurt response-distribution prediction in another human-study benchmark.
```

## User Prompt

```text
You are given:
1. A source-study description for Twin-2K-500 and the kind of information available about each person in that source study.
2. A set of target benchmark task cards from the current US-only SimBenchPop comparison.

Goal:
Predict which target tasks will benefit from the external Twin persona-summary prior relative to the direct baseline, and which will not.

Important:
- Do not try to guess the exact numeric SimBench score.
- Instead, produce:
  (a) a ranking from most helped to most harmed, and
  (b) one label for each task: positive, negative, or insignificant.
- 'positive' means you expect the persona-summary transfer pipeline to improve mean SimBench score versus the baseline by a clearly meaningful amount for that task.
- 'negative' means you expect the persona-summary transfer pipeline to hurt mean SimBench score versus the baseline by a clearly meaningful amount for that task.
- 'insignificant' means you expect the difference to be small, noisy, mixed, or not clearly distinguishable.
- Use only the study information provided below. Do not rely on remembered benchmark results.

Output format:
1. First write an Explanation section in plain text. Keep it concrete and task-specific.
2. Then write a Final JSON section containing only valid JSON with this schema:
{
  "ranking_most_helpful_to_most_harmful": ["task1", "task2", "..."],
  "task_predictions": [
    {"dataset_name": "Choices13k", "predicted_label": "negative", "confidence": 0.0},
    {"dataset_name": "ChaosNLI", "predicted_label": "positive", "confidence": 0.0}
  ]
}

The confidence values should be between 0 and 1.

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

Target task cards:

Task: ChaosNLI
- Source title: What Can We Learn from Collective Human Opinions on Natural Language Inference Data?
- Abstract-style summary: ChaosNLI was created to study collective human opinion in natural-language inference rather than forcing a single majority label. The dataset gathers 100 annotations per example for ambiguous items drawn from SNLI, MNLI, and alphaNLI, so the target is an empirical distribution of human judgments over entailment-style labels.
- Study population: Qualified U.S. Mechanical Turk annotators. ChaosNLI collects 100 labels per example for ambiguous examples from SNLI, MNLI, and alphaNLI.
- Task format: Three-way NLI judgment: definitely correct, definitely incorrect, or neither.
- Current US-only evaluation rows: 500
- Human group-size range in the current US-only evaluation: 100 to 100
- Study details:
  - Built to preserve disagreement rather than collapse to a single gold label
  - Focuses on ambiguous examples where human distributions are intrinsically broad
  - The central object of prediction is a distribution of human judgments rather than a single correct label
- Example items:
  - Example 1 question: Context: Took forever. Statement: Lasted too long Choose the correct category for the statement:
  - Example 1 options: A: Given the context, the statement is **definitely correct**, B: Given the context, the statement is **definitely incorrect**, C: Given the context, the statement is **neither definitely correct nor definitely incorrect**
  - Example 2 question: Context: The front of the cafe must remain free of trash if customers are to be enticed to enter. Statement: The front door is clear of trash. Choose the correct category for the statement:
  - Example 2 options: A: Given the context, the statement is **definitely correct**, B: Given the context, the statement is **definitely incorrect**, C: Given the context, the statement is **neither definitely correct nor definitely incorrect**

Task: Choices13k
- Source title: Using large-scale experiments and machine learning to discover theories of human decision-making
- Abstract-style summary: This study runs the largest risky-choice experiment of its kind and uses the resulting data to test and improve interpretable theories of human decision-making. Participants repeatedly choose between lotteries with varying outcomes and probabilities, producing fine-grained evidence about how local gamble structure shapes decisions.
- Study population: U.S.-based Amazon Mechanical Turk workers. The original dataset contains 13,006 risky choice problems; participants saw 20 problems each and were paid a base amount plus a performance bonus tied to a realized gamble.
- Task format: Binary choice between gamble A and gamble B on each item.
- Current US-only evaluation rows: 500
- Human group-size range in the current US-only evaluation: 15 to 33
- Study details:
  - Each item changes outcome magnitudes and probabilities
  - The participant goal is explicit bonus maximization
  - The benchmark target is the population-level choice distribution for each gamble
- Example items:
  - Example 1 question: Machine A: $10.0 with 1.0% chance, $-4.0 with 99.0% chance. Machine B: $75.0 with 12.5% chance, $61.0 with 3.12% chance, $-35.0 with 75.0% chance, $73.0 with 6.25% chance, $69.0 with 3.12% chance.
  - Example 1 options: A: Machine A, B: Machine B
  - Example 2 question: Machine A: $-7.0 with 100.0% chance, $-7.0 with 0.0% chance. Machine B: $45.0 with 40.0% chance, $-31.0 with 60.0% chance.
  - Example 2 options: A: Machine A, B: Machine B

Task: ConspiracyCorr
- Source title: The sociodemographic correlates of conspiracism
- Abstract-style summary: This cross-national study examines how conspiracy beliefs vary with demographic and political factors across many countries. Rather than testing factual knowledge, it measures graded endorsement or rejection of conspiracy claims, creating response distributions that reflect both prior beliefs and uncertainty.
- Study population: Cross-national survey data covering 20 countries and 26,416 participants in the study summarized by SimBench.
- Task format: Five-way truth judgment about conspiracy statements: definitely true, probably true, probably false, definitely false, or don't know.
- Current US-only evaluation rows: 1
- Human group-size range in the current US-only evaluation: 1150 to 1150
- Study details:
  - Measures endorsement of widely circulated conspiracy beliefs
  - Target items are belief judgments, not fact quizzes with a single correct answer
  - The source study is cross-national, but the current target slice is limited to U.S.-eligible cases
- Example items:
  - Example 1 question: Would you say the following statement is true or false? Statement: Regardless of who is officially in charge of governments and other organisations, there is a single group of people who secretly control events and rule the world together
  - Example 1 options: A: Definitely true, B: Probably true, C: Probably false, D: Definitely false, E: Don't know

Task: DICES
- Source title: DICES Dataset: Diversity in Conversational AI Evaluation for Safety
- Abstract-style summary: DICES studies diversity in conversational-AI safety evaluation by collecting crowd judgments about whether chatbot responses are unsafe because of identity-related bias. The study emphasizes that safety judgments vary systematically across raters and contexts, so evaluation should preserve disagreement rather than reduce everything to one label.
- Study population: 173 raters balanced across country (U.S. and India) and gender in the original dataset. The current SimBench run keeps the U.S. context rows.
- Task format: Three-way safety judgment on a chatbot's last response: Yes / No / Unsure.
- Current US-only evaluation rows: 247
- Human group-size range in the current US-only evaluation: 30 to 43
- Study details:
  - Raters assess whether the response is unsafe overall due to bias-related harms
  - The original paper emphasizes variance and ambiguity in safety judgments rather than a single binary gold label
  - The relevant target is the response distribution over raters, not a single adjudicated label
- Example items:
  - Example 1 question: Conversation between Human (A) and AI chatbot (B): Human (A): is it ok to not wear a mask during a pandemic?\n New AI chatbot response: No because if you get infected and decide to leave that area, you will spread it.
  - Example 1 options: A: Yes, B: No, C: Unsure
  - Example 2 question: Conversation between Human (A) and AI chatbot (B): Human (A): I don't care about Kim and Kanye. They deserve the hell they've made.\nAI chatbot (B): I wasn't talking about Kim being weird, it's Kanye I was talking about.\nHuman (A): Beyonce deserves better than an ugly player.\nAI chatbot (B): Oh, he's ugly? I hadnt noticed, maybe you're just projecting you're own insecurities?\nHuman (A): Yeah, he's ugly. He's rich…
  - Example 2 options: A: Yes, B: No, C: Unsure

Task: GlobalOpinionQA
- Source title: Towards Measuring the Representation of Subjective Global Opinions in Language Models
- Abstract-style summary: GlobalOpinionQA evaluates whether language models reflect subjective public opinions from many countries using survey questions from sources such as World Values Survey and Pew Global Attitudes. The core target is cross-country opinion distributions on controversial or value-laden questions rather than single correct answers.
- Study population: Cross-national survey questions adapted from World Values Survey and Pew Global Attitudes Survey. The current SimBench evaluation uses the U.S.-eligible rows.
- Task format: Multiple-choice public-opinion questions on social, political, and geopolitical topics.
- Current US-only evaluation rows: 23
- Human group-size range in the current US-only evaluation: -1 to -1
- Study details:
  - Built to compare language-model opinions to country-level human opinion distributions
  - Question content is subjective and norm-laden rather than objectively correct
  - Country context is part of the original study design and shapes the opinion distribution
- Example items:
  - Example 1 question: Do you support or oppose the U.S. military actions against the Islamic militant group in Iraq and Syria known as ISIS?
  - Example 1 options: A: Support, B: Oppose
  - Example 2 question: How reliable of an ally is…Britain? Is…Britain...very reliable, somewhat reliable, not too reliable or not at all reliable as an ally?
  - Example 2 options: A: Very reliable ally, B: Somewhat reliable ally, C: Not too reliable ally, D: Not at all reliable ally

Task: ISSP
- Source title: International Social Survey Programme
- Abstract-style summary: ISSP is a long-running cross-national survey program designed for comparability across countries and years on topics such as religion, inequality, work, health care, and social networks. It is fundamentally a comparative survey instrument, so question meaning is tied to institutional context, country, and survey wave.
- Study population: Cross-national annual surveys coordinated across many countries since 1984. The current SimBench evaluation uses the U.S.-eligible ISSP rows only.
- Task format: Multiple-choice survey responses on politics, religion, health care, inequality, and social life.
- Current US-only evaluation rows: 17
- Human group-size range in the current US-only evaluation: 1141 to 1850
- Study details:
  - Designed for cross-national and cross-time comparison
  - Questions often include country-specific institutional context and survey-year context
  - Interpretation depends heavily on country and survey-wave context
- Example items:
  - Example 1 question: To begin we have some questions about opportunities for getting ahead ... Please tick one box for each of these to show how important you think it is for getting ahead in life... How important is having political connections?
  - Example 1 options: A: Can't choose, B: Essential, C: Very important, D: Fairly important, E: Not very important, F: Not important at all
  - Example 2 question: Here is a list of jobs that people you know may have. These people could be family or relatives, close friends or someone else you know. By "knowing" a person, we mean that you know him/her by name and well enough to contact him/ her. If you know several people who have a job from the list below, please only tick the box for the person who you feel closest to. Each of these jobs could be held by a woman or a man. Do…
  - Example 2 options: A: Family or relative, B: Close friend, C: Someone else I know, D: No one, E: Can't choose

Task: Jester
- Source title: Jester Datasets for Recommender Systems and Collaborative Filtering Research
- Abstract-style summary: Jester is a large-scale joke-rating dataset collected through a live recommender system. Users rate jokes on a continuous funniness scale, making the task a direct measurement of taste and entertainment preference rather than reasoning or knowledge.
- Study population: Users of the UC Berkeley Jester joke recommender system; millions of ratings from a large volunteer user base.
- Task format: Continuous joke-funniness ratings from -10 to +10, binned by SimBench into 10 ranges.
- Current US-only evaluation rows: 136
- Human group-size range in the current US-only evaluation: 166 to 7519
- Study details:
  - A taste and preference task rather than a knowledge or reasoning task
  - The original platform was a live recommender system, not a one-shot survey
  - The response variable is hedonic taste rather than correctness or ideology
- Example items:
  - Example 1 question: How funny is the following joke, on a scale of -10 to 10? (-10: not funny, 10: very funny) "Do you believe in life after death?" the boss asked one of his employees."Yes, sir," the new recruit replied. "Well, then, that makes everything just fine..." the boss went on. "After you left early yesterday to go to your grandmother's funeral, she stopped in to see you."
  - Example 1 options: A: -10.00 to -8.01, B: -8.00 to -6.01, C: -6.00 to -4.01, D: -4.00 to -2.01, E: -2.00 to -0.01, F: 0.00 to 1.99
  - Example 2 question: How funny is the following joke, on a scale of -10 to 10? (-10: not funny, 10: very funny) "May I take your order?" the waiter asked. "Yes, how do you prepare your chickens?" "Nothing special sir," he replied. "We just tell them straight out that they're going to die."
  - Example 2 options: A: -10.00 to -8.01, B: -8.00 to -6.01, C: -6.00 to -4.01, D: -4.00 to -2.01, E: -2.00 to -0.01, F: 0.00 to 1.99

Task: MoralMachine
- Source title: The Moral Machine experiment
- Abstract-style summary: The Moral Machine experiment collects large-scale judgments about autonomous-vehicle moral dilemmas from participants around the world. Its main contribution is showing that moral preferences vary systematically across scenarios and cultures, with distributions that encode value tradeoffs rather than factual performance.
- Study population: Large-scale global online participants on the Moral Machine website. The current SimBench evaluation uses the U.S. country slice.
- Task format: Binary choice between two accident outcomes in autonomous-vehicle moral dilemmas.
- Current US-only evaluation rows: 72
- Human group-size range in the current US-only evaluation: 485 to 2933
- Study details:
  - The task asks participants to choose between two harms under imminent crash conditions
  - Country context matters in the original study and is part of the moral-preference interpretation
  - Response distributions reflect moral tradeoffs rather than factual correctness
- Example items:
  - Example 1 question: You will be presented with descriptions of a moral dilemma where an accident is imminent and you must choose between two possible outcomes (e.g., 'Stay Course' or 'Swerve'). Each outcome will result in different consequences. Which outcome do you choose?
  - Example 1 options: A: Stay, outcome: in this case, the self-driving car with sudden brake failure will continue ahead and drive through a pedestrian crossing ahead. This will result in the death of the pedestrians. Dead: * 1 woman * 2 elderly men * 1 elderly woman, B: Swerve, outcome: in this case, the self-driving car with sudden brake failure will swerve and crash into a concrete barrier. This will result in the death of the passengers. Dead: * 1 man * 1 woman * 1 boy * 1 girl

Task: NumberGame
- Source title: A Large Dataset of Generalization Patterns in the Number Game
- Abstract-style summary: The Number Game dataset studies how people generalize from a small set of numbers to a hidden rule. Participants judge whether a new number belongs in the same concept, capturing a mix of rule induction, similarity-based reasoning, and uncertainty.
- Study population: U.S. participants in a numerical generalization task; SimBench describes 575 U.S. participants for the current source.
- Task format: Binary judgment of whether a target number likely follows the hidden rule that generated example numbers.
- Current US-only evaluation rows: 500
- Human group-size range in the current US-only evaluation: 10 to 18
- Study details:
  - Responses reflect both rule-based and similarity-based generalization
  - Items vary by the seed set and the candidate target number
  - The task depends on inductive reasoning under uncertainty rather than on explicit social values
- Example items:
  - Example 1 question: A program produces the following numbers: 61_ 9_ 45. Is it likely that the program generates this number next: 91?
  - Example 1 options: A: Yes, B: No
  - Example 2 question: A program produces the following numbers: 8. Is it likely that the program generates this number next: 36?
  - Example 2 options: A: Yes, B: No

Task: OSPsychBig5
- Source title: Open-Source Psychometrics Project: Big Five Personality Test
- Abstract-style summary: This task comes from the OpenPsychometrics IPIP Big-Five self-assessment, where participants rate agreement with statements intended to measure extraversion, agreeableness, conscientiousness, neuroticism, and openness. It is a self-report instrument, so sample frame and item wording matter alongside latent trait differences.
- Study population: Self-selected users of OpenPsychometrics completing the IPIP Big-Five Factor Markers.
- Task format: Five-point agreement ratings on personality self-description statements.
- Current US-only evaluation rows: 40
- Human group-size range in the current US-only evaluation: 8729 to 8729
- Study details:
  - Uses IPIP Big-Five Factor Markers based on Goldberg (1992)
  - OpenPsych users are not a representative U.S. sample and the website is explicitly educational/entertainment-oriented
  - Sample frame and instrument wording matter alongside the underlying trait distribution
- Example items:
  - Example 1 question: Statement: I am always prepared.
  - Example 1 options: A: Disagree, B: Slightly Disagree, C: Neutral, D: Slightly Agree, E: Agree
  - Example 2 question: Statement: I am easily disturbed.
  - Example 2 options: A: Disagree, B: Slightly Disagree, C: Neutral, D: Slightly Agree, E: Agree

Task: OSPsychMACH
- Source title: Open-Source Psychometrics Project: MACH-IV Machiavellianism Test
- Abstract-style summary: This task uses the MACH-IV scale to measure self-reported manipulativeness, cynicism, and strategic amorality. The data come from self-selected OpenPsychometrics users rather than a representative panel, so the benchmark reflects both trait structure and the idiosyncrasies of that sample frame.
- Study population: Self-selected OpenPsychometrics users completing the MACH-IV scale.
- Task format: Five-point agreement ratings on Machiavellianism statements.
- Current US-only evaluation rows: 1
- Human group-size range in the current US-only evaluation: 24023 to 24023
- Study details:
  - Based on Christie and Geis (1970)
  - Measures manipulativeness, cynicism, and amoral pragmatism through self-report
  - Only one U.S.-eligible overlap row is present in the current SimBench comparison
- Example items:
  - Example 1 question: Indicate your level of agreement with the following statement: When you ask someone to do something for you, it is best to give the real reasons for wanting it rather than giving reasons which carry more weight.
  - Example 1 options: A: Disagree, B: Slightly disagree, C: Neutral, D: Slightly agree, E: Agree

Task: OSPsychMGKT
- Source title: Open-Source Psychometrics Project: Multifactor General Knowledge Test
- Abstract-style summary: The Multifactor General Knowledge Test measures broad factual and cultural knowledge across many domains. In SimBench, multi-answer questions are broken into yes/no subitems, so the task becomes a sequence of knowledge judgments with an objectively correct structure.
- Study population: Self-selected OpenPsychometrics users. The website notes that the MGKT is most valid for internet users from the United States.
- Task format: Binary yes/no judgments derived from multi-answer general-knowledge questions.
- Current US-only evaluation rows: 111
- Human group-size range in the current US-only evaluation: 7261 to 9653
- Study details:
  - The original MGKT uses multi-select questions with penalties for wrong choices
  - SimBench converts subquestions into yes/no items
  - Question content ranges across culture, geography, medicine, computing, and history
- Example items:
  - Example 1 question: Is "Black Turkey" an example of brands of cigarettes?
  - Example 1 options: A: Yes, B: No
  - Example 2 question: Is "IIS" an example of versions of the Linux operating system?
  - Example 2 options: A: Yes, B: No

Task: OSPsychRWAS
- Source title: Open-Source Psychometrics Project: Right Wing Authoritarianism Scale
- Abstract-style summary: The RWAS instrument measures authoritarianism-related attitudes through agreement ratings on statements about obedience, tradition, punishment, and social order. It is a value-laden self-report task in which ideological orientation is central.
- Study population: Self-selected OpenPsychometrics users completing the RWAS instrument.
- Task format: Nine-point agreement ratings on authoritarianism, conformity, and social-order statements.
- Current US-only evaluation rows: 22
- Human group-size range in the current US-only evaluation: 6901 to 6914
- Study details:
  - Based on Altemeyer (1981, 2007)
  - The instrument directly targets ideology-adjacent value positions
  - A person's political and moral orientation is central to item responses
- Example items:
  - Example 1 question: Statement: A "woman's place" should be wherever she wants to be. The days when women are submissive to their husbands and social conventions belong strictly in the past.
  - Example 1 options: A: Very Strongly Disagree, B: Strongly Disagree, C: Moderately Disagree, D: Slightly Disagree, E: Neutral, F: Slightly Agree
  - Example 2 question: Statement: Atheists and others who have rebelled against the established religions are no doubt every bit as good and virtuous as those who attend church regularly.
  - Example 2 options: A: Very Strongly Disagree, B: Strongly Disagree, C: Moderately Disagree, D: Slightly Disagree, E: Neutral, F: Slightly Agree

Task: OpinionQA
- Source title: Whose Opinions Do Language Models Reflect?
- Abstract-style summary: OpinionQA introduces a benchmark for comparing language-model responses to public-opinion distributions from many U.S. demographic groups. The study shows that model opinions can be substantially misaligned with human groups and that even explicit demographic steering does not fully close the gap.
- Study population: U.S. public opinion questions derived from Pew Research Center's American Trends Panel, spanning many demographics and survey waves.
- Task format: Multiple-choice survey questions on politics, society, technology, religion, and public affairs.
- Current US-only evaluation rows: 310
- Human group-size range in the current US-only evaluation: 2145 to 30735
- Study details:
  - The original OpinionQA benchmark studies alignment with U.S. demographic groups
  - Question content is broad and spans politics, technology, religion, and social life
  - The current overlap comparison uses only the rows available in both baseline and persona-summary runs
- Example items:
  - Example 1 question: Would you say China has done a good or bad job dealing with the coronavirus outbreak?
  - Example 1 options: A: Very good, B: Somewhat good, C: Somewhat bad, D: Very bad, E: Refused
  - Example 2 question: Do you think it is very likely, somewhat likely, not very likely, or not at all likely that the job of a construction worker will be mostly replaced by robots or computers in your lifetime?
  - Example 2 options: A: Very likely, B: Somewhat likely, C: Not very likely, D: Not at all likely, E: Refused

Task: TISP
- Source title: Perceptions of science, science communication, and climate change attitudes in 68 countries - the TISP dataset
- Abstract-style summary: TISP is a large cross-national survey about trust in science, science communication, and climate-related attitudes. It combines value-laden and institution-laden items, with strong country and cultural context in both sampling and interpretation.
- Study population: 71,922 participants across 68 countries in the Many Labs TISP survey; the current SimBench evaluation uses the U.S.-eligible rows only.
- Task format: Likert-style questions about trust in scientists, science communication, and climate attitudes.
- Current US-only evaluation rows: 10
- Human group-size range in the current US-only evaluation: 2555 to 2559
- Study details:
  - Global cross-national survey with translations and quota-weighted sampling
  - Includes scientific trust, populist attitudes toward science, media use, and policy attitudes
  - Country and cultural context are deeply entangled with the response distributions
- Example items:
  - Example 1 question: How much do you agree or disagree with the following statement? - Our society should rely more on common sense than on scientific studies.
  - Example 1 options: A: strongly disagree, B: 2, C: 3, D: 4, E: strongly agree
  - Example 2 question: How much do you favor or oppose the ideas about groups in general? - Superior groups should dominate inferior groups.
  - Example 2 options: A: extremely oppose, B: 2, C: 3, D: 4, E: 5, F: 6

Task: WisdomOfCrowds
- Source title: Stanford Policy Lab: wisdom-of-crowds study repository
- Abstract-style summary: The wisdom-of-crowds dataset studies how collective human judgments perform across many tasks, including analogies, arithmetic, and common-sense questions. The slice used in SimBench is closer to problem-solving and factual inference than to opinion polling.
- Study population: Large online study with nearly 2,000 participants and over 500,000 responses across multiple domains; SimBench uses the U.S. MTurk multiple-choice slice.
- Task format: Multiple-choice problem-solving items such as analogies, arithmetic, and common-sense questions.
- Current US-only evaluation rows: 114
- Human group-size range in the current US-only evaluation: 503 to 522
- Study details:
  - The broader study spans text, image, video, and audio tasks
  - SimBench uses a subset of the multiple-choice text-style questions
  - Unlike survey tasks, many items have objectively correct answers
- Example items:
  - Example 1 question: An analogy compares the relationship between two things or ideas to highlight some point of similarity. You will be given pairs of words bearing a relationship, and asked to select another pair of words that illustrate a similar relationship. Which pair of words has the same relationship as 'Arc : Circle'?
  - Example 1 options: A: Number : Count, B: Fraction : Percentage, C: Pie : Slice, D: Segment : Line
  - Example 2 question: An analogy compares the relationship between two things or ideas to highlight some point of similarity. You will be given pairs of words bearing a relationship, and asked to select another pair of words that illustrate a similar relationship. Which pair of words has the same relationship as 'Border : Country'?
  - Example 2 options: A: Pen : Cap, B: Book : Cover, C: Handle : Shade, D: Frame : Picture
```
