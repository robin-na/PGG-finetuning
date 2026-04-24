# SimBench Task Diagnosis

## Choices13k

- Rows compared: 500
- Mean TVD: baseline `0.1749`, Twin `0.2199`, Twin raw `0.2162`
- Plot: [choices13k_distribution_diagnosis.png](./choices13k_distribution_diagnosis.png)
- Note: Raw Twin heuristic uses direct Twin lottery and time-risk tasks only. This isolates the source risk signal without an LLM in the loop.
- Diagnosis: Twin raw risk signals retain much more question-to-question structure than the current Twin+LLM pipeline. The current persona prompting appears to wash out gamble sensitivity and overpredict one stable gamble preference.

## DICES

- Rows compared: 247
- Mean TVD: baseline `0.1708`, Twin `0.1522`, Twin raw `N/A`
- Plot: [dices_distribution_diagnosis.png](./dices_distribution_diagnosis.png)
- Note: Raw Twin heuristic is not applicable: Twin does not include a directly comparable safety-annotation task.
- Diagnosis: Twin LLM shifts mass toward unsafe and unsure relative to the human labels, which is consistent with Twin injecting a broad moral alarm signal instead of the narrow safety taxonomy used in DICES.

## OSPsychBig5

- Rows compared: 40
- Mean TVD: baseline `0.1400`, Twin `0.2210`, Twin raw `0.1652`
- Plot: [ospsychbig5_distribution_diagnosis.png](./ospsychbig5_distribution_diagnosis.png)
- Note: Raw Twin heuristic uses raw Twin Big Five scores mapped to item polarity. It is approximate and intentionally simple, but it uses no LLM.
- Diagnosis: Twin raw Big Five signals are more faithful than Twin+LLM, but the current Twin cards still over-idealize the person and push responses away from the middle, especially on negative-keyed items.

## OSPsychRWAS

- Rows compared: 22
- Mean TVD: baseline `0.3198`, Twin `0.2052`, Twin raw `N/A`
- Plot: [ospsychrwas_distribution_diagnosis.png](./ospsychrwas_distribution_diagnosis.png)
- Note: Raw Twin heuristic is not applicable: Twin has ideology/religion proxies, but no directly comparable RWAS measurement.
- Diagnosis: Twin improves on this task despite lacking a direct raw RWAS signal, which suggests the value is coming from broad ideology/religion proxies rather than exact item-level transport.

## OpinionQA

- Rows compared: 310
- Mean TVD: baseline `0.1648`, Twin `0.1551`, Twin raw `N/A`
- Plot: [opinionqa_distribution_diagnosis.png](./opinionqa_distribution_diagnosis.png)
- Note: Raw Twin heuristic is not applicable: Twin has relevant demographics and attitudes, but no directly comparable per-question survey responses. Also note that option letters are heterogeneous across OpinionQA questions, so A/B/C/... aggregation is only a coarse shape comparison.
- Diagnosis: At the coarse task level, baseline and Twin are both close to the human option-position distribution. Residual errors here are more local and question-specific than task-wide.
