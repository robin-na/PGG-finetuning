# SimBench

Local copy of the SimBench benchmark from Hugging Face:
https://huggingface.co/datasets/pitehu/SimBench

Paper:
https://arxiv.org/abs/2510.17516

GitHub evaluation repo:
https://github.com/pitehu/SimBench_release

Downloaded on 2026-04-17.

Files:
- `SimBenchPop.csv`: plain-text export of the population-level split (`7,167` rows across `20` source datasets).
- `SimBenchPop.pkl`: official pickle format expected by the upstream `generate_answers.py` script.
- `SimBenchGrouped.csv`: plain-text export of the demographic-group split (`6,343` rows across `5` survey datasets).
- `SimBenchGrouped.pkl`: official pickle format expected by the upstream `generate_answers.py` script.
- `README.hf.md`: the original dataset card downloaded from Hugging Face.

Notes:
- `SimBenchPop` covers one default population prompt per question.
- `SimBenchGrouped` adds demographic conditioning for `Afrobarometer`, `ESS`, `ISSP`, `LatinoBarometro`, and `OpinionQA`.
- The Hugging Face dataset viewer currently reports a schema mismatch between the two CSV files, so loading the individual files directly is more reliable than relying on the web preview.
