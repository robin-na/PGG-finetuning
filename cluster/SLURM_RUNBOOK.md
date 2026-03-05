# SLURM Runbook (PGG Benchmark Pipeline)

This is a copy-paste reference for running benchmark micro-eval jobs on the cluster.

## Scope

This runbook covers:

1. Environment setup checks (`joblib`, `scikit-learn`).
2. GPU request patterns (`h100`, `h200`, `a100`).
3. Benchmark split runs for:
   - `no_archetype`
   - `random_archetype`
   - `oracle_archetype`
   - `retrieved_archetype`
4. Queue/debug commands (`squeue`, `sacct`, `sprio`, `scontrol`, `sinfo`).
5. Common errors and fixes.

Ready-made `benchmark/data` (40-game paired-manifest) scripts:

- `benchmark/run_scripts/slurm_macro_benchmark_filtered_no_archetype.sh`
- `benchmark/run_scripts/slurm_macro_benchmark_filtered_random_archetype.sh`
- `benchmark/run_scripts/slurm_macro_benchmark_filtered_oracle_archetype.sh`
- `benchmark/run_scripts/slurm_macro_benchmark_filtered_retrieved_archetype.sh`

## Canonical Paths/Vars

Use these in scripts:

```bash
SPLIT_ROOT="benchmark/data_ood_splits/player_count/high_to_low"
SPLIT_REL="benchmark_ood/player_count/high_to_low"
RUN_ROOT="outputs/benchmark/runs/${SPLIT_REL}"
ARCH_ROOT="${RUN_ROOT}/archetype_retrieval"

ORACLE_POOL="Persona/archetype_oracle_gpt51_val.jsonl"
RETRIEVED_POOL="${ARCH_ROOT}/validation_wave/synthetic_archetype_ridge_val.jsonl"
RETRIEVED_TRACE="${ARCH_ROOT}/validation_wave/synthetic_archetype_ridge_val_trace.jsonl"
```

## One-Time Environment Checks

Activate env:

```bash
eval "$(/home/software/anaconda3/2023.07/bin/conda shell.bash hook)"
conda activate /home/robinna/.conda/envs/llm_conda
```

Install deps:

```bash
python -m pip install --upgrade joblib scikit-learn
python -c "import joblib, sklearn; print('joblib', joblib.__version__, 'sklearn', sklearn.__version__)"
```

## SLURM Header (Recommended)

Use one GPU unless you intentionally parallelize:

```bash
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1     # change to h200:1 if desired
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
```

Notes:

1. `--gres=gpu:h100:1` or `--gres=gpu:h200:1` is explicit and reliable.
2. `--constraint='h100|h200|a100'` may fail with `Invalid feature specification` on clusters where these are GRES types, not node features.

## Minimal Script Body (Shared)

```bash
set -euo pipefail
mkdir -p logs
cd "${SLURM_SUBMIT_DIR:-$HOME/PGG-finetuning}"

eval "$(/home/software/anaconda3/2023.07/bin/conda shell.bash hook)"
conda activate /home/robinna/.conda/envs/llm_conda

which python
nvidia-smi
```

## Variant Commands

### 1) `no_archetype`

```bash
RUN_ID="${SLURM_JOB_ID:-manual}_r${SLURM_RESTART_COUNT:-0}_no_arch"
MAX_GAMES="${MAX_GAMES:-20}"
BASE_MODEL="${BASE_MODEL:-google/gemma-3-27b-it}"

COMMON_MICRO_ARGS=(
  --micro-arg "--provider local"
  --micro-arg "--base_model ${BASE_MODEL}"
  --micro-arg "--use_peft False"
  --micro-arg "--include_reasoning True"
  --micro-arg "--run_id ${RUN_ID}"
)
if [[ -n "${MAX_GAMES}" ]]; then
  COMMON_MICRO_ARGS+=(--micro-arg "--max_games ${MAX_GAMES}")
fi

python benchmark/scripts/run_split_pipeline.py \
  --split-root "${SPLIT_ROOT}" \
  --stages micro \
  --archetype-mode none \
  --micro-variant no_archetype \
  "${COMMON_MICRO_ARGS[@]}"
```

### 2) `oracle_archetype`

```bash
RUN_ID="${SLURM_JOB_ID:-manual}_r${SLURM_RESTART_COUNT:-0}_oracle"
MAX_GAMES="${MAX_GAMES:-20}"
BASE_MODEL="${BASE_MODEL:-google/gemma-3-27b-it}"

COMMON_MICRO_ARGS=(
  --micro-arg "--provider local"
  --micro-arg "--base_model ${BASE_MODEL}"
  --micro-arg "--use_peft False"
  --micro-arg "--include_reasoning True"
  --micro-arg "--run_id ${RUN_ID}"
)
if [[ -n "${MAX_GAMES}" ]]; then
  COMMON_MICRO_ARGS+=(--micro-arg "--max_games ${MAX_GAMES}")
fi

test -s "${ORACLE_POOL}"

python benchmark/scripts/run_split_pipeline.py \
  --split-root "${SPLIT_ROOT}" \
  --stages micro \
  --archetype-mode matched_summary \
  --archetype-summary-pool "${ORACLE_POOL}" \
  --micro-variant oracle_archetype \
  "${COMMON_MICRO_ARGS[@]}"
```

### 3) `retrieved_archetype`

Idempotent ensure logic:

```bash
LATEST_RUN_FILE="${ARCH_ROOT}/model_runs/latest_run.txt"
need_train=0
if [[ ! -s "${LATEST_RUN_FILE}" ]]; then
  need_train=1
else
  RUN_DIR_RAW="$(cat "${LATEST_RUN_FILE}")"
  if [[ "${RUN_DIR_RAW}" = /* ]]; then RUN_DIR="${RUN_DIR_RAW}"; else RUN_DIR="${PWD}/${RUN_DIR_RAW}"; fi
  if [[ ! -d "${RUN_DIR}" ]]; then
    need_train=1
  elif ! find "${RUN_DIR}" -path "*/models/ridge.joblib" | grep -q .; then
    need_train=1
  fi
fi

if [[ "${need_train}" -eq 1 ]]; then
  python benchmark/scripts/run_split_pipeline.py \
    --split-root "${SPLIT_ROOT}" \
    --stages archetype-train,archetype-validate
fi

RUN_DIR_RAW="$(cat "${LATEST_RUN_FILE}")"
if [[ "${RUN_DIR_RAW}" = /* ]]; then RUN_DIR="${RUN_DIR_RAW}"; else RUN_DIR="${PWD}/${RUN_DIR_RAW}"; fi

if [[ ! -s "${RETRIEVED_POOL}" ]]; then
  python Persona/archetype_retrieval/generate_synthetic_persona_val.py \
    --run-dir "${RUN_DIR}" \
    --demographics-csv "${SPLIT_ROOT}/demographics/demographics_numeric_val.csv" \
    --environment-csv "${SPLIT_ROOT}/processed_data/df_analysis_val.csv" \
    --oracle-summary-jsonl "${ORACLE_POOL}" \
    --model ridge \
    --output-jsonl "${RETRIEVED_POOL}" \
    --output-trace-jsonl "${RETRIEVED_TRACE}"
fi
```

Then run micro:

```bash
RUN_ID="${SLURM_JOB_ID:-manual}_r${SLURM_RESTART_COUNT:-0}_retrieved"
MAX_GAMES="${MAX_GAMES:-20}"
BASE_MODEL="${BASE_MODEL:-google/gemma-3-27b-it}"

COMMON_MICRO_ARGS=(
  --micro-arg "--provider local"
  --micro-arg "--base_model ${BASE_MODEL}"
  --micro-arg "--use_peft False"
  --micro-arg "--include_reasoning True"
  --micro-arg "--run_id ${RUN_ID}"
)
if [[ -n "${MAX_GAMES}" ]]; then
  COMMON_MICRO_ARGS+=(--micro-arg "--max_games ${MAX_GAMES}")
fi

python benchmark/scripts/run_split_pipeline.py \
  --split-root "${SPLIT_ROOT}" \
  --stages micro \
  --archetype-mode matched_summary \
  --archetype-summary-pool "${RETRIEVED_POOL}" \
  --micro-variant retrieved_archetype \
  "${COMMON_MICRO_ARGS[@]}"
```

### 4) `random_archetype`

In this project, `random_archetype` samples from the learning-wave oracle pool:

```bash
RUN_ID="${SLURM_JOB_ID:-manual}_r${SLURM_RESTART_COUNT:-0}_random"
MAX_GAMES="${MAX_GAMES:-20}"
BASE_MODEL="${BASE_MODEL:-google/gemma-3-27b-it}"
LEARN_ORACLE_POOL="Persona/archetype_oracle_gpt51_learn.jsonl"

test -s "${LEARN_ORACLE_POOL}"

COMMON_MICRO_ARGS=(
  --micro-arg "--provider local"
  --micro-arg "--base_model ${BASE_MODEL}"
  --micro-arg "--use_peft False"
  --micro-arg "--include_reasoning True"
  --micro-arg "--run_id ${RUN_ID}"
)
if [[ -n "${MAX_GAMES}" ]]; then
  COMMON_MICRO_ARGS+=(--micro-arg "--max_games ${MAX_GAMES}")
fi

python benchmark/scripts/run_split_pipeline.py \
  --split-root "${SPLIT_ROOT}" \
  --stages micro \
  --archetype-mode random_summary \
  --archetype-summary-pool "${LEARN_ORACLE_POOL}" \
  --micro-variant random_archetype \
  "${COMMON_MICRO_ARGS[@]}"
```

## Refresh Run Pointers

```bash
python benchmark/scripts/run_split_pipeline.py \
  --split-root "${SPLIT_ROOT}" \
  --stages index
```

This updates:

`outputs/benchmark/runs/<split-relative>/run_index.json`

## Queue/Debug Commands

Job queue:

```bash
squeue -u $USER -o "%.18i %.10P %.24j %.8T %.10M %.12R"
```

Priority:

```bash
sprio -u $USER
sprio -w
```

Job details:

```bash
scontrol show job <JOBID> | egrep "Priority=|Reason=|QOS=|TRES=|Partition="
```

Accounting (including preemption/restarts):

```bash
sacct -j <JOBID> -D --format=JobID,JobName,State,Start,End,Elapsed,ExitCode,NodeList
```

GPU/node availability:

```bash
sinfo -p mit_normal_gpu -N -h -o "%N %T %G %f" | sort
```

## Common Failure Cases

1. `latest_run.txt: No such file or directory`
   - Retrieval outputs are missing on that machine.
   - Fix: run `archetype-train,archetype-validate` or sync that split’s `archetype_retrieval/model_runs`.

2. `ModuleNotFoundError: No module named 'joblib'`
   - Install in the active env.

3. Duplicate micro output timestamps under one job id
   - Usually job preempted and requeued (`mit_preemptable`).
   - Use `--run_id ${SLURM_JOB_ID}_r${SLURM_RESTART_COUNT}_<variant>` to make restarts explicit.

4. `Invalid feature specification`
   - Constraint string unsupported for that partition/site.
   - Request explicit GRES type instead (for example `--gres=gpu:h100:1`).

## Consistency Notes

1. Split consistency comes from `--split-root` and `--data_root` wiring in `run_split_pipeline.py`.
2. `MAX_GAMES` affects only micro game selection, not retrieval train/validate or synthetic generation.
3. For stable variant comparison, keep the same:
   - `SPLIT_ROOT`
   - `MAX_GAMES`
   - `BASE_MODEL`
   - provider settings
