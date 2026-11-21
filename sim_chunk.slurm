#!/bin/bash
#SBATCH --job-name=sim-chunk
#SBATCH --partition=compute
#SBATCH --account=bm1344
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=250G
#SBATCH --time=08:00:00
#SBATCH --output=logs/sim_chunk_%x_%j.out
#SBATCH --error=logs/sim_chunk_%x_%j.err

# Runs lines START_LINE..END_LINE of TASK_FILE ("source idate kind"), CONCURRENCY at a time,
# one single-threaded python per task. Tasks are resumable: finished eddies are skipped.
set -uo pipefail

TASK_FILE="${TASK_FILE:-sim_tasks.txt}"
START_LINE="${START_LINE:-1}"
END_LINE="${END_LINE:-42}"
CONCURRENCY="${CONCURRENCY:-42}"      # 42 x ~6 GB < 250 GB
FREQ="${FREQ:-10D}"
REGION="${REGION:-SO}"
OUTROOT="${OUTROOT:-/work/bk1377/b382473/composites/}"
LOG_DIR="${LOG_DIR:-/scratch/b/b382473/logs/sim}"
PYTHON_CMD="${PYTHON_CMD:-/work/bk1377/b382473/conda/envs/env_gribscan2360/bin/python}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p logs "${LOG_DIR}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHON_CMD FREQ REGION OUTROOT LOG_DIR DRY_RUN
umask 002

run_task() {
  local src="$1" idate="$2" kind="$3"
  local tag="${src}_${REGION}_${kind}_$(printf '%03d' "$idate")"
  local cmd="${PYTHON_CMD} eddytransform_run_levante_sim.py ${src} ${idate} ${kind} --region ${REGION} --freq ${FREQ} --outroot ${OUTROOT}"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY_RUN: ${cmd} > ${LOG_DIR}/${tag}.out 2> ${LOG_DIR}/${tag}.err"
  else
    echo "[$(date)] START ${tag}"
    ${cmd} > "${LOG_DIR}/${tag}.out" 2> "${LOG_DIR}/${tag}.err" || echo "[$(date)] FAILED ${tag}"
  fi
}
export -f run_task

sed -n "${START_LINE},${END_LINE}p" "${TASK_FILE}" | grep -v '^\s*$' \
  | xargs -P "${CONCURRENCY}" -L1 bash -c 'run_task "$0" "$1" "$2"'
