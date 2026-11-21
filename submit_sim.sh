#!/usr/bin/env bash
# Submit sim_chunk.slurm over sim_tasks.txt in chunks.
#   MODE=dry     (default) print the sbatch commands only
#   MODE=test    sbatch --test-only: validated by the scheduler, nothing submitted
#   MODE=submit  really submit; job ids are appended to sim_chunk_jobids.txt
set -euo pipefail
cd "$(dirname "$0")"

TASK_FILE="${TASK_FILE:-sim_tasks.txt}"
CHUNK="${CHUNK:-42}"
MODE="${MODE:-dry}"
FREQ="${FREQ:-10D}"
REGION="${REGION:-SO}"

lines=$(wc -l < "$TASK_FILE")
echo "TASK_FILE=$TASK_FILE lines=$lines CHUNK=$CHUNK MODE=$MODE FREQ=$FREQ REGION=$REGION"

start=1
while (( start <= lines )); do
  end=$(( start + CHUNK - 1 )); (( end > lines )) && end=$lines
  cmd=( sbatch --export=ALL,TASK_FILE="$TASK_FILE",START_LINE="$start",END_LINE="$end",CONCURRENCY="$CHUNK",FREQ="$FREQ",REGION="$REGION"
        -J "sim_${REGION}_${start}-${end}" )
  case "$MODE" in
    dry)    printf '%q ' "${cmd[@]}" sim_chunk.slurm; printf '\n' ;;
    test)   "${cmd[@]}" --test-only sim_chunk.slurm ;;
    submit) "${cmd[@]}" sim_chunk.slurm | tee -a sim_chunk_jobids.txt ;;
    *) echo "unknown MODE=$MODE" >&2; exit 2 ;;
  esac
  start=$(( end + 1 ))
done
