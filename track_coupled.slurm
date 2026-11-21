#!/bin/bash
#SBATCH --partition=shared
#SBATCH --account=bm1344
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/track_%x_%j.out
#SBATCH --error=logs/track_%x_%j.err
# Usage: sbatch -J track_<run> --export=ALL,RUN=<ng5_rcbmf|ng5_deepoff> track_coupled.slurm
# regrid -> identify (32 stripes in parallel) -> track (cyclonic + anticyclonic in parallel). Resumable.
set -uo pipefail
RUN="${RUN:?set RUN}"
NS="${NS:-32}"
G=/work/bk1377/b382473/conda/envs/env_gribscan2360/bin/python
Y=/work/bk1377/b382473/conda/envs/pyeddytracker/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
umask 002
echo "[$(date)] regrid $RUN"
$G track_coupled.py regrid "$RUN" || { echo REGRID FAILED; exit 1; }
echo "[$(date)] identify $RUN, $NS stripes"
for i in $(seq 0 $((NS-1))); do
  $Y track_coupled.py identify "$RUN" --stripe $i --nstripes $NS > "logs/track_${RUN}_identify_$i.log" 2>&1 &
done
wait
n_a=$(ls /work/bk1377/b382473/eddytracks/$RUN/0.25deg/eddyID/eddyID_anticyclonic_????????.nc | wc -l)
n_c=$(ls /work/bk1377/b382473/eddytracks/$RUN/0.25deg/eddyID/eddyID_cyclonic_????????.nc | wc -l)
echo "[$(date)] identified: $n_a anticyclonic days, $n_c cyclonic days"
[[ "$n_a" -eq 425 && "$n_c" -eq 425 ]] || { echo "INCOMPLETE identification (expected 425), not tracking"; exit 2; }
echo "[$(date)] track $RUN"
$Y track_coupled.py track "$RUN" --type cyclonic > "logs/track_${RUN}_track_cyclonic.log" 2>&1 &
$Y track_coupled.py track "$RUN" --type anticyclonic > "logs/track_${RUN}_track_anticyclonic.log" 2>&1 &
wait
echo "[$(date)] done"; ls -la /work/bk1377/b382473/eddytracks/$RUN/0.25deg/eddyTracks/
