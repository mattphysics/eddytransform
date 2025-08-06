#!/bin/bash
# Submit eddy compositing jobs

# Define experiments
exps=(
    # 'tco1279-eerie_production_202407'
    # 'tco1279-eerie_production_202408-c_0_a_LR20'
    'tco399-eerie_production_202407'
    # 'tco399-eerie_production_202408-c_0_a_LR20'
)

# Define kinds
kinds=(
    # cyclonic
    anticyclonic
)

# Define years
years=(
    {2010..2021}
)

# Loop over experiments, kinds, and years
for exp in "${exps[@]}"; do
  for kind in "${kinds[@]}"; do
    for year in "${years[@]}"; do
        echo "Processing exp=${exp}, kind=${kind}, year=${year}"
        while true; do
          Njobs=$(squeue --me | wc -l)
          Njobs=$((Njobs - 1))  # subtract header line

          if [ "$Njobs" -lt 90 ]; then
            break
          fi

          sleep 10  # wait 10 seconds before checking again
        done
        
        # Your actual command goes here, e.g.:
        # python process_data.py --exp "${exp}" --kind "${kind}" --year "${year}"

	      jid1=$(sbatch --wrap="module load conda && conda activate env_gribscan2360 && python eddytransform_run_atos_v2.py ${year} ${kind} ${exp} --rw 1" -q nf -t 2-0 --mem=128000 --mail-type=FAIL --mail-user='matthias.aengenheyster@ecmwf.int' -o /ec/res4/scratch/neam/logs/out/composites_GS_${exp}_${kind}_${year}.out -e /ec/res4/scratch/neam/logs/err/composites_GS_${exp}_${kind}_${year}.err -J composites_GS_${exp}_${kind}_${year} | awk '{print $4}')
        # echo sbatch --wrap="module load conda && conda activate env_gribscan2360 && python eddytransform_run_atos_v2.py ${year} ${kind} ${exp} --rw 1" -q nf -t 2-0 --mem=128000 --mail-type=FAIL --mail-user='matthias.aengenheyster@ecmwf.int' -o /ec/res4/scratch/neam/logs/out/composites_GS_${exp}_${kind}_${year}.out -e /ec/res4/scratch/neam/logs/err/composites_GS_${exp}_${kind}_${year}.err -J composites_GS_${exp}_${kind}_${year}
        
    done
    echo "Done submitting kind $kind" 
  done
  echo "Done submitting exp $exp" 
done

echo "DONE submitting"
