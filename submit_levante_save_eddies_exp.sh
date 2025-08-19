#!/bin/bash

# Submit jobs that write eddies to zarr, on Jasmin

models=(
    ifs-fesom2-sr
)

exps=(
  #hist-1950
  highres-future-ssp245
)

kinds=(
    cyclonic
    #anticyclonic
)

region=GS

dataset='2D_daily_avg'
realm='atmos'
grid='gr025'

Nloop=6

loops=(
    {1..${Nloop}}
)


for model in "${models[@]}"; do
  for exp in "${exps[@]}"; do
    for kind in "${kinds[@]}"; do
      echo "Writing zarr for composites of model=${model}, exp=${exp}, kind=${kind}"

      jid1=$(sbatch --wrap="source /home/b/b382473/.bashrc && conda activate env_gribscan2360 && python save_eddies_rot_to_zarr_exp.py ${model} ${exp} ${kind} ${region} ${dataset} ${realm} ${grid}" -q nf -t 08:00:00 --mail-type=FAIL --mail-user='matthias.aengenheyster@ecmwf.int' -o /scratch/b/b382473/logs/out/save_eddies_rot_to_zarr_${model}_${exp}_${kind}_${region}_${dataset}_${realm}_${grid}.out -e /scratch/b/b382473/logs/err/save_eddies_rot_to_zarr_${model}_${exp}_${kind}_${region}_${dataset}_${realm}_${grid}.err --mail-type=FAIL --mail-user='matthias.aengenheyster@ecmwf.int' --mem=32000 --account eerie -J save_eddies_rot_to_zarr_${model}_${exp}_${kind}_${region} --account=bk1377 --partition=compute --qos=normal | awk '{print $4}')
      echo "$jid1"
      
      jid2=$(sbatch --dependency=afterany:${jid1} --wrap="source /home/b/b382473/.bashrc && conda activate env_gribscan2360 && python fix_eddies_rot_to_zarr_exp.py ${model} ${exp} ${kind} ${region} ${dataset} ${realm} ${grid}" -q nf -t 08:00:00 --mail-type=FAIL --mail-user='matthias.aengenheyster@ecmwf.int' -o /scratch/b/b382473/logs/out/fix_eddies_rot_to_zarr_${model}_${exp}_${kind}_${region}_${dataset}_${realm}_${grid}.out -e /scratch/b/b382473/logs/err/fix_eddies_rot_to_zarr_${model}_${exp}_${kind}_${region}_${dataset}_${realm}_${grid}.err --mail-type=FAIL --mail-user='matthias.aengenheyster@ecmwf.int' --mem=32000 --account eerie -J fix_eddies_rot_to_zarr_${model}_${exp}_${kind}_${region} --account=bk1377 --partition=compute --qos=normal | awk '{print $4}')
      echo "$jid2 -> $jid1"

      sleep 2
      echo "Starting re-submission loop, submit ${Nloop} times"
      jidi2=${jid2}

      # for loop in "${loops[@]}"; do
      #   echo "loop: ${loop}"
      for ((loop=1; loop<=Nloop; loop++)); do
        echo "loop: $loop"
        jidi1=$(sbatch --dependency=afterany:${jidi2} --wrap="source /home/b/b382473/.bashrc && conda activate env_gribscan2360 && python save_eddies_rot_to_zarr_exp.py ${model} ${exp} ${kind} ${region} ${dataset} ${realm} ${grid}" -q nf -t 08:00:00 --mail-type=FAIL --mail-user='matthias.aengenheyster@ecmwf.int' -o /scratch/b/b382473/logs/out/save_eddies_rot_to_zarr_${model}_${exp}_${kind}_${region}_${dataset}_${realm}_${grid}_l${loop}.out -e /scratch/b/b382473/logs/err/save_eddies_rot_to_zarr_${model}_${exp}_${kind}_${region}_${dataset}_${realm}_${grid}_l${loop}.err --mail-type=FAIL --mail-user='matthias.aengenheyster@ecmwf.int' --mem=32000 --account eerie -J save_eddies_rot_to_zarr_${model}_${exp}_${kind}_${region}_l${loop} --account=bk1377 --partition=compute --qos=normal | awk '{print $4}')
        echo "$jidi1 -> $jidi2"
        
        jidi2=$(sbatch --dependency=afterany:${jidi1} --wrap="source /home/b/b382473/.bashrc && conda activate env_gribscan2360 && python fix_eddies_rot_to_zarr_exp.py ${model} ${exp} ${kind} ${region} ${dataset} ${realm} ${grid}" -q nf -t 08:00:00 --mail-type=FAIL --mail-user='matthias.aengenheyster@ecmwf.int' -o /scratch/b/b382473/logs/out/fix_eddies_rot_to_zarr_${model}_${exp}_${kind}_${region}_${dataset}_${realm}_${grid}_l${loop}.out -e /scratch/b/b382473/logs/err/fix_eddies_rot_to_zarr_${model}_${exp}_${kind}_${region}_${dataset}_${realm}_${grid}_l${loop}.err --mail-type=FAIL --mail-user='matthias.aengenheyster@ecmwf.int' --mem=32000 --account eerie -J fix_eddies_rot_to_zarr_${model}_${exp}_${kind}_${region}_l${loop} --account=bk1377 --partition=compute --qos=normal | awk '{print $4}')
        echo "$jidi2 -> $jidi1"
      done
      echo "Done with looping"
      sleep 2
    done
    echo "Done with $kinds"
  done
  echo "Done with $exps"
done
echo "Done with $models"

echo "Done submitting"
echo "DONE"

