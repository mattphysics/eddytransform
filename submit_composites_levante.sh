#!/bin/bash
# Submit eddy compositing jobs

echo submit Southern Ocean
sleep 3

# Define experiments
# exps=(
#     'hist-1950'
# )

# Define kinds
kinds=(
    cyclonic
    anticyclonic
)

# Define years
years=(
    # {2010..2021}
    {1955..1969}
)

model='ifs-fesom2-sr'
dataset='2D_daily_avg'
exp='eerie-control-1950'
# exp='highres-future-ssp245'
realm='atmos'
grid='gr025'

varnames=(
  avg_sst
  avg_ci
  mean2t
  mean10ws
  mslhf
  msshf
  tprate
  avg_2sh
  meantcc
  avg_blh
  avg_tcwv
  avg_tcw
  msnlwrf
  msnswrf
  mtnlwrf
  mtnswrf
)

# Loop over experiments, kinds, and years
# for exp in "${exps[@]}"; do
for year in "${years[@]}"; do
  for kind in "${kinds[@]}"; do
    for vn in "${varnames[@]}"; do
      # if [ "$year" -le 2014 ]; then
      #   exp=hist-1950
      # else
      #   exp=highres-future-ssp245
      # fi

      if [[ "$vn" == "${varnames[0]}" ]]; then
        rw=1
      else
        rw=0
      fi

      echo "Processing exp=${exp}, kind=${kind}, year=${year}, vn=${vn}"

      while true; do
        Njobs=$(squeue --me | wc -l)
        Njobs=$((Njobs - 1))  # subtract header line

        # if [ "$Njobs" -lt 400 ]; then
        if [ "$Njobs" -lt 900 ]; then
          break
        else
          echo "currently running $Njobs jobs"
        fi

        sleep 10  # wait 10 seconds before checking again
      done
      
      # Your actual command goes here, e.g.:
      # python process_data.py --exp "${exp}" --kind "${kind}" --year "${year}"
      
      # --mem=128000
      # # all variables in one job:
      # sbatch --wrap="source /home/b/b382473/.bashrc && conda activate env_gribscan2360 && python eddytransform_run_levante.py ${year} ${kind} ${model} ${exp} ${dataset} ${realm} ${grid} --rw 1" -q nf -t 08:00:00 --mail-type=FAIL --mail-user='matthias.aengenheyster@ecmwf.int' -o /scratch/b/b382473/logs/out/composites_GS_${model}_${exp}_${dataset}_${realm}_${grid}_${kind}_${year}.out -e /scratch/b/b382473/logs/err/composites_GS_${model}_${exp}_${dataset}_${realm}_${grid}_${kind}_${year}.err -J composites_GS_${model}_${exp}_${dataset}_${realm}_${grid}_${kind}_${year} --account=bk1377 --partition=compute --qos=normal
      # by variable:

      # Gulf Stream
      sbatch --wrap="source /home/b/b382473/.bashrc && conda activate env_gribscan2360 && python eddytransform_run_levante.py ${year} ${kind} ${model} ${exp} ${dataset} ${realm} ${grid} --vn ${vn} --rw ${rw}" -q nf -t 08:00:00 --mail-type=FAIL --mail-user='matthias.aengenheyster@ecmwf.int' -o /scratch/b/b382473/logs/out/composites_GS_${model}_${exp}_${dataset}_${realm}_${grid}_${kind}_${vn}_${year}.out -e /scratch/b/b382473/logs/err/composites_GS_${model}_${exp}_${dataset}_${realm}_${grid}_${kind}_${vn}_${year}.err -J composites_GS_${model}_${exp}_${dataset}_${realm}_${grid}_${kind}_${vn}_${year} --account=bk1377 --partition=compute --qos=normal
      # Southern Ocean
      # sbatch --wrap="source /home/b/b382473/.bashrc && conda activate env_gribscan2360 && python eddytransform_run_levante_SO.py ${year} ${kind} ${model} ${exp} ${dataset} ${realm} ${grid} --vn ${vn} --rw ${rw}" -q nf -t 08:00:00 --mail-type=FAIL --mail-user='matthias.aengenheyster@ecmwf.int' -o /scratch/b/b382473/logs/out/composites_SO_${model}_${exp}_${dataset}_${realm}_${grid}_${kind}_${vn}_${year}.out -e /scratch/b/b382473/logs/err/composites_SO_${model}_${exp}_${dataset}_${realm}_${grid}_${kind}_${vn}_${year}.err -J composites_SO_${model}_${exp}_${dataset}_${realm}_${grid}_${kind}_${vn}_${year} --account=bk1377 --partition=compute --qos=normal
      
      # echo "Done submitting varname $vn" 
    echo "Done submitting kind $kind" 
    done
  echo "Done submitting year $year" 
  done
done
# echo "Done submitting exp $exp" 

echo "Done submitting"
echo $model $dataset $realm $grid $years $kinds $varnames

echo "DONE submitting"
