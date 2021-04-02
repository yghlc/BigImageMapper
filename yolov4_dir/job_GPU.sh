#!/bin/bash
#SBATCH --job-name=yolov4
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=24:00:00
#SBATCH --partition=sgpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lingcao.huang@colorado.edu
#SBATCH --output=%j.out

module purge

module load singularity/3.6.4
#module load cuda/10.2

#echo $PATH
#echo $LD_LIBRARY_PATH

echo "== This is the scripting step! =="
./runIN_yoltv4_noconda_sing.sh
#which nvidia-smi
#nvidia-smi
echo "== End of Job =="



