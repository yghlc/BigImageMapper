#!/bin/bash
#SBATCH --job-name=testGPU
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --account=def-tlantz
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lingcaohuang@uvic.ca
#SBATCH --output=%j.out

module purge

module load StdEnv/2020 apptainer/1.1.8


#echo $PATH
#echo $LD_LIBRARY_PATH

echo "== This is the scripting step! =="
./runIN_sing.sh
#which nvidia-smi
#nvidia-smi
echo "== End of Job =="



