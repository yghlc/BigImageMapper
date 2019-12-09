#!/bin/bash
#SBATCH -J install
#SBATCH -N 1 
#SBATCH --gres=gpu:GTX1080Ti:1
#SBATCH --mem-per-cpu=4G
#SBATCH -c 4


# the task
./run_INsingularity_itsc_miniconda.sh
