#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1

module load gcc/9.3.0 cuda/11.4

source /home/chawla/projects/def-rmsouza/chawla/garbage_project/bin/activate

echo "Script running"

python /home/chawla/projects/def-rmsouza/chawla/img_classification.py

#SBATCH -o /home/chawla/projects/def-rmsouza/chawla/results/train/output.out
#SBATCH -e /home/chawla/projects/def-rmsouza/chawla/results/train/error.err

echo "Script finishing running!"
