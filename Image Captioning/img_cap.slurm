#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:40:00
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1

module load gcc/9.3.0 arrow/11 cuda/11.4

source /home/chawla/projects/def-rmsouza/chawla/garbage_project/bin/activate

echo "Script running"

python /home/chawla/projects/def-rmsouza/chawla/image_cap.py 

#SBATCH -o /home/chawla/projects/def-rmsouza/chawla/results/train/output.out
#SBATCH -e /home/chawla/projects/def-rmsouza/chawla/results/train/error.err

echo "Script finishing running!"
