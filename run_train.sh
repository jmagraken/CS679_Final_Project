#!/bin/bash

# Set resource requirements: Queues are limited to seven day allocations
# Time format: HH:MM:SS
#SBATCH --time=24:15:00
#SBATCH --mem=24GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Set output file destinations (optional)
# By default, output will appear in a file in the submission directory:
# slurm-$job_number.out
# This can be changed:
#SBATCH -o JOB%j.out # File to which STDOUT will be written
#SBATCH -e JOB%j-err.out # File to which STDERR will be written

# email notifications: Get email when your job starts, stops, fails, completes...
# Set email address
#SBATCH --mail-user=igcasaso@uwaterloo.ca
# Set types of notifications (from the options: BEGIN, END, FAIL, REQUEUE, ALL):
#SBATCH --mail-type=ALL

# Load up your conda environment
# Set up environment on watgpu.cs or in interactive session (use `source` keyword instead of `conda`)
source activate imagand

# Task to run
#~/cuda-samples/Samples/5_Domain_Specific/nbody/nbody -benchmark -device=0 -numbodies=16777216
python train.py \
    --save_dir "./output/chemberta_10m_embed_model/" \
    --num_epochs 5000 \
    --noise_type "none" \
    --embed_model "chemberta_10m"