#!/bin/bash

# UCD SONIC Slurm Job Configuration for Swin Verification
#SBATCH --job-name=swin_check
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j_swin_check.log
#SBATCH --error=logs/%j_swin_check.err
#SBATCH --exclude=sonicgpu[1-12]

# Load modules
module load anaconda3/2024.10-1

# Run check script
echo "Running SwinUNETR gradient diagnosis on SONIC..."
/scratch/25208443/vlm3d_env/bin/python -u diagnose_swin_gradients.py
echo "Verification finished."
