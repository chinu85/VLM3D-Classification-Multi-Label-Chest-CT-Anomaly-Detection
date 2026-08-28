#!/bin/bash

# UCD SONIC Slurm Job Configuration for Threshold Optimization
#SBATCH --job-name=vlm3d_optimize
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j_optimize.log
#SBATCH --error=logs/%j_optimize.err
#SBATCH --exclude=sonicgpu[1-12]
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sourabh.kumawat@ucdconnect.ie

# Create logs directory if it doesn't exist
mkdir -p logs

# Load anaconda module
module load anaconda3/2024.10-1

# Force unbuffered output
export PYTHONUNBUFFERED=1

echo "Starting Swin threshold optimization job on SONIC at: $(date)"
# Run threshold optimization on the best Swin model checkpoint
/scratch/25208443/vlm3d_env/bin/python -u optimize_thresholds.py --use_swin
echo "Optimization job finished at: $(date)"
