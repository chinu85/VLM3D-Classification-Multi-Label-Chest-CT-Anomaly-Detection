#!/bin/bash

# UCD SONIC Slurm Job Configuration for Model Ensembling & Threshold Optimization
#SBATCH --job-name=vlm3d_ensemble
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j_ensemble.log
#SBATCH --error=logs/%j_ensemble.err
#SBATCH --exclude=sonicgpu[1-12]
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sourabh.kumawat@ucdconnect.ie

# Create logs directory if it doesn't exist
mkdir -p logs

# Load anaconda module
module load anaconda3/2024.10-1

# Force unbuffered output
export PYTHONUNBUFFERED=1

echo "Starting Model Ensembling & Threshold Optimization job on SONIC at: $(date)"
# Run the ensembling script with equal weights for DenseNet201 and SwinViT-3D
/scratch/25208443/vlm3d_env/bin/python -u ensemble_models.py --densenet_path best_model.pth --swin_path best_model_swin.pth --save_path optimized_thresholds_ensemble.json --weight_densenet 0.5 --weight_swin 0.5
echo "Ensemble job finished at: $(date)"
