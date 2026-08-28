#!/bin/bash

# UCD SONIC Slurm Job: 2-Model Ensemble Evaluation (DenseNet201 + SuPreM SwinViT-3D)
#
# Runs 8-fold TTA on the full validation set for both models, then:
#   1. Finds optimal ensemble weights via Nelder-Mead (maximises macro AUROC)
#   2. Per-class threshold sweep to maximise macro F1
#   3. Saves results to optimized_thresholds_ensemble_suprem.json
#      (inference.py reads weights + thresholds directly from this file)
#
# Run order:
#   Step 1: sbatch submit_ensemble_suprem.sh         # after any training job completes
#   Step 2: copy optimized_thresholds_ensemble_suprem.json -> Docker /models/
#
# Usage: sbatch submit_ensemble_suprem.sh
#
#SBATCH --job-name=ensemble_suprem
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=logs/%j_ensemble.log
#SBATCH --error=logs/%j_ensemble.err
#SBATCH --exclude=sonicgpu[1-12]
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sourabh.kumawat@ucdconnect.ie

mkdir -p logs

module load anaconda3/2024.10-1
source /scratch/25208443/vlm3d_env/bin/activate

export PYTHONUNBUFFERED=1

echo "Starting ensemble evaluation (DenseNet201 + SuPreM SwinViT) at: $(date)"
echo "  DenseNet : best_model.pth"
echo "  SwinViT  : best_model_swin_suprem.pth"
echo "  Weights  : Nelder-Mead optimised (--optimize_weights)"

python -u ensemble_models.py \
    --densenet_path   best_model.pth \
    --swin_path       best_model_swin_suprem.pth \
    --save_path       optimized_thresholds_ensemble_suprem.json \
    --optimize_weights

echo "Ensemble job finished at: $(date)"
