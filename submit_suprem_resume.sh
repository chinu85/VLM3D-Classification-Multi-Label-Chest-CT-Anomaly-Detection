#!/bin/bash

# UCD SONIC Slurm Job: Plain resume of SuPreM SwinViT-3D after wall-time kill
# Resumes from best_model_checkpoint_swin_suprem.pth (epoch=83, 0-indexed; best AUROC 0.7834 / F1 0.4251)
# Continues cosine LR schedule from where it was (NO --reset_scheduler — this is a resume, not a new warm restart)
# Weak class ASL gamma override retained: Lung nodule (9), Fibrotic sequela (11), Bronchiectasis (16)
#
# Usage: sbatch submit_suprem_resume.sh
#
#SBATCH --job-name=suprem_resume
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j_suprem_resume.log
#SBATCH --error=logs/%j_suprem_resume.err
#SBATCH --exclude=sonicgpu[1-12]
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sourabh.kumawat@ucdconnect.ie

mkdir -p logs

module load anaconda3/2024.10-1
source /scratch/25208443/vlm3d_env/bin/activate

export PYTHONUNBUFFERED=1

weights_file="/scratch/25208443/pretrain_weights/supervised_suprem_swinunetr_2100.pth"

echo "Resuming SuPreM training (wall-time kill recovery) at: $(date)"
echo "  Checkpoint : best_model_checkpoint_swin_suprem.pth (epoch=83 stored, best AUROC 0.7834)"
echo "  Strategy   : plain resume — cosine LR continues, no scheduler reset"
echo "  Weak class : --weak_class_gamma 2.0 (classes 9, 11, 16)"
echo "  Epochs     : 150 total (will run epochs 85-150, ~66 remaining)"

python -u train.py \
    --use_swin \
    --ssl_weights_path "$weights_file" \
    --checkpoint_path  best_model_checkpoint_swin_suprem.pth \
    --best_model_path  best_model_swin_suprem.pth \
    --num_epochs       150 \
    --lr               1e-4 \
    --weak_class_gamma 2.0

echo "Warm-restart training job finished at: $(date)"
