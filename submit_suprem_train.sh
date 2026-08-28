#!/bin/bash

# UCD SONIC Slurm Job Configuration for SuPreM SwinViT-3D Training
#SBATCH --job-name=suprem_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j_suprem_train.log
#SBATCH --error=logs/%j_suprem_train.err
#SBATCH --exclude=sonicgpu[1-12]
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sourabh.kumawat@ucdconnect.ie

# Create logs directory if it doesn't exist
mkdir -p logs

# Load anaconda module
module load anaconda3/2024.10-1

# Verify/Create environment
source /scratch/25208443/vlm3d_env/bin/activate

# 1. Download SuPreM weights if not already present
weights_dir="/scratch/25208443/pretrain_weights"
weights_file="${weights_dir}/supervised_suprem_swinunetr_2100.pth"

if [ ! -f "$weights_file" ]; then
    echo "Downloading SuPreM weights..."
    mkdir -p "$weights_dir"
    wget -O "$weights_file" https://huggingface.co/MrGiovanni/SuPreM/resolve/main/supervised_suprem_swinunetr_2100.pth
else
    echo "SuPreM weights already downloaded."
fi

# 2. Verify SuPreM weights compatibility
echo "Verifying SuPreM weights compatibility..."
python verify_suprem_weights.py --weights_path "$weights_file"
if [ $? -ne 0 ]; then
    echo "ERROR: SuPreM weights verification failed. Aborting training."
    exit 1
fi

# 3. Start training from scratch with SuPreM weights
echo "Starting training job with SuPreM weights at: $(date)"
export PYTHONUNBUFFERED=1

python -u train.py \
    --use_swin \
    --ssl_weights_path "$weights_file" \
    --checkpoint_path best_model_checkpoint_swin_suprem.pth \
    --best_model_path best_model_swin_suprem.pth

echo "Training job finished at: $(date)"
