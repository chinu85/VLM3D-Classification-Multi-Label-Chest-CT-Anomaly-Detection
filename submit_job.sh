#!/bin/bash

# UCD SONIC Slurm Job Configuration
#SBATCH --job-name=vlm3d_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j_train.log
#SBATCH --error=logs/%j_train.err
#SBATCH --exclude=sonicgpu[1-12]
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sourabh.kumawat@ucdconnect.ie

# Create logs directory if it doesn't exist
mkdir -p logs

# Load anaconda module to get base Python
module load anaconda3/2024.10-1

# Automatically create the environment if it doesn't exist yet!
if [ ! -d "/scratch/25208443/vlm3d_env" ]; then
    echo "First time setup: Creating Python environment in /scratch/25208443/vlm3d_env..."
    python -m venv /scratch/25208443/vlm3d_env
    source /scratch/25208443/vlm3d_env/bin/activate
    echo "Installing PyTorch for CUDA 12.4 (this takes a couple minutes)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --no-cache-dir
    echo "Installing dependencies..."
    pip install "monai[all]" nibabel torchmetrics simpleitk h5py rich markdown protobuf werkzeug --no-cache-dir
    echo "Environment setup complete!"
fi

# Debug: verify we are using the correct Python
echo "--- Environment Debug ---"
echo "Python binary: /scratch/25208443/vlm3d_env/bin/python"
/scratch/25208443/vlm3d_env/bin/python --version
/scratch/25208443/vlm3d_env/bin/python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
echo "-------------------------"

# Force unbuffered output so we see everything instantly
export PYTHONUNBUFFERED=1

# Run the training script
echo "Starting training job on SONIC at: $(date)"
/scratch/25208443/vlm3d_env/bin/python -u train.py
echo "Training job finished at: $(date)"
