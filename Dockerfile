# VLM3D Challenge -- Docker Image
# Base: PyTorch 2.6 + CUDA 12.4 (matches training environment exactly)
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

LABEL maintainer="sourabh.kumawat@ucdconnect.ie"
LABEL description="VLM3D Multi-Abnormality Chest CT Classifier -- DenseNet201 + SwinViT-3D Ensemble"
LABEL version="2.0"

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies -- pinned to match training environment
RUN pip install --no-cache-dir \
        "monai[all]==1.4.0" \
        nibabel \
        scipy \
        scikit-learn \
        numpy

# Working directory and inference script
WORKDIR /app
COPY inference.py /app/inference.py

# Model weights and thresholds baked into image at build time.
# Files required in the build context (copy from SONIC before building):
#   best_model.pth                              (DenseNet201,   ~103 MB)
#   best_model_swin_suprem.pth                  (SwinViT-3D,    ~256 MB)
#   optimized_thresholds_ensemble_3model.json   (ensemble thresholds)
RUN mkdir -p /models
COPY best_model.pth                             /models/best_model.pth
COPY best_model_swin_suprem.pth                 /models/best_model_swin_suprem.pth
COPY optimized_thresholds_ensemble_3model.json  /models/optimized_thresholds_ensemble_3model.json

# I/O directories (bind-mounted by the challenge evaluator)
RUN mkdir -p /input /output

# The challenge evaluator runs:  docker run --gpus all <image>
ENTRYPOINT ["python", "/app/inference.py"]
