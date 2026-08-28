import nibabel as nib
import numpy as np
import os

def analyze_sample(file_path):
    print(f"--- Analyzing Sample: {os.path.basename(file_path)} ---")
    
    # Load image
    img = nib.load(file_path)
    data = img.get_fdata()
    header = img.header
    
    # 1. Spatial Properties
    print(f"Shape: {data.shape}")
    print(f"Dtype: {data.dtype}")
    print(f"Voxel Spacing (mm): {header.get_zooms()[:3]}")
    
    # 2. Intensity Properties (HU Units)
    flat_data = data.flatten()
    print(f"Min HU: {np.min(flat_data):.2f}")
    print(f"Max HU: {np.max(flat_data):.2f}")
    print(f"Mean HU: {np.mean(flat_data):.2f}")
    print(f"Std HU: {np.std(flat_data):.2f}")
    
    # 3. Percentiles (Useful for windowing decisions)
    p = np.percentile(flat_data, [1, 5, 25, 50, 75, 95, 99])
    print(f"Percentiles (1, 5, 25, 50, 75, 95, 99): {p}")
    
    # 4. Check for air/background
    # Typically air is -1000 HU
    air_percentage = np.sum(flat_data < -900) / len(flat_data) * 100
    print(f"Approx. Air/Background Content: {air_percentage:.2f}%")

if __name__ == "__main__":
    sample_path = r"c:\Users\shmso\UCD\Spring\Data challanges\dataset\train_1_a_1.nii"
    if os.path.exists(sample_path):
        analyze_sample(sample_path)
    else:
        print(f"Error: Sample not found at {sample_path}")
