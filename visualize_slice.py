import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

def save_slice(file_path, output_path):
    img = nib.load(file_path)
    data = img.get_fdata()
    
    # Take a central slice (Axial)
    mid_z = data.shape[2] // 2
    slice_data = data[:, :, mid_z]
    
    # Rotate for correct orientation (usually NIfTI is flipped)
    slice_data = np.rot90(slice_data)
    
    plt.figure(figsize=(10, 10))
    # Use a wide colormap to see the range
    plt.imshow(slice_data, cmap='gray')
    plt.colorbar(label='Intensity')
    plt.title(f"Central Axial Slice - {os.path.basename(file_path)}")
    plt.axis('off')
    plt.savefig(output_path)
    plt.close()
    print(f"Slice saved to {output_path}")

if __name__ == "__main__":
    sample_path = r"c:\Users\shmso\UCD\Spring\Data challanges\dataset\train_1_a_1.nii"
    output_img = "sample_slice.png"
    if os.path.exists(sample_path):
        save_slice(sample_path, output_img)
