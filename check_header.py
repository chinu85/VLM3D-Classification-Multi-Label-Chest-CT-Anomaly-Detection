import nibabel as nib
import os

def check_header(file_path):
    img = nib.load(file_path)
    header = img.header
    print(f"--- Header for {os.path.basename(file_path)} ---")
    print(header)
    
    # Check for scl_slope and scl_inter
    print(f"\nscl_slope: {header['scl_slope']}")
    print(f"scl_inter: {header['scl_inter']}")
    
    # Check data range in the raw array vs scaled
    raw_data = img.get_data_dtype()
    print(f"Data Dtype: {raw_data}")

if __name__ == "__main__":
    sample_path = r"c:\Users\shmso\UCD\Spring\Data challanges\dataset\train_1_a_1.nii"
    if os.path.exists(sample_path):
        check_header(sample_path)
