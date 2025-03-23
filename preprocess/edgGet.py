# Author: Zhj
# Date: 2025-03-23
# Description: This file is part of the MG-UNet project

import copy
import os
import numpy as np
from nibabel import Nifti1Image
from scipy.ndimage import gaussian_filter, laplace
import nibabel as nib


def isotropic_gradient(nii_path, output_path, sigma=1.0):
    """
    Compute the isotropic gradient of each unique non-zero value in the NIfTI image.

    Parameters:
        nii_path (str): Directory path containing NIfTI files.
        output_path (str): Directory path where processed files will be saved.
        sigma (float): Standard deviation for Gaussian kernel. Default is 1.0.
    """
    # List all files in the given directory
    file_list = os.listdir(nii_path)

    for file in file_list:
        # Load NIfTI file
        nii = nib.load(os.path.join(nii_path, file))
        data = nii.get_fdata()

        # Get unique values from the image data
        unique_values = np.unique(data)

        for value in unique_values:
            if value == 0:
                continue  # Skip zero values

            # Create a deep copy of the original data
            demo = copy.deepcopy(data)

            # Set the current value to 100 for processing
            demo[data == value] = 100

            # Set all other values to 0
            demo[demo != 100] = 0

            # Apply Gaussian filter to smooth the data
            filtered_data = gaussian_filter(demo, sigma=sigma)

            # Apply Laplacian operator to compute the gradient
            gradient = laplace(filtered_data)

            # Thresholding: set specific ranges to fixed values
            gradient[gradient > 8] = 100
            gradient[gradient < 99.999] = 0

            # Create a new NIfTI image object with the gradient data
            output_image = Nifti1Image(gradient, nii.affine)

            # Prepare the output filename
            basename, _ = os.path.splitext(file)
            basename, _ = os.path.splitext(basename)  # Handle double extensions if necessary
            output_filename = os.path.join(output_path, f"{basename}_{int(value)}.nii.gz")

            # Save the output image
            output_image.to_filename(output_filename)
            print(f"Processed image saved to {output_filename}")


# Call the function with specified paths
# isotropic_gradient(r'D:\Dataset005_totalteethSingleTooth\Dataset029_onetoallTeeth\labelsTr',
#                    r'D:\pythonProject\bianyuan\\')