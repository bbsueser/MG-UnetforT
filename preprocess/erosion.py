# Author: Zhj
# Date: 2025-03-23
# Description: This file is part of the MG-UNet project

import os
import nibabel as nib
from skimage.morphology import erosion, cube
import numpy as np


def grayscale_erosion(input_path, output_path, selem_size):
    """
    Perform grayscale erosion on a NIfTI image using a cubic structuring element.

    Parameters:
        input_path (str): Path to the input NIfTI file.
        output_path (str): Path where the output NIfTI file will be saved.
        selem_size (int): Size of the cubic structuring element.
    """
    # Load the NIfTI file
    img = nib.load(input_path)

    # Retrieve the image data
    data = img.get_fdata()

    # Define the structuring element
    # Note: For 3D images, cube is used instead of disk to maintain isotropy in all directions
    selem = cube(selem_size)  # Generates a cube structure element

    # Perform grayscale erosion
    eroded_data = erosion(data, selem)

    # Create a new NIfTI image object with the eroded data
    new_img = nib.Nifti1Image(eroded_data, img.affine, img.header)

    # Save the modified NIfTI file to the specified output path
    nib.save(new_img, output_path)


# Function usage
label_path = r'D:\nnUNetv2\nnUNet_raw\Dataset028_alloneTeeth\labelsTr'
output_path = r'D:\nnUNetv2\nnUNet_raw\Dataset047_fushihou38selem3\hou4'
labels = os.listdir(label_path)
selem_size = 6  # Size of the structuring element

# Ensure the output directory exists
if not os.path.exists(output_path):
    os.makedirs(output_path)

for label in labels:
    grayscale_erosion(os.path.join(label_path, label),
                      os.path.join(output_path, label),
                      selem_size)
    print(f"Processed {label} with structuring element size {selem_size}")