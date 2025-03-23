# Author: Zhj
# Date: 2025-03-23
# Description: This file is part of the MG-UNet project

import os
import numpy as np
import nibabel as nib
from skimage.measure import label as measure_label, regionprops


def labelsprocess(files_path, output_path):
    """
    Process the NIfTI files in the given directory by applying connected component analysis.

    Parameters:
        files_path (str): Path to the input directory containing NIfTI files.
        output_path (str): Path to the output directory where processed files will be saved.
    """
    # List all files in the input directory
    files = os.listdir(files_path)

    for file in files:
        nii_file = os.path.join(files_path, file)
        img = nib.load(nii_file)

        # Load image data, affine transformation matrix, and header information
        data = img.get_fdata()  # Get image data
        affine = img.affine  # Get affine transformation matrix
        header = img.header  # Get header information

        # Apply thresholding: set non-zero values to 1 for further processing
        mask = data == 1
        data[mask] = 1

        labels_out = None


        # Set a maximum voxel count threshold for connected components
        # max_size_threshold = max(data.shape)  # Example threshold, adjust based on your needs
        max_size_threshold = 600
        # Create an array of the same shape as the original image to store filtered labels
        filtered_labels = np.zeros_like(labels_out)
        props = regionprops(labels_out)

        # Iterate through all connected components and retain only those smaller than the threshold
        for prop in props:
            if prop.area <= max_size_threshold:  # Use prop.volume for 3D images if needed
                filtered_labels[labels_out == prop.label] = prop.label

        # Now, each unique non-zero value in `filtered_labels` represents a different connected region (tooth)

        # Create a new Nifti1Image object with the filtered labels
        new_img = nib.Nifti1Image(filtered_labels, affine=affine, header=header)

        # Save the processed image
        nib.save(new_img, os.path.join(output_path, file))


def TobeOne(input_nii, output_nii):
    """
    Convert all non-zero voxels in the input NIfTI file to 1 and save the modified image.

    Parameters:
        input_nii (str): Path to the input NIfTI file.
        output_nii (str): Path to save the processed NIfTI file.
    """
    # Load the input NIfTI image
    img = nib.load(input_nii)

    # Load image data and affine transformation matrix
    data = img.get_fdata()
    affine = img.affine

    # Apply thresholding: set all non-zero values to 1
    mask = data > 0
    data[mask] = 1

    # Create a new Nifti1Image object with the modified data
    modified_label = nib.Nifti1Image(data, affine=affine)

    # Save the modified image
    nib.save(modified_label, output_nii)

    print(f"{input_nii} has been processed.")

# Note: Ensure that the necessary modules are imported at the beginning of the script.