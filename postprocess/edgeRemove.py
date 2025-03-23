# Author: Zhj
# Date: 2025-03-23
# Description: This file is part of the MG-UNet project

import os.path
import numpy as np
import nibabel as nib
from skimage.measure import label, regionprops


def subtract_nii_files(file1_path, file2_path, output_path):
    """
    Subtract the data of two NIfTI files where the second file's values are 1.

    Parameters:
        file1_path (str): Path to the first NIfTI file.
        file2_path (str): Path to the second NIfTI file.
        output_path (str): Path to save the resulting NIfTI file.
    """
    # Load the input NIfTI images
    img1 = nib.load(file1_path)
    img2 = nib.load(file2_path)

    # Get image data
    data1 = img1.get_fdata()
    data2 = img2.get_fdata()

    # Ensure both files have the same shape
    assert data1.shape == data2.shape, "The input NIfTI files must have the same shape."

    # Set all non-zero values in data2 to 1
    data2[data2 > 0] = 1

    # Subtract data2 from data1 where data2 is 1, otherwise keep data1 unchanged
    result = np.where(data1 != 0, data1 - data2, data1)

    # Create a new NIfTI image object with the result
    result_nii = nib.Nifti1Image(result, img1.affine, img1.header)

    # Save the resulting image
    nib.save(result_nii, output_path)


def process_nii(input_path, output_path):
    """
    Process the NIfTI file by applying connected component analysis and filtering based on size.

    Parameters:
        input_path (str): Path to the input NIfTI file.
        output_path (str): Path to save the processed NIfTI file.
    """
    # Load the input NIfTI image
    img = nib.load(input_path)

    # Get image data
    data = img.get_fdata()

    # Apply connected component analysis
    labeld_data, num_features = label(data, connectivity=1, return_num=True)

    # Example threshold, adjust based on your needs
    max_size_threshold = 500

    # Create an array of the same shape as the original image to store filtered labels
    filtered_labels = np.zeros_like(labeld_data)

    # Get properties of labeled regions
    props = regionprops(labeld_data)

    # Iterate through all connected components and retain only those smaller than the threshold
    for prop in props:
        if prop.area <= max_size_threshold:  # Use prop.volume for 3D images if needed
            filtered_labels[labeld_data == prop.label] = prop.label

    # Create a new NIfTI image object with the filtered labels
    result_nii = nib.Nifti1Image(filtered_labels, img.affine, img.header)

    # Save the processed image
    nib.save(result_nii, output_path)


def calculate_dice(file1_path, file2_path, value1, value2):
    """
    Calculate the Dice coefficient between two NIfTI files for specific values.

    Parameters:
        file1_path (str): Path to the first NIfTI file.
        file2_path (str): Path to the second NIfTI file.
        value1 (int): Specific value to extract from the first file.
        value2 (int): Specific value to extract from the second file.

    Returns:
        float: Dice coefficient.
    """
    # Load the input NIfTI images
    img1 = nib.load(file1_path)
    img2 = nib.load(file2_path)

    # Get image data
    data1 = img1.get_fdata()
    data2 = img2.get_fdata()

    # Ensure both files have the same shape
    assert data1.shape == data2.shape, "The input NIfTI files must have the same shape."

    # Extract regions with specific values
    mask1 = (data1 == value1)
    mask2 = (data2 == value2)

    # Calculate Dice coefficient
    intersection = np.sum(mask1 * mask2)
    dice = (2. * intersection + 1e-8) / (np.sum(mask1) + np.sum(mask2) + 1e-8)

    return dice


# # Define paths
#
# # List all files in the directories
# all_Teeth_files = os.listdir(all_Teeth_path)
# boundary_Teeth_files = os.listdir(boundary_Teeth_path)
#
# # Process each pair of files
# for all_Teeth_file, boundary_Teeth in zip(all_Teeth_files, boundary_Teeth_files):
#     file1_path = os.path.join(all_Teeth_path, all_Teeth_file)
#     file2_path = os.path.join(boundary_Teeth_path, boundary_Teeth)
#     output1_path = os.path.join(output_path, all_Teeth_file)
#
#     # Subtract NIfTI files
#     subtract_nii_files(file1_path, file2_path, output1_path)
#
#     # Optionally, uncomment the following lines to process the output further
#     # output2_path = os.path.join(processed_path, all_Teeth_file)
#     # process_nii(output1_path, output2_path)