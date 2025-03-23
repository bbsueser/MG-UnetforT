# Author: Zhj
# Date: 2025-03-23
# Description: This file is part of the MG-UNet project

import numpy as np
import nibabel as nib
import scipy.ndimage as ndimage
from skimage import measure, segmentation
import os
import matplotlib.pyplot as plt


def load_nii(file_path):
    """
    Load a NIfTI file and return the data and metadata.

    Parameters:
        file_path (str): Path to the NIfTI file.

    Returns:
        tuple: A tuple containing the image data and the NIfTI object.
    """
    nii = nib.load(file_path)
    data = nii.get_fdata()
    return data, nii


def get_centroids(labeled_image):
    """
    Compute centroids for each connected component in the labeled image.

    Parameters:
        labeled_image (np.ndarray): Image with different gray values representing different objects.

    Returns:
        list: List of centroid coordinates for each connected component.
    """
    labeled_image = labeled_image.astype(np.int32)
    props = measure.regionprops(labeled_image)
    centroids = [prop.centroid for prop in props]
    return centroids


def create_markers(centroids, shape):
    """
    Create a marker image where each centroid is used as a seed point.

    Parameters:
        centroids (list): List of centroid coordinates.
        shape (tuple): Shape of the marker image.

    Returns:
        np.ndarray: Marker image.
    """
    markers = np.zeros(shape, dtype=int)
    for i, centroid in enumerate(centroids):
        markers[int(centroid[0]), int(centroid[1]), int(centroid[2])] = i
    return markers


def apply_watershed(binary_image, markers):
    """
    Apply the watershed algorithm on the binary image using the markers.

    Parameters:
        binary_image (np.ndarray): Binary image.
        markers (np.ndarray): Marker image.

    Returns:
        np.ndarray: Segmented image.
    """
    distance = ndimage.distance_transform_edt(binary_image)
    labels = segmentation.watershed(-distance, markers, mask=binary_image)

    max_size_threshold = 200  # Example threshold, adjust based on your needs

    # Create an array of the same shape as the original image to store filtered labels
    filtered_labels = np.zeros_like(labels)
    props = measure.regionprops(labels)

    # Iterate through all connected components and retain only those smaller than the threshold
    for prop in props:
        if prop.area > max_size_threshold:  # Use prop.volume for 3D images if needed
            filtered_labels[labels == prop.label] = prop.label

    return filtered_labels


def save_nii(data, ref_nii, output_path):
    """
    Save the segmented result as a NIfTI file.

    Parameters:
        data (np.ndarray): Segmentation data.
        ref_nii (nib.Nifti1Image): Reference NIfTI image for affine and header information.
        output_path (str): Path to save the output file.
    """
    new_nii = nib.Nifti1Image(data, affine=ref_nii.affine, header=ref_nii.header)
    nib.save(new_nii, output_path)
    print(f"Processed image saved to {output_path}")


def multi_waterShed(semantic_seg, centroids, output_dir):
    """
    Perform multi-watershed segmentation using semantic segmentation and centroid seeds.

    Parameters:
        semantic_seg (str): Path to the semantic segmentation NIfTI file.
        centroids (str): Path to the centroid seeds NIfTI file.
        output_dir (str): Directory to save the output files.
    """
    binary_nii_path = os.path.join(semantic_seg)
    labeled_nii_path = os.path.join(centroids)
    output_file = os.path.join(output_dir)

    binary_image, binary_nii = load_nii(binary_nii_path)
    labeled_image, _ = load_nii(labeled_nii_path)

    # Compute centroids for each connected component
    centroids = get_centroids(labeled_image)

    # Create marker image
    markers = create_markers(centroids, binary_image.shape)

    # Apply watershed algorithm
    labels = apply_watershed(binary_image, markers)

    # Save segmentation results
    save_nii(labels, binary_nii, output_file)

