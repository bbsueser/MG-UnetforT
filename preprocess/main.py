# Author: Zhj
# Date: 2025-03-23
# Description: This file is part of the MG-UNet project

import argparse
import os
from edgGet import isotropic_gradient  # Assuming this function processes images to get gradients
from erosion import grayscale_erosion  # Assuming this function performs grayscale erosion


def main(label_input_dir, centroid_output_dir, edge_output_dir):
    """
    Main function to process label images to compute centroids and edges.

    Parameters:
        label_input_dir (str): Path to the input directory containing label files.
        centroid_output_dir (str): Path to the output directory for centroids.
        edge_output_dir (str): Path to the output directory for edges.
    """
    # Simple check to ensure directories exist, more detailed error handling may be needed in practice
    if not os.path.isdir(label_input_dir):
        print(f"Error: The label input directory '{label_input_dir}' does not exist.")
        return

    if not os.path.exists(centroid_output_dir):
        os.makedirs(centroid_output_dir)
        print(f"Created output directory for centroids: {centroid_output_dir}")

    if not os.path.exists(edge_output_dir):
        os.makedirs(edge_output_dir)
        print(f"Created output directory for edges: {edge_output_dir}")

    print("Processing...")

    # List all files in the given directory
    files = os.listdir(label_input_dir)

    for file in files:
        file_path = os.path.join(label_input_dir, file)
        if os.path.isfile(file_path):
            isotropic_gradient(file_path, os.path.join(edge_output_dir, file))
            grayscale_erosion(file_path, os.path.join(edge_output_dir, file))
            print(f"Processed {file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process label images to compute centroids and edges.")

    parser.add_argument('label_input_dir', type=str, help='Path to the input directory containing label files')
    parser.add_argument('centroid_output_dir', type=str, help='Path to the output directory for centroids')
    parser.add_argument('edge_output_dir', type=str, help='Path to the output directory for edges')

    args = parser.parse_args()

    main(args.label_input_dir, args.centroid_output_dir, args.edge_output_dir)