# Author: Zhj
# Date: 2025-03-23
# Description: This file is part of the MG-UNet project

import os
import argparse
import numpy as np
from labelsprocess import labelsprocess
from MultiWaterShed import multi_waterShed
from edgeRemove import subtract_nii_files


def main(centroid_input_dir, centroid_output_dir, edge_input_dir, semantic_seg_input_dir,
         instance_seg_struct_input_dir):
    """
    Main function to process medical imaging data.

    Parameters:
        centroid_input_dir (str): Path to the input directory containing centroid files.
        centroid_output_dir (str): Path to the output directory for processed centroid files.
        edge_input_dir (str): Path to the input directory containing edge files.
        semantic_seg_input_dir (str): Path to the input directory containing semantic segmentation files.
        instance_seg_struct_input_dir (str): Path to the input directory containing instance segmentation structure files.
    """
    # List all files in the centroid input directory
    files = os.listdir(centroid_input_dir)

    # Define a temporary directory for edge-removed files
    remove_edge = "/"

    for file in files:
        # Construct full paths for each file based on their respective directories
        centroid_input_path = os.path.join(centroid_input_dir, file)
        centroid_output_path = os.path.join(centroid_output_dir, file)
        semantic_seg_input_path = os.path.join(semantic_seg_input_dir, file)
        edge_input_path = os.path.join(edge_input_dir, file)

        # Process centroids
        labelsprocess(centroid_input_path, centroid_output_path)

        # Subtract edges from semantic segmentation images
        subtract_nii_files(semantic_seg_input_path, edge_input_path, os.path.join(remove_edge, file))

        # Perform multi-watershed segmentation
        multi_waterShed(os.path.join(remove_edge, file), centroid_output_path,
                        os.path.join(instance_seg_struct_input_dir, file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process medical imaging data.")

    # Add arguments for each directory path
    parser.add_argument('centroid_input_dir', type=str, help='Path to the input directory containing centroid files.')
    parser.add_argument('centroid_output_dir', type=str,
                        help='Path to the output directory for processed centroid files.')
    parser.add_argument('edge_input_dir', type=str, help='Path to the input directory containing edge files.')
    parser.add_argument('semantic_seg_input_dir', type=str,
                        help='Path to the input directory containing semantic segmentation files.')
    parser.add_argument('instance_seg_struct_input_dir', type=str,
                        help='Path to the input directory containing instance segmentation structure files.')

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args.centroid_input_dir, args.centroid_output_dir, args.edge_input_dir,
         args.semantic_seg_input_dir, args.instance_seg_struct_input_dir)