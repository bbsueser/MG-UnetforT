# TeethSeg

MG-UNet is a tooth segmentation tool based on [nnUNet](https://github.com/MIC-DKFZ/nnUNet). This project aims to provide a simple and easy-to-use framework for tooth segmentation tasks.

---

## Project Overview

This project implements tooth segmentation tasks based on the nnUNet framework, including the following modules:
- **Data Preprocessing**: Handles centroid and edge processing.
- **Model Training**: Supports various trainers and encoders (e.g., 3D-UNet and 3D-UNet with residual blocks).
- **Postprocessing**: Executes morphological operations to optimize segmentation results.

---

## Environment Setup

Before using this project, ensure that you have correctly configured the following environment:
- **Python Version**: Install the appropriate Python version as required by nnUNetv2.
- **PyTorch Version**: Refer to the [nnUNetv2 official documentation](https://github.com/MIC-DKFZ/nnUNet) for installation.
- **Other Dependencies**: Follow the nnUNet official guide to install all necessary dependencies.

> **Tip**: It is recommended to refer directly to the [nnUNet official documentation](https://github.com/MIC-DKFZ/nnUNet) for environment setup.

---

## Usage Instructions

### Data Preprocessing

The data preprocessing scripts are located in the `preprocess` folder. These scripts are mainly used to handle centroid and edge processing tasks.

1. Configure the data paths.
2. Run the following command to execute the preprocessing script:
   ```bash
   python preprocess/main.py
### Model Traning
1、You can choose different trainers and encoders for model training. The currently supported encoder options include:
Standard 3D-UNet
3D-UNet with residual blocks (whether it performs better depends on your specific task).
### Postprocessing
1、Configure the output paths.
2、Run the following command to execute the postprocessing script
python postprocess/main.py
