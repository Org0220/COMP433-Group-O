# Project Overview

We're working on a project for image classification of tissue images into the following classes:

- **Preview Tiles Small Fragmented Tissue**
- **Preview Tiles Small One Piece Tissue**
- **Preview Tiles with No Tissue**
- **Preview Tiles with Some Faint Tissue**
- **Preview Tiles_Faint Tissue**
- **Preview Tiles_Ink Marks**
- **Preview Tiles_Large Solid Tissue**

This project leverages a semi-supervised approach to efficiently use unlabeled data for representation learning and then applies a classification head on top for supervised learning on labeled data.

# Methodology

We are using the **Bootstrap Your Own Latent (BYOL)** methodology on top of an existing **ResNet** model. The training process involves:

1. **Unsupervised Pretraining with BYOL**:  
   The ResNet backbone is first pretrained using BYOL. This allows the model to learn robust representations from the unlabeled dataset of tissue images, reducing the need for large amounts of labeled data.

2. **Supervised Fine-Tuning**:  
   After pretraining, we add a classification head on top of the ResNet backbone. This classification head is trained on a labeled subset of the data for the final classification task.

# Key Components

- **ResNet Backbone**:  
  A high-performance convolutional neural network architecture used for feature extraction.
- **BYOL Training**:  
  BYOL is a self-supervised learning method that learns representations from unlabeled data. It does not require negative pairs or a memory bank. Instead, it uses two networks (online and target) to bootstrap each other’s representations.

- **Classification Head**:  
  A small set of layers (often a fully-connected layer or a small MLP) attached to the pretrained ResNet backbone. This head is trained with labeled samples to distinguish between the tissue classes.

# Setup and Installation

1. **Dependencies**:

   - Python 3.10+
   - PyTorch (with GPU support)
   - Other libraries as indicated in `requirements.txt`

2. **Installation**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Dataset Setup**:
   - The dataset should be provided by the Huron team.
   - Put the folder with Sliced_Images inside the repository directory
   - The folder structure should look like this:
   ```
    HURONPROJECT
    ├── Sliced_Images
    ├── data
    ├── runs
    ├── src
    ...
   ```
   - download this file and put it in the data directory: https://drive.google.com/file/d/1M8l8b1WB66UIohbDBlUV2WDWIcz6nnAu/view?usp=sharing
4. **Training**:
   - run the main.py and it will ask you to enter a name for your working directory, then it will start training the model.
   - The model will be saved in the runs directory with the name you entered.
   - The training process will take a long time, so it is recommended to run it on a machine with a GPU.
5. **Evaluation**:
