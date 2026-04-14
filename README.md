# LCA-Net: A Lightweight Context-Attention Network for Hazard Segmentation

This repository contains the official implementation of the paper:
"LCA-Net: A lightweight context-attention network for multi-class hazard segmentation in transmission line corridors"

## 1. Introduction
LCA-Net is a deep learning model specifically designed for identifying potential hazards (such as trees, buildings, and vehicles) in power line corridor environments. It is built upon the DeepLabv3+ framework with a ResNet-18 backbone, incorporating a novel Lightweight Context-Attention (LCA) module to balance segmentation accuracy and real-time performance on edge devices.

## 2. Repository Structure
```text
LCA-Net-Project/
├── data/
│   └── new label.zip         # Re-annotated hazard labels for power line scenes
├── models/
│   └── lca_net.py            # Core architecture of LCA-Net 
├── utils/
│   ├── custom_dataset.py     # Data loading and preprocessing scripts
│   └── loss.py               # Loss functions 
├── train.py                  # Main training script
└── requirements.txt          # Python dependencies
```

## 3. Dataset Preparation
The datasets used in this study are derived from public sources and have been meticulously re-annotated for power line hazard recognition tasks.

Original Images: Please download the raw imagery from the original repository: https://github.com/R3ab/ttpla_dataset

Annotations: We provide the refined labels in the data/folder. Extract new label.zip and ensure the directory structure matches the configuration in custom_dataset.py.

Classes: The model is trained to segment:Person,Tree,Building,Vehicle,Background (other corridor elements)

## 4. Installation
This project requires Python 3.10+ and PyTorch. To install the necessary packages, run:
```text
pip install -r requirements.txt
```

## 5. Usage
To train the LCA-Net from scratch or using pre-trained weights:
```text
python train.py
```

## 6. Replicability Note
In accordance with the Replicable Research Principle of the journal, we provide the full source code and annotation files. Any researcher can reproduce our results by following the steps above.
