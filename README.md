# Nuclei classification using CNN and GNNs

This code classifies nuclei based on DAPI fluorescence using a convolutional neural network or a graph neural network. The CNN is based on the standard ResNet34 archietecture. The GNN classification is based on calculated morphological features for each nuclei. The original use case for this code was a lung cancer image dataset consisting of more than 10000 nuclei images containing cancerous or normal lung cells. Work is in progress to extend the nuclei classification to identify cell types within the lung tissue.

## Getting Started

### Prerequisites
The following packages are required to run the code:
1. numpy 1.26.0
2. scikit-learn 1.3.0
3. h5py 3.8.0
4. pytorch 1.12.1
5. torch-geometric 2.3.0

## Running the code

To run this code, run either main_cnn.py or main_gnn.py

## Authors

* **Jen Yi Wong**

## Acknowledgments

* This work was done as a research intern in Prof. Russell Schwartz's lab
