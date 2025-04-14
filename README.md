# Clustered Federated Learning to Support Context-dependent CSI Decoding

Heasung Kim, The University of Texas at Austin

This repository contains the implementation of "Clustered Federated Learning to support Context-dependent CSI Decoding".


## 📂 Dataset Access
The **CSI dataset** used in our **federated learning** experiments is publicly available at the Texas Data Repository:

🔗 https://doi.org/10.18738/T8/AFEKXT

The dataset consists of eight heterogeneous local CSI datasets, each split into training and test sets.

## 🚀 Quick Start
To get started quickly, please refer to:  `main_notebook.ipynb`

This notebook provides a detailed explanation of the dataset, runs the proposed algorithm, and visualizes the results.






### Note 

It extends the work presented in:

>Kim, Heasung, Hyeji Kim, and Gustavo de Veciana.
Clustered Federated Learning via Gradient-Based Partitioning,
ICML 2024 – Forty-first International Conference on Machine Learning.

This codebase introduces additional components specifically designed for wireless CSI compression and feedback, including dataloaders, custom datasets, and tools for analyzing context-dependent decoder behavior in federated settings.


### References

The following papers and codebases inspired this repository:

>Sattler, Felix, Klaus-Robert Müller, and Wojciech Samek. "Clustered federated learning: Model-agnostic distributed multitask optimization under privacy constraints." IEEE transactions on neural networks and learning systems 32.8 (2020): 3710-3722. https://github.com/felisat/clustered-federated-learning

>Ghosh, Avishek, et al. "An efficient framework for clustered federated learning." Advances in Neural Information Processing Systems 33 (2020): 19586-19597, https://github.com/jichan3751/ifca
