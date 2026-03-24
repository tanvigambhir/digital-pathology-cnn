# Deep Learning for Digital Pathology: Detecting Metastatic Cancer

## Project Overview
This project builds an industrial-grade machine learning pipeline to detect metastatic cancer in histopathology scans of lymph node sections. Utilizing PyTorch, the pipeline processes raw biological images to identify subtle morphological phenotypes of cancer that are incredibly difficult to detect with the human eye.

This project was built to demonstrate proficiency in handling unstructured medical data, writing custom PyTorch data loaders, implementing deep learning regularization (Dropout), and evaluating models using strict clinical metrics.

## Dataset & The Custom Pipeline
* **Data Source:** PatchCamelyon (PCam) / Histopathologic Cancer Detection benchmark (Subset of 50,000 images).
* **Data Integrity:** Unlike heavily leaked datasets, PCam is strictly split at the patient level, ensuring zero overlap between training and validation sets. A high score here reflects true biological generalization.
* **Custom `Dataset` Class:** Real-world data is messy. Rather than relying on pre-organized folders, I engineered a custom PyTorch `Dataset` class to dynamically map clinical labels from a CSV to raw `.tif` files on the hard drive, efficiently streaming tensors into memory via batched `DataLoaders`.

## Deep Learning Architecture
* **The Engine:** A custom Convolutional Neural Network (CNN) featuring three convolutional blocks (Conv2d -> ReLU -> MaxPool2d) designed to hierarchically extract cellular structures and tumor microenvironment markers.
* **Regularization (Dropout):** To combat the massive overfitting problem prevalent in medical ML, a `Dropout(p=0.5)` layer was integrated into the fully connected classification head. This forces the network to learn robust, generalized features rather than memorizing training pixels.
* **Optimization:** Adam optimizer paired with Cross-Entropy Loss.

## Clinical Evaluation (ROC / AUC)
In biotech and oncology, accuracy is an insufficient metric due to the severe cost of False Negatives. The model's performance was evaluated using clinical gold standards:
1. **Confusion Matrix:** To track the exact distribution of False Positives vs. False Negatives.
2. **ROC Curve & AUC:** The Area Under the Curve (AUC) was calculated using Softmax probabilities to prove the model's absolute capability to distinguish between healthy and metastatic tissue across varying decision thresholds.

## Tech Stack
* `Python`, `PyTorch`, `Torchvision` (CNN Architecture, Custom DataLoaders, Tensor transformations)
* `Scikit-Learn`, `Seaborn`, `Matplotlib` (Clinical ROC/AUC evaluation and visualizations)
* `Pandas`, `PIL (Pillow)` (Data mapping and raw image processing)

---
*Created by Tanvi Gambhir as a demonstration of applying scalable Deep Learning architectures to complex digital pathology.*
