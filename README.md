# Automated Steel Defect Detection: A Computer Vision Project for Quality Control

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Hugging%20Face-yellow?style=for-the-badge)](https://huggingface.co/spaces/aaburakhia/steel-defect-detection)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

This project is a comprehensive, end-to-end deep learning pipeline designed to automatically identify and segment manufacturing defects on steel surfaces. 

---

### 🚀 Live Interactive Demo

A live version of this tool is deployed on Hugging Face Spaces. You can interact with the model directly here:

[https://huggingface.co/spaces/aaburakhia/steel-defect-detection](https://huggingface.co/spaces/aaburakhia/steel-defect-detection)

---

### 🎯 The Project: Goal and Findings

#### Goal
The objective was to build a robust computer vision model to automate quality control in steel manufacturing. This project bridges deep domain expertise in materials engineering with advanced AI skills, demonstrating a practical, high-value industrial application.

#### Key Findings & Scientific Insights

1.  **High Performance on Industrial Data:** A state-of-the-art U-Net model with an `EfficientNet-B4` backbone was trained using advanced techniques, including a combined Dice + Focal loss function and a cosine annealing learning rate scheduler. This resulted in a strong final **Validation Loss of 0.0824**.

2.  **Successful Multi-Class Segmentation:** The model proved highly effective at not only detecting defects but also correctly classifying them into four distinct categories, even when multiple defect types were present in the same image. The identified classes are: **Pitting / Dots**, **Fine Vertical Lines**, **Scratches / Abrasions**, and **Surface Patches**.

3.  **End-to-End Deployment:** The trained model was successfully packaged into a live, interactive web application using Gradio and deployed on Hugging Face Spaces, proving the ability to deliver a complete, user-ready solution from research to production.

---

### 🛠️ Technical Methodology

-   **Data Source:** The model was trained on the Severstal Steel Defect Detection dataset from Kaggle.
-   **Data Augmentation:** The `albumentations` library was used to apply a pipeline of augmentations (e.g., Flips, Affine transforms, Coarse Dropout) to improve model generalization.
-   **Modeling:** A U-Net architecture with a pre-trained `EfficientNet-B4` backbone was implemented using PyTorch and the `segmentation-models-pytorch` library. The model was trained on a Kaggle GPU, utilizing checkpointing to save the best-performing weights.
-   **Deployment:** The final model and application logic were packaged into a user-friendly interface using **Gradio** and deployed for public access on **Hugging Face Spaces**.

---

### 📂 Repository Structure

The repository is organized as follows:

```
.
├── best_model_advanced.pth   # The final, trained model weights (stored with Git LFS)
├── examples/                 # Example images for the Gradio app
│   ├── 6fa9e1d65.jpg
│   └── ...
├── app.py                    # The Python script for the Gradio application
├── requirements.txt          # Python dependencies for the application
└── Steel Defect Detection.ipynb  # The complete training notebook
```

---
