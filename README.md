# NeuroScan AI: Brain Tumor Detection Portal
An advanced clinical decision support portal utilizing a custom-trained convolutional neural network (CNN) from scratch to analyze cranial MRI scans and identify tumor categories.

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [CNN Architecture & Hyperparameters](#cnn-architecture--hyperparameters)
3. [Data Preprocessing & Augmentation](#data-preprocessing--augmentation)
4. [Explainable AI (Grad-CAM)](#explainable-ai-grad-cam)
5. [Evaluation Results](#evaluation-results)
6. [Security & Validation Safeguards](#security--validation-safeguards)
7. [Installation & Execution Guide](#installation--execution-guide)
8. [Submission Checklist](#submission-checklist)

---

## 🔬 Project Overview
This project is developed as an assignment submission for the Brain Tumor Detection Intern role. The system classifies cranial MRI scans into four distinct categories:
1. **Glioma**: Primary brain tumors originating from glial cells.
2. **Meningioma**: Tumors arising from the meningeal membranes surrounding the brain.
3. **Pituitary Tumor**: Growth occurring in the pituitary gland.
4. **No Tumor**: Healthy brain MRI scan indicating no detectable tumor mass.

The system comprises a custom training script (`train.py`) that designs and trains a CNN from scratch, and an interactive Flask web portal (`app.py`) for live clinical inference.

---

## 🧠 CNN Architecture & Hyperparameters
Following the strict constraints of the assignment, the architecture is trained entirely from scratch (pre-trained networks or transfer learning are completely prohibited).

### Model Summary
The custom CNN utilizes a sequential pipeline containing:
* **Input Layer**: accepts `128x128x3` color MRI inputs.
* **4 Convolutional Blocks**:
  * **Block 1**: 32 filters of `(3,3)`, ReLU, Batch Normalization, `(2,2)` Max Pooling, 25% Dropout.
  * **Block 2**: 64 filters of `(3,3)`, ReLU, Batch Normalization, `(2,2)` Max Pooling, 25% Dropout.
  * **Block 3**: 128 filters of `(3,3)`, ReLU, Batch Normalization, `(2,2)` Max Pooling, 30% Dropout.
  * **Block 4**: 256 filters of `(3,3)`, ReLU, Batch Normalization, `(2,2)` Max Pooling, 30% Dropout.
* **Fully Connected Classifier**:
  * Flatten layer.
  * Dense layer (512 nodes, ReLU), 50% Dropout.
  * Dense layer (256 nodes, ReLU), 40% Dropout.
  * Dense layer (128 nodes, ReLU), 30% Dropout.
  * Output layer (4 nodes, Softmax activation) representing categorical probabilities.

### Training Configuration
* **Optimizer**: Adam Optimizer.
* **Loss Function**: Categorical Cross-Entropy.
* **Epochs**: 30 epochs.
* **Callbacks**:
  * **Early Stopping**: Monitored on validation loss (`patience=5`, restoring best weights).
  * **Model Checkpoint**: Saves the best state as `model.keras` monitored on `val_accuracy`.

---

## 🎨 Data Preprocessing & Augmentation
To enhance generalization and prevent overfitting, the following preprocessing steps are implemented in `train.py`:
1. **Resizing**: All inputs are normalized to `128x128` resolution.
2. **Normalization**: Pixels are scaled to a `[0.0, 1.0]` range.
3. **Dataset Split**: Partitioned into **70% Training**, **15% Validation**, and **15% Testing** using a stratified split to maintain class ratios.
4. **Augmentation Layer**:
   * Horizontal flipping.
   * Random rotations (up to 10%).
   * Random zooming (up to 10%).
   * Random contrast adjustments (up to 10%).
   * **Shifts**: Random translation (up to 10% height/width).
   * **Brightness**: Random brightness modifications (up to 10%).

---

## 🔍 Explainable AI (Grad-CAM)
To align with advanced clinical standards (Bonus Feature), the web application implements **Grad-CAM (Gradient-Weighted Class Activation Mapping)**.
* When a physician uploads an MRI scan, the model calculates the gradients of the top predicted class with respect to the feature map of the last convolutional layer (`conv2d_3`).
* These gradients are globally pooled to calculate the relative channel importances, producing a 2D activation heatmap.
* The heatmap is superimposed on the original scan using a Jet color map, displaying exactly which brain regions the CNN focused on to draw its diagnostic inference.

---

## 📊 Evaluation Results
Training outputs classification performance metrics automatically:
* **Accuracy/Loss curves**: Saved as `accuracy.png` and `loss.png`.
* **Confusion Matrix**: Saved as `confusion_matrix.png`.
* **Classification Report**: Outputs Precision, Recall, and F1-score for all 4 categories.

---

## 🛡️ Security & Validation Safeguards
Developed with senior developer safety patterns:
1. **Size Validation**: The server rejects file uploads larger than **5 MB** using `MAX_CONTENT_LENGTH` checks.
2. **Extension Whitelisting**: Whitelists only `.png`, `.jpg`, and `.jpeg` formats.
3. **Empty File Rejection**: Validates file byte lengths to prevent corrupt, empty uploads.
4. **Filename Collision Prevention**: Hashes uploaded filenames using UUID4 before saving, protecting user data from being overridden.
5. **Path Traversal Shield**: Applies `secure_filename` to prevent path traversal directory attacks.
6. **Error Boundaries**: Inference and CV2 decodings are wrapped in exception catch blocks. If a corrupt file bypasses basic filters, the app displays a clear warning banner rather than crashing.

---

## 🚀 Hugging Face Model Repository
The trained model weights (`model.keras`) are hosted on Hugging Face to keep the Git repository lightweight.
* **Model URL**: [Siddharthv06/BrainTumorAI](https://huggingface.co/Siddharthv06/BrainTumorAI)

The web application handles model retrieval automatically. If `model.keras` is not found locally, the portal will download the weights directly from Hugging Face on its first run.

---

## 🚀 Installation & Execution Guide

### Prerequisites
* Python 3.8 or higher.
* Recommended virtual environment.

### Setup Steps
1. Clone or extract the project repository.
2. Create and activate a python virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Execution
* **To train the model**:
  Place the class-partitioned dataset folders inside `Dataset/` and run:
  ```bash
  python train.py
  ```
* **To start the web application**:
  Run the Flask server. On the first launch, it will automatically download `model.keras` from Hugging Face if it's not present:
  ```bash
  python app.py
  ```
  Open `http://127.0.0.1:5000` in your web browser.

* **To upload/update the model on Hugging Face**:
  To update or re-upload your trained `model.keras` to the Hugging Face repository, use the custom upload helper script:
  ```bash
  python upload_model.py
  ```
  This will prompt you for your Hugging Face write access token and securely handle the upload with a progress bar.

---

## 🎒 Submission Checklist
* [x] **Source Code**: Sanitized `app.py`, `train.py`, templates, and style sheets.
* [x] **Trained Model**: Saved `model.keras` state.
* [x] **Jupyter Notebook**: Available in workspace (see notebooks section).
* [x] **Requirements list**: Saved `requirements.txt`.
* [x] **Documentation**: Full `README.md` details.
* [x] **Bonus Items**: Grad-CAM overlays, prediction logs history, and print-ready PDF reports.
