# Med-Explain: ViT Interpretability on PathMNIST

This repository contains code for finetuning a Vision Transformer (ViT) on the PathMNIST dataset and evaluating its interpretability using CheferCAM and Attention Rollout.

## Repository Structure

### 1. `finetune_vit_base.py`
**Purpose**: Finetunes a pre-trained ViT (`vit_base_patch16_224`) on the PathMNIST dataset.
-   **Functionality**:
    -   Loads the PathMNIST dataset using `medmnist`.
    -   Finetunes the model for classification.
    -   Evaluates performance using Accuracy and AUC.
    -   Saves the best model to `best_model.pth`.

### 2. `generate_explanations.py`
**Purpose**: Generates and visualizes explanation maps for the finetuned model.
-   **Functionality**:
    -   Loads the finetuned model and a subset of the test data (500 samples).
    -   Generates **Attention Rollout** (class-agnostic) maps to visualize partial attention flow.
    -   Generates **CheferCAM** (Transformer Attribution, class-specific) maps using Layer-wise Relevance Propagation (LRP).
    -   Produces a 4-column composite visualization for each sample:
        1.  Original Image with True Label.
        2.  Prediction Class Probabilities (Bar Chart).
        3.  CheferCAM Heatmap Overlay.
        4.  Attention Rollout Heatmap Overlay.

### 3. `perturbation_analysis.py`
**Purpose**: Quantitatively evaluates the faithfulness of the explanations (CheferCAM) using perturbation analysis.
-   **Functionality**:
    -   **MoRF (Most Relevant First)**: Iteratively masks the most important patches identified by CheferCAM and measures the drop in prediction confidence. A steeper drop indicates a better explanation.
    -   **LeRF (Least Relevant First)**: Iteratively masks the least important patches. The prediction confidence should remain stable.
    -   Plots the prediction probability curves for both MoRF and LeRF over a range of masking percentages (0% to 100%).

## Setup & Usage

### Dependencies
-   `torch`, `torchvision`
-   `timm`
-   `medmnist`
-   `opencv-python`
-   `matplotlib`
-   `tqdm`

### Running the Scripts
1.  **Finetune the model**:
    ```bash
    python finetune_vit_base.py
    ```
2.  **Generate Explanations**:
    ```bash
    python generate_explanations.py
    ```
3.  **Run Evaluation**:
    ```bash
    python perturbation_analysis.py
    ```
