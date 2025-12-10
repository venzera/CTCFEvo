
# Peptide Binding Prediction with ESM Embeddings

This project predicts peptide binding specificity using protein language model embeddings from **ESM (Evolutionary Scale Modeling)** and ensemble machine learning classifiers.

## Project Overview

The notebook demonstrates a workflow to classify peptide sequences as binders or non-binders. It leverages the **ESMC-300M** model to extract rich semantic features from amino acid sequences, which are then fed into various supervised learning models.

## Workflow

1.  **Feature Extraction**:
    *   Uses the `esmc_300m` model from the `esm` library.
    *   Generates embeddings for peptide sequences by mean-pooling residue representations.

2.  **Model Training**:
    *   **Classifiers**: XGBoost, Random Forest, and a **Stacking Classifier** (Super Learner).
    *   **Base Learners** for Stacking: XGBoost, Random Forest, KNN, SVM, Gradient Boosting, AdaBoost.
    *   **Meta Learner**: Logistic Regression.
    *   **Evaluation Strategy**: Leave-One-Out Cross-Validation (LOO-CV) is used to robustly estimate performance on the small dataset.

3.  **Inference**:
    *   Applies the trained ensemble model to predict binding for a set of held-out CTCF homologs and mutants (e.g., *Ciona intestinalis* / Styela).

4.  **Visualization**:
    *   **UMAP**: Implemented via `scanpy` to visualize the manifold of peptide embeddings.
    *   **t-SNE**: implemented via `sklearn` for alternative dimensionality reduction.

## Dependencies

*   `esm` (Evolutionary Scale Modeling)
*   `torch`
*   `pandas`
*   `numpy`
*   `scikit-learn`
*   `xgboost`
*   `scanpy`
*   `matplotlib`

## Setup & Usage

1.  **Install Requirements**:
    ```bash
    pip install esm scanpy xgboost scikit-learn pandas numpy matplotlib
    ```

2.  **Data**:
    *   Ensure `peptide_binding_data.csv` is in the working directory.

3.  **Run**:
    *   Execute the Jupyter notebook cells sequentially. The notebook will download the ESM model weights automatically on the first run.

## Results

The Stacking Classifier typically yields the most robust performance (Accuracy, AUC-ROC, MCC) compared to individual models. The visualization steps provide a qualitative assessment of how well the ESM embeddings separate binders from non-binders.


## References where data was obtained

Li, Y., Haarhuis, J. H., Sedeño Cacciatore, Á., Oldenkamp, R., van Ruiten, M. S., Willems, L., ... & Panne, D. (2020). The structural basis for cohesin–CTCF-anchored loops. Nature, 578(7795), 472-476.

Liu, H., Li, H., Cai, Q., Zhang, J., Zhong, H., Hu, G., ... & Zhang, M. (2025). ANKRD11 binding to cohesin suggests a connection between KBG syndrome and Cornelia de Lange syndrome. Proceedings of the National Academy of Sciences, 122(4), e2417346122
