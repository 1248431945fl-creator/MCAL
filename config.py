# config.py
# This file centralizes the management of hyperparameters, excluding model logic
import torch

config = {
    # ---------- Data-related ----------
    "data": {
        "fmri_path": "<fMRI data path>",               # Path to fMRI PCC data, replace with actual path
        "audio_path": "<audio feature data path>",      # Path to audio features, replace with actual path
        "label_path": "<label data path>",              # Path to labels, replace with actual path
        "n_brain_regions": "<number of brain regions>", # Number of brain regions (e.g., 90)
        "n_speech_features": "<speech feature dimension>",  # Dimension of compressed speech features (e.g., 64)
        "speech_channels": "<speech feature channels>", # Number of speech feature channels (e.g., 64)
        "speech_length": "<speech feature time length>",  # Length of speech features (e.g., 640)
    },

    # ---------- Training-related ----------
    "training": {
        "batch_size": "<batch size>",                  # Batch size (e.g., 16)
        "num_epochs": "<number of training epochs>",    # Number of training epochs (e.g., 51)
        "n_splits": "<number of cross-validation splits>",  # Number of cross-validation splits (e.g., 5)
        "random_seed": "<random seed>",                 # Random seed for reproducibility
        "device": "cuda:<GPU number>" if torch.cuda.is_available() else "cpu",  # Device selection (auto chooses GPU or CPU)
        "lr": "<learning rate>",                        # Learning rate (e.g., 0.0001)
        "weight_decay": "<weight decay>",               # Weight decay (e.g., 0.0005)
    },

    # ---------- Model structure-related ----------
    "model": {
        "hidden_size": "<hidden layer size>",           # Hidden layer size (e.g., 128)
        "num_classes": "<number of classes>",           # Number of classes (e.g., 2, MDD/HC)

        # Hyperparameters for ROI Mask module
        "roi_mask": {
            "roi_sparsity": "<sparsity regularization coefficient>",  # Sparsity regularization coefficient (e.g., 0.1)
            "smooth_lambda": "<smoothing regularization coefficient>", # Smoothing regularization coefficient (e.g., 0.05)
            "lambda_p": "<indirect effect weight>",                   # Indirect effect weight (e.g., 0.5)
            "damping_factor": "<PageRank damping factor>",            # PageRank damping factor (e.g., 0.85)
            "grandag_hidden": "<GraNDAG hidden layer size>",           # GraNDAG hidden layer size (e.g., 64)
            "grandag_layers": "<GraNDAG layers>",                     # Number of GraNDAG layers (e.g., 3)
            "grandag_lambda_a": "<GraNDAG Lagrange multiplier>",      # GraNDAG Lagrange multiplier (e.g., 1.0)
            "grandag_rho_init": "<GraNDAG penalty parameter initial>", # GraNDAG penalty parameter initial (e.g., 1.0)
            "grandag_rho_factor": "<GraNDAG penalty parameter update factor>",  # GraNDAG penalty parameter update factor (e.g., 2.0)
        }
    }
}
