# config.py
# This file centrally manages all hyperparameters, excluding model logic
import torch

config = {
    # ---------- Data Configuration ----------
    "data": {
        "fmri_path": "./fMri_pcc.npy",               # fMRI PCC data path
        "audio_path": "stft_data_fixed_10/all_stfts_64x640.npy",  # Speech feature path
        "label_path": "./fMri_label.npy",            # Label path
        "n_brain_regions": 128,                      # Number of brain regions
        "n_speech_features": 64,                     # Compressed speech feature dimension
        "speech_channels": 64,                       # AudioInfoCollect output channels
        "speech_length": 640,                        # Speech feature temporal length
    },

    # ---------- Training Configuration ----------
    "training": {
        "batch_size": 64,
        "num_epochs": 50,
        "n_splits": 5,                               # K-fold cross-validation folds
        "random_seed": 63,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "lr": 0.0001,                               # Learning rate
        "weight_decay": 0.0001,                      # Weight decay
    },

    # ---------- Model Architecture Configuration ----------
    "model": {
        "hidden_size": 128,                          # Hidden layer dimension
        "num_classes": 2,                            # Number of classes (MDD/HC)

        # ROI mask module specific hyperparameters
        "roi_mask": {
            "roi_sparsity": 0.1,                     # Sparsity regularization coefficient
            "smooth_lambda": 0.05,                   # Smoothness regularization coefficient
            "lambda_p": 0.5,                         # Indirect effect weight
            "damping_factor": 0.85,                  # PageRank damping factor
            "grandag_hidden": 10,                    # GraNDAG hidden layer dimension
            "grandag_layers": 2,                     # Number of GraNDAG layers
            "grandag_lambda_a": 0,                 # GraNDAG Lagrange multiplier
            "grandag_rho_init": 0.001,                 # GraNDAG penalty parameter initial value
            "grandag_rho_factor": 10,               # GraNDAG penalty parameter update factor
        }
    }

}
