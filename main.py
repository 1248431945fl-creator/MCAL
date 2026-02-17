import os
from torch.backends import cudnn
from torch.utils.data import DataLoader
from data.MODMA_processing import load_mel_features
from models.CDA import CDA
from models.CRD import CRD
from models.Lossfunction import FocalLoss
from uitil.dataloader import MultimodalDataset, evaluate_model
from sklearn.model_selection import StratifiedKFold
import random
import pandas as pd
from models.speech_encoder import SpeechEncoder
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, OneCycleLR
from config import config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

class AuxClassifier(nn.Module):
    def __init__(self, hidden_size, num_classes, dropout=0.3):
        super(AuxClassifier, self).__init__()
        hidden = hidden_size // 2

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden, num_classes, bias=False),
            nn.Dropout(dropout))

    def forward(self, x):
        return self.classifier(x)

class MainClassifier(nn.Module):
    def __init__(self, hidden_size, num_classes, dropout=0.3):
        super(MainClassifier, self).__init__()
        hidden1 = hidden_size
        hidden2 = hidden_size // 2
        hidden3 = hidden_size // 4
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden1, bias=False),
            nn.BatchNorm1d(hidden1),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden1, hidden2, bias=False),
            nn.BatchNorm1d(hidden2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden2, hidden3, bias=False),
            nn.BatchNorm1d(hidden3),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden3, num_classes, bias=False),
            nn.Dropout(dropout))

    def forward(self, x):
        return self.classifier(x)


class MLA(nn.Module):
    """Multi-modal Learning Architecture"""
    def __init__(self, model, lr=1e-3, weight_decay=1e-4, optimizer_type='adamw', scheduler_type='plateau', num_epochs=100, steps_per_epoch=None):
        super(MLA, self).__init__()
        self.model = model
        # ==================== Optimizer Selection ====================
        if optimizer_type == 'adam':
            self.optimizer = optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-8)
        elif optimizer_type == 'adamw':
            # AdamW
            self.optimizer = optim.AdamW(model.parameters(),lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-8)
        elif optimizer_type == 'sgd':
            # SGD with momentum
            self.optimizer = optim.SGD(model.parameters(),lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
        elif optimizer_type == 'radam':
            # RAdam - Adaptive learning rate
            self.optimizer = optim.RAdam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

        # ==================== Learning Rate Scheduler ====================
        if scheduler_type == 'cosine':
            # Cosine Annealing with Warm Restarts
            self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2, eta_min=1e-6 )
        elif scheduler_type == 'plateau':
            # ReduceLROnPlateau - Adaptive based on validation metrics
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6)
        elif scheduler_type == 'onecycle':
            # OneCycleLR - Super-convergence strategy
            if steps_per_epoch is None:
                raise ValueError("OneCycleLR requires steps_per_epoch")
            self.scheduler = OneCycleLR(self.optimizer, max_lr=lr * 10, epochs=num_epochs,
                steps_per_epoch=steps_per_epoch, pct_start=0.3, anneal_strategy='cos', div_factor=25.0, final_div_factor=1e4)
        elif scheduler_type == 'step':
            # StepLR - Step-wise decay
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,step_size=30,gamma=0.1 )
        elif scheduler_type == 'none':
            self.scheduler = None
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")

        self.scheduler_type = scheduler_type
        # ==================== Loss Weights (Tunable Parameters) ====================
        self.loss_weights = {
            'main': 1.0,  # Main loss weight
            'aux': 1.0,  # Auxiliary loss weight
            'causal': 0.05,  # Causal loss weight[0.01, 0.05, 0.1, 0.5]
            'dist': 0.4,  # Distance loss weight[0.1, 0.4, 0.7, 1]
            'cent': 1  # Center loss weight[0.1, 0.4, 0.7, 1]
        }
        # Gradient clipping threshold
        self.max_grad_norm = 1.0

    def forward(self, x, a, labels, mode=True):
        # Forward propagation
        logits, loss_main, aux_loss, loss_cent, loss_dist, loss_causal = self.model(x, a, labels)
        # ==================== Loss Calculation ====================
        loss_total = (self.loss_weights['main'] * loss_main +self.loss_weights['aux'] * aux_loss +
                self.loss_weights['causal'] * loss_causal + self.loss_weights['dist'] * loss_dist +
                self.loss_weights['cent'] * loss_cent)
        # ==================== Training Mode: Backpropagation and Optimization ====================
        if mode:
            # Backpropagation
            loss_total.backward(retain_graph=True)
            # Gradient clipping (prevent gradient explosion)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.max_grad_norm)
            # Optimizer update
            self.optimizer.step()
            self.optimizer.zero_grad()
        return logits, loss_main, aux_loss, loss_cent, loss_dist, loss_causal, loss_total

    def step_scheduler(self, metric=None):
        if self.scheduler is None:
            return

        if self.scheduler_type == 'plateau':
            if metric is None:
                raise ValueError("ReduceLROnPlateau requires metric")
            self.scheduler.step(metric)
        elif self.scheduler_type == 'onecycle':
            # OneCycle updates after each batch
            pass  # Called manually in training loop
        else:
            self.scheduler.step()

    def get_lr(self):
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']

    def set_loss_weights(self, **kwargs):
        """Dynamically adjust loss weights"""
        for key, value in kwargs.items():
            if key in self.loss_weights:
                self.loss_weights[key] = value
                print(f"Updated loss weight '{key}': {value}")


class MCAL(nn.Module):
    def __init__(self,data_cfg,model_cfg, roi_mask_cfg):
        super(MCAL, self).__init__()
        n_brain_regions = data_cfg["n_brain_regions"]
        n_speech_features = data_cfg["n_speech_features"]
        speech_channels = data_cfg["speech_channels"]
        hidden_size = model_cfg["hidden_size"]
        roi_mask_kwargs = roi_mask_cfg
        self.num_classes = model_cfg["num_classes"]
        self.speech_encoder = SpeechEncoder(speech_channels=speech_channels, hidden_size=hidden_size)
        self.CRD = CRD(n_brain_regions=n_brain_regions,n_speech_features=n_speech_features, speech_channels=speech_channels, hidden_size=hidden_size,**roi_mask_kwargs)
        self.CDA = CDA(n_brain_regions=data_cfg["n_brain_regions"], n_speech_features=data_cfg["n_speech_features"],
            speech_channels=data_cfg["speech_channels"], hidden_size=model_cfg["hidden_size"], num_classes=model_cfg["num_classes"], roi_mask_kwargs=roi_mask_cfg).cuda()
      # ------------------- Loss Functions --------------
        # Auxiliary classifiers
        self.aux_brain_spec = AuxClassifier(hidden_size, self.num_classes, dropout=0.1)
        self.aux_audio_spec = AuxClassifier(hidden_size, self.num_classes, dropout=0.1)
        # Main classifier
        self.classifier = MainClassifier( hidden_size, self.num_classes, dropout=0.1)
        # --------------------------------
        # self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.criterion = FocalLoss(gamma=2)
    def forward(self, b, a, label):
        # Speech feature extraction
        a_,a_common,a_specific = self.speech_encoder(a)
        # Apply causal ROI mask
        masked_pcc, loss_causal, ni, importance_matrix = self.CRD(b, a_)

        a_distinct, a_shared, b_distinct, b_shared, feat_avg_shared, loss_cent, loss_dist = self.CDA(masked_pcc, a_common,a_specific, label)

        # -------- Auxiliary Classifiers -------------------------------
        pred_brain_spec = self.aux_brain_spec(b_distinct)
        pred_audio_spec = self.aux_audio_spec(a_distinct)
        loss_brain_spec = self.criterion(pred_brain_spec, label)
        loss_audio_spec = self.criterion(pred_audio_spec, label)
        aux_loss = loss_brain_spec + loss_audio_spec
        # -------- Main Classifier -------------------------------
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, device=device)
        if label.dim() == 0:
            label = label.unsqueeze(0)
        # -----------------------------------------------
        logits = self.classifier(feat_avg_shared)
        loss_main = self.criterion(logits, label)
        # -----------------------------------------------

        return logits, loss_main, aux_loss, loss_cent, loss_dist, loss_causal


def train_with_dataloader(train_data, audio_feature, train_label, train_cfg, MCAL_model):
    kf = StratifiedKFold(n_splits=train_cfg["n_splits"], random_state=train_cfg["random_seed"], shuffle=True)
    all_results = []
    kfold_index = 0

    for train_idx, test_idx in kf.split(train_data, train_label):
        setup_seed(train_cfg["random_seed"])
        print(f"Starting fold {kfold_index + 1} cross-validation")
        # Model re-initiation
        data_cfg = config["data"]
        train_cfg = config["training"]
        model_cfg = config["model"]
        roi_mask_cfg = model_cfg["roi_mask"]
        MCAL_model = MCAL(data_cfg, model_cfg, roi_mask_cfg)
        # Samples
        X_train, X_test = train_data[train_idx], train_data[test_idx]
        A_train, A_test = audio_feature[train_idx], audio_feature[test_idx]
        Y_train, Y_test = train_label[train_idx], train_label[test_idx]
        # DataLoader
        train_dataset = MultimodalDataset(X_train, A_train, Y_train)
        test_dataset = MultimodalDataset(X_test, A_test, Y_test)
        train_loader = DataLoader(train_dataset, batch_size=train_cfg["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
        # Train
        mla = MLA(model=MCAL_model, lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"], optimizer_type='adam', scheduler_type='plateau',steps_per_epoch=None).cuda()
        total_params = sum(p.numel() for p in mla.parameters())
        trainable_params = sum(p.numel() for p in mla.parameters() if p.requires_grad)
        print(f"Model parameters - Total: {total_params}, Trainable: {trainable_params}")

        for epoch in range(train_cfg["num_epochs"]):
            mla.train()
            epoch_losses = {'loss_total': [], 'loss_main': [], 'aux_loss': [], 'loss_cent': [], 'loss_dist': [], 'loss_causal': []}

            for batch_idx, batch_data in enumerate(train_loader):
                brain_batch = batch_data['brain'].cuda()
                audio_batch = batch_data['audio'].cuda()
                label_batch = batch_data['label'].cuda()

                logits, loss_main, aux_loss, loss_cent, loss_dist, loss_causal, loss_total = mla(brain_batch,audio_batch,label_batch)

                epoch_losses['loss_total'].append(loss_total.item())
                epoch_losses['loss_main'].append(loss_main.item())
                epoch_losses['aux_loss'].append(aux_loss.item())
                epoch_losses['loss_cent'].append(loss_cent.item())
                epoch_losses['loss_dist'].append(loss_dist.item())
                epoch_losses['loss_causal'].append(loss_causal.item())

            avg_losses = {k: sum(v) / len(v) for k, v in epoch_losses.items()}

            print(f'\n{"=" * 80}')
            print(f'Epoch [{epoch + 1}/{train_cfg["num_epochs"]}] Training Losses:')
            print(f'  Total Loss: {avg_losses["loss_total"]:.6f}')
            print(f'  Loss main: {avg_losses["loss_main"]:.6f}')
            print(f'  Loss aux: {avg_losses["aux_loss"]:.6f}')
            print(f'  Loss center: {avg_losses["loss_cent"]:.6f}')
            print(f'  Loss Dist: {avg_losses["loss_dist"]:.6f}')
            print(f'  Loss Causal: {avg_losses["loss_causal"]:.6f}')
            print(f'{"=" * 80}')

            train_acc = evaluate_model(mla, train_loader, mode='train')
            print(f'Train Accuracy: {train_acc:.4f}')
        # Test
        test_acc, test_precision, test_recall, test_f1, all_preds, all_labels = evaluate_model(mla, test_loader, mode='test')
        all_results.append([kfold_index + 1, test_acc, test_precision, test_recall, test_f1])
        # Print
        all_preds_clean = [int(x) for x in all_preds]
        all_labels_clean = [int(x) for x in all_labels]
        print("=" * 80)
        print("Final Test Results:")
        print(f"Predictions: {all_preds_clean}")
        print(f"Labels: {all_labels_clean}")
        print(f'Test Accuracy: {test_acc:.4f}')
        print(f'Test Precision: {test_precision:.4f}')
        print(f'Test Recall: {test_recall:.4f}')
        print(f'Test F1-Score: {test_f1:.4f}')
        print("=" * 80)

        del mla
        del MCAL_model
        torch.cuda.empty_cache()
        kfold_index += 1

    return all_results


def compare_labels(labels, train_label):
    labels_array = np.array(labels)
    train_label_array = np.array(train_label)

    if labels_array.shape != train_label_array.shape:
        print(f"Different shapes: labels{labels_array.shape} vs train_label{train_label_array.shape}")
        return False

    is_equal = np.array_equal(labels_array, train_label_array)

    if is_equal:
        print("✓ labels and train_label are completely equal")
    else:
        num_diff = np.sum(labels_array != train_label_array)
        print(f"✗ labels and train_label are not equal, {num_diff} positions differ")

    return is_equal


def main():
    # Parameters
    data_cfg = config["data"]
    train_cfg = config["training"]
    model_cfg = config["model"]
    roi_mask_cfg = model_cfg["roi_mask"]
    # Model initialization
    MCAL_model = MCAL(data_cfg, model_cfg, roi_mask_cfg)
    print(MCAL_model)

    # Load data
    train_data = np.load('./data/MODMA_EEG_pcc.npy')
    features, labels, info = load_mel_features('./data/mel_features_data.npy')
    sample, seg_number, bins, frame = features.shape
    audio_feature = features.reshape(-1,bins, frame)
    train_label = np.load('./data/MODMA_EEG_label.npy')
    result = compare_labels(labels, train_label)
    if result:
        print("Labels are completely equal")
    else:
        print("Labels are not equal")

    all_results = train_with_dataloader(train_data=train_data, audio_feature=audio_feature,train_label=train_label, train_cfg=train_cfg, MCAL_model=MCAL_model)
    if all_results:
        print("\nFinal Results Summary:")
        df = pd.DataFrame(all_results, columns=['epoch', 'acc', 'precision', 'recall', 'f1'])
        print(df)
        print("\nMean per epoch:")
        print(df.groupby('epoch').mean())
        print(f"\nOverall mean:")
        print(df[['acc', 'precision', 'recall', 'f1']].mean())

def init_seeds(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

if __name__ == "__main__":
    init_seeds(1)
    device_index = 0  # Target GPU index
    torch.cuda.set_device(device_index)
    torch.set_default_dtype(torch.float32)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    main()