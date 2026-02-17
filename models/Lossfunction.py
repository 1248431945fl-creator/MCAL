import matplotlib
matplotlib.use('Agg')
import torch.nn.functional as F
import torch
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def entropy_loss(f_com, f_spec, eps=1e-6, delta=1e-8):
    def logdet_cov(x):
        # x: (B, D)
        B, D = x.shape
        x_center = x - x.mean(dim=0, keepdim=True)
        cov = (x_center.T @ x_center) / (B - 1)
        # Add diagonal perturbation to ensure invertibility
        cov_reg = cov + eps * torch.eye(D, device=x.device)
        return torch.logdet(cov_reg)  # Scalar

    logdet_com = logdet_cov(f_com)
    logdet_spec = logdet_cov(f_spec)
    diff = torch.abs(logdet_com - logdet_spec)
    loss = 1.0 / (diff + delta)
    return loss


class CenterLoss(nn.Module):
    def __init__(self, num_classes=10, feat_dim=2):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        batch_size = x.size(0)
        centers_batch = self.centers[labels]
        loss = torch.pow(x - centers_batch, 2).sum() / batch_size
        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()