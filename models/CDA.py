import matplotlib

from models.Lossfunction import CenterLoss, entropy_loss

matplotlib.use('Agg')
from models.brain_encoder import BrainEncoder
from models.Class_wise_alignment import ClassWiseDistributionAlignment
import torch
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CDA(nn.Module):
    def __init__(self, n_brain_regions, n_speech_features, speech_channels, hidden_size, num_classes, roi_mask_kwargs):
        super(CDA, self).__init__()
        self.hidden_size = hidden_size
        self.n_brain_regions = n_brain_regions
        self.n_speech_features = n_speech_features
        self.speech_channels = speech_channels
        self.num_classes = num_classes
        # Encoders and projectors
        self.brain_encoder = BrainEncoder(num_class=num_classes, hidden=hidden_size)
        # Class-wise alignment
        self.align_model = ClassWiseDistributionAlignment()
        # ---------------- Loss Functions ----------------
        self.alignment_cos_sim = nn.CosineSimilarity(dim=1)
        self.cent_loss = CenterLoss(num_classes, hidden_size)


    def abs_cos_sim(self, x, y):
        return (self.alignment_cos_sim(x, y).abs()).sum()

    def forward(self, x, a_shared, a_distinct, label):

        b_shared, b_distinct = self.brain_encoder(x)
        # -----------------------------------------------------------
        loss_orth_b = self.abs_cos_sim(b_shared, b_distinct)
        loss_orth_a = self.abs_cos_sim(a_shared, a_distinct)
        loss_entropy_b = entropy_loss(b_shared, b_distinct)
        loss_entropy_a = entropy_loss(a_shared, a_distinct)
        # -----------------------------------------------------------
        b_shared_reshaped = b_shared.unsqueeze(1)
        a_shared_reshaped = a_shared.unsqueeze(1)
        feat_a_shared, feat_b_shared = self.align_model(b_shared_reshaped, a_shared_reshaped)
        feat_a_shared = feat_a_shared.squeeze()
        feat_b_shared = feat_b_shared.squeeze()
        # -----------------------------------------------------------
        loss1 = self.cent_loss(feat_b_shared, label)
        loss2 = self.cent_loss(feat_a_shared, label)
        loss_cent = loss1 + loss2
        # ----------------- Logit Pooling --------
        h1 = feat_a_shared
        h2 = feat_b_shared
        if len(h1.shape) == 1:
            h1 = h1.unsqueeze(0)
        if len(h2.shape) == 1:
            h2 = h2.unsqueeze(0)
        term1 = torch.stack([h1 + h2, h1 + h2, h1, h2], dim=2)
        term2 = torch.stack([torch.zeros_like(h1), torch.zeros_like(h1), h1, h2], dim=2)
        feat_avg_shared = torch.logsumexp(term1, dim=2) - torch.logsumexp(term2, dim=2)


        if len(feat_avg_shared.shape) == 1:
            feat_avg_shared = feat_avg_shared.unsqueeze(0)

        loss_dist = (loss_orth_a + loss_entropy_a + loss_orth_b + loss_entropy_b)

        return a_distinct, a_shared, b_distinct, b_shared, feat_avg_shared, loss_cent, loss_dist