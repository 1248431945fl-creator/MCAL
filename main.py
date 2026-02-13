import matplotlib
matplotlib.use('Agg')
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import KFold
from sklearn import metrics
import random
import torch.optim as optim
import pandas as pd
import networkx as nx

from brain_encoder import BrainEncoder
from speech_encoder import SpeechEncoder
from RegNet import RegNet_lite

# 导入配置
from config import config

device = torch.device(config["training"]["device"])

class GraNDAG(nn.Module):
    def __init__(self, n_nodes, hidden_size, num_layers,
                 lambda_a, rho_init, rho_factor):
        super(GraNDAG, self).__init__()
        self.n_nodes = n_nodes
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lambda_a = lambda_a
        self.rho = rho_init
        self.rho_factor = rho_factor

        self.networks = nn.ModuleList()
        for _ in range(n_nodes):
            layers = []
            in_dim = n_nodes - 1
            for i in range(num_layers):
                out_dim = hidden_size if i < num_layers - 1 else 1
                layers.append(nn.Linear(in_dim, out_dim))
                if i < num_layers - 1:
                    layers.append(nn.ReLU())
                in_dim = out_dim
            self.networks.append(nn.Sequential(*layers))

    def forward(self, X):
        reconstructions = torch.zeros_like(X)
        for j in range(self.n_nodes):
            indices = [i for i in range(self.n_nodes) if i != j]
            inputs = X[:, indices]
            outputs = self.networks[j](inputs)
            reconstructions[:, j] = outputs.squeeze()
        return reconstructions

    def get_adjacency_matrix(self):
        A = torch.zeros((self.n_nodes, self.n_nodes), device=device)
        for j in range(self.n_nodes):
            network_j = self.networks[j]
            linear_layers = [layer for layer in network_j if isinstance(layer, nn.Linear)]
            if not linear_layers:
                continue
            C = torch.abs(linear_layers[-1].weight)
            for l in range(len(linear_layers) - 2, -1, -1):
                W_abs = torch.abs(linear_layers[l].weight)
                C = torch.matmul(C, W_abs)
            idx = 0
            for i in range(self.n_nodes):
                if i != j:
                    A[i, j] = C[0, idx]
                    idx += 1
        return A

    def apply_fixed_mask(self, adjacency_matrix, n_brain, n_speech):
        n_total = adjacency_matrix.shape[0]
        assert n_total == n_brain + n_speech
        with torch.no_grad():
            mask = torch.zeros((n_total, n_total), device=device)
            for i in range(n_brain):
                for j in range(n_brain, n_total):
                    mask[i, j] = 1.0
        return adjacency_matrix * mask

    def compute_augmented_lagrangian(self, X):
        reconstructions = self.forward(X)
        recon_loss = F.mse_loss(reconstructions, X)
        A = self.get_adjacency_matrix()
        try:
            expA = torch.matrix_exp(A)
            h = torch.trace(expA) - self.n_nodes
        except:
            h = torch.tensor(10.0, device=device)
        L_ca = recon_loss + self.lambda_a * h + self.rho * h * h / 2.0
        return L_ca, recon_loss.item(), h.item(), A

    def update_rho(self, h):
        if h > 0.25 * self.lambda_a:
            self.rho *= self.rho_factor


class LearnableROIMask(nn.Module):
    def __init__(self,
                 n_brain_regions,
                 n_speech_features,
                 speech_channels,
                 roi_sparsity,
                 smooth_lambda,
                 lambda_p,
                 damping_factor,
                 grandag_hidden,
                 grandag_layers,
                 grandag_lambda_a,
                 grandag_rho_init,
                 grandag_rho_factor):
        super().__init__()
        self.n_brain_regions = n_brain_regions
        self.n_speech_features = n_speech_features
        self.speech_channels = speech_channels

        self.node_mask = nn.Parameter(torch.randn(n_brain_regions))

        self.grandag = GraNDAG(
            n_nodes=n_brain_regions + n_speech_features,
            hidden_size=grandag_hidden,
            num_layers=grandag_layers,
            lambda_a=grandag_lambda_a,
            rho_init=grandag_rho_init,
            rho_factor=grandag_rho_factor
        )

        self.roi_sparsity = nn.Parameter(torch.tensor(roi_sparsity, dtype=torch.float32))
        self.smooth_lambda = nn.Parameter(torch.tensor(smooth_lambda, dtype=torch.float32))
        self.lambda_p = nn.Parameter(torch.tensor(lambda_p, dtype=torch.float32))
        self.damping_factor = damping_factor

        self.speech_projection = nn.Sequential(
            nn.Linear(speech_channels, 128),
            nn.ReLU(),
            nn.Linear(128, n_speech_features)
        )

        self.save_intermediate_results = False
        self.importance_matrix_history = []
        self.node_importance_history = []
        self.adjacency_matrix_history = []

    def _compute_sparsity_loss(self, M):
        return torch.sum(torch.abs(M)) * self.roi_sparsity

    def _compute_smoothness_loss(self, M):
        N = self.n_brain_regions
        loss = 0
        for i in range(N - 1):
            loss += torch.sum((M[i, :] - M[i + 1, :]) ** 2)
        for j in range(N - 1):
            loss += torch.sum((M[:, j] - M[:, j + 1]) ** 2)
        return loss * self.smooth_lambda

    def _compute_direct_effect(self, adjacency_matrix):
        return adjacency_matrix[:self.n_brain_regions, self.n_brain_regions:]

    def _compute_indirect_effect(self, adjacency_matrix):
        N = self.n_brain_regions
        A_brain = adjacency_matrix[:N, :N].detach().cpu().numpy()
        try:
            G = nx.DiGraph()
            for i in range(N):
                G.add_node(i)
            for i in range(N):
                for j in range(N):
                    w = A_brain[i, j]
                    if abs(w) > 1e-8:
                        G.add_edge(i, j, weight=float(abs(w)))
            pr = nx.pagerank(G, alpha=self.damping_factor, max_iter=200, tol=1e-6)
            eta = torch.zeros(N, device=device)
            for node, score in pr.items():
                eta[node] = score
            if eta.sum() > 0:
                eta = eta / eta.sum()
        except Exception:
            eta = torch.ones(N, device=device) / N
        return eta

    def _compute_node_importance(self, adjacency_matrix):
        beta = self._compute_direct_effect(adjacency_matrix)
        eta = self._compute_indirect_effect(adjacency_matrix)
        direct = torch.sum(torch.abs(beta), dim=1)
        importance = direct + self.lambda_p * eta
        if importance.sum() > 0:
            importance = importance / importance.sum()
        return importance, direct, eta

    def _create_causal_mask(self, node_importance):
        p = node_importance.unsqueeze(1)
        outer = torch.matmul(p, p.t())
        return torch.sigmoid(outer)

    def forward(self, fmri, speech):
        N = self.n_brain_regions
        S = self.n_speech_features

        diag_mask = ~torch.eye(N, dtype=torch.bool, device=device)
        node_strength = (fmri * diag_mask).sum(dim=2) / (N - 1)

        if len(speech.shape) == 3:
            speech_pooled = F.adaptive_avg_pool1d(speech, 1).squeeze(-1)
        else:
            speech_pooled = speech
        speech_compressed = self.speech_projection(speech_pooled)

        combined = torch.cat([node_strength, speech_compressed], dim=1)
        mean_feat = torch.mean(combined, dim=0, keepdim=True)
        L_ca, recon_loss, h_val, adj = self.grandag.compute_augmented_lagrangian(mean_feat)

        masked_adj = self.grandag.apply_fixed_mask(adj, N, S)
        self.grandag.update_rho(h_val)

        node_imp, direct, indirect = self._compute_node_importance(masked_adj)
        M = self._create_causal_mask(node_imp)

        base = torch.sigmoid(self.node_mask)
        combined_mask = base.unsqueeze(0) * base.unsqueeze(1) * M

        masked_pcc = fmri * combined_mask.unsqueeze(0)

        L_sparse = self._compute_sparsity_loss(combined_mask)
        L_smooth = self._compute_smoothness_loss(combined_mask)
        total_loss = L_sparse + L_smooth + L_ca

        if self.save_intermediate_results:
            self.importance_matrix_history.append(combined_mask.detach().cpu().numpy())
            self.node_importance_history.append(node_imp.detach().cpu().numpy())
            self.adjacency_matrix_history.append(masked_adj.detach().cpu().numpy())

        return masked_pcc, total_loss, node_imp, combined_mask

    def enable_save_intermediate(self):
        self.save_intermediate_results = True

    def disable_save_intermediate(self):
        self.save_intermediate_results = False

    def clear_history(self):
        self.importance_matrix_history = []
        self.node_importance_history = []
        self.adjacency_matrix_history = []


class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction='none', log_target=True)

    def forward(self, p: torch.Tensor, q: torch.Tensor):
        p, q = F.softmax(p.view(-1, p.size(-1)), dim=1), F.softmax(q.view(-1, q.size(-1)), dim=1)
        m = 0.5 * (p + q)
        m_log = m.log()
        p_log = p.log()
        q_log = q.log()
        kl_pm = self.kl(m_log, p_log)
        kl_qm = self.kl(m_log, q_log)
        jsd = 0.5 * (kl_pm + kl_qm).sum()
        return jsd


class CrossModalContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, roi_features, speech_features, labels):
        roi_features = F.normalize(roi_features, dim=1)
        speech_features = F.normalize(speech_features, dim=1)
        sim_matrix = torch.mm(roi_features, speech_features.T) / self.temperature
        pos_mask = (labels.view(-1, 1) == labels.view(1, -1)).float()
        pos_mask.fill_diagonal_(1)
        logits = torch.exp(sim_matrix)
        pos_sim = torch.sum(logits * pos_mask, dim=1)
        neg_sim = torch.sum(logits, dim=1) - pos_sim
        loss = -torch.log(pos_sim / (pos_sim + neg_sim)).mean()
        return loss


def entropy_loss(f_com, f_spec, eps=1e-6, delta=1e-8):
    def logdet_cov(x):
        # x: (B, D)
        B, D = x.shape
        x_center = x - x.mean(dim=0, keepdim=True)
        cov = (x_center.T @ x_center) / (B - 1)
        # 添加对角扰动保证可逆
        cov_reg = cov + eps * torch.eye(D, device=x.device)
        return torch.logdet(cov_reg)  # 标量

    logdet_com = logdet_cov(f_com)
    logdet_spec = logdet_cov(f_spec)
    diff = torch.abs(logdet_com - logdet_spec)
    loss = 1.0 / (diff + delta)
    return loss


class ClassConsistencyLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(ClassConsistencyLoss, self).__init__()
        self.temperature = temperature

    def forward(self, logits_weak, logits_strong):
        probs_weak = F.softmax(logits_weak / self.temperature, dim=1)
        log_probs_strong = F.log_softmax(logits_strong / self.temperature, dim=1)
        loss = F.kl_div(log_probs_strong, probs_weak, reduction='batchmean')
        return loss


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))

    def forward(self, features, labels):
        if len(features.shape) == 1:
            features = features.unsqueeze(0)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, device=self.device)
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)

        batch_size = features.size(0)
        distmat = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes)
        distmat = distmat + torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(features, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).to(self.device)
        labels_expanded = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels_expanded.eq(classes.expand(batch_size, self.num_classes))
        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        return loss

class FUSION(nn.Module):
    def __init__(self,
                 n_brain_regions,
                 n_speech_features,
                 speech_channels,
                 hidden_size,
                 num_classes,
                 align,
                 roi_mask_kwargs):
        super(FUSION, self).__init__()
        self.hidden_size = hidden_size
        self.align_model = align
        self.n_brain_regions = n_brain_regions
        self.n_speech_features = n_speech_features
        self.speech_channels = speech_channels
        self.num_classes = num_classes

        self.roi_mask = LearnableROIMask(
            n_brain_regions=n_brain_regions,
            n_speech_features=n_speech_features,
            speech_channels=speech_channels,
            **roi_mask_kwargs
        )

        self.brain_shared = BrainEncoder(n_brain_regions, hidden_size)
        self.brain_distinct = BrainEncoder(n_brain_regions, hidden_size)

        self.audio_shared = SpeechEncoder(speech_channels, hidden_size)
        self.audio_distinct = SpeechEncoder(speech_channels, hidden_size)

        self.linear_b = nn.Linear(n_brain_regions, hidden_size)
        self.linear_compress = nn.Linear(2 * hidden_size, hidden_size)

        self.aux_brain_spec = nn.Linear(hidden_size, 2)
        self.aux_audio_spec = nn.Linear(hidden_size, 2)

        self.alignment_cos_sim = nn.CosineSimilarity(dim=1)
        self.jsd = JSD()
        self.criterion = nn.CrossEntropyLoss()
        self.cent_loss = CenterLoss(2, hidden_size, device)

        self.classifier = nn.Linear(hidden_size, num_classes)

        self.masked_pcc_history = []
        self.importance_matrix_history = []
        self.save_masked_pcc = False

    def clear_history(self):
        self.masked_pcc_history = []
        self.importance_matrix_history = []

    def enable_save_masked_pcc(self):
        self.save_masked_pcc = True

    def disable_save_masked_pcc(self):
        self.save_masked_pcc = False

    def abs_cos_sim(self, x, y):
        return (self.alignment_cos_sim(x, y).abs()).sum()

    def forward(self, x, a, label, save_masked_pcc=False):
        # 应用因果ROI掩码
        x, constraint_loss, ni, importance_matrix = self.roi_mask(x, a)

        if save_masked_pcc or self.save_masked_pcc:
            self.masked_pcc_history.append(x.detach().cpu().numpy())
            self.importance_matrix_history.append(importance_matrix.detach().cpu().numpy())

        b_shared = self.brain_shared(x)
        b_distinct = self.brain_distinct(x)

        a_shared = self.audio_shared(a)
        a_distinct = self.audio_distinct(a)

        loss_sim_b = self.abs_cos_sim(b_shared, b_distinct)
        loss_sim_a = self.abs_cos_sim(a_shared, a_distinct)

        loss_entropy_b = entropy_loss(b_shared, b_distinct)
        loss_entropy_a = entropy_loss(a_shared, a_distinct)

        feat_b_distinct = self.linear_b(b_distinct)
        feat_b_distinct = self.linear_compress(feat_b_distinct)

        feat_a_distinct = a_distinct

        b_shared_reshaped = b_shared.unsqueeze(1)
        a_shared_reshaped = a_shared.unsqueeze(1)
        feat_a_shared, feat_b_shared = self.align_model(b_shared_reshaped, a_shared_reshaped)
        feat_a_shared = feat_a_shared.squeeze()
        feat_b_shared = feat_b_shared.squeeze()

        pred_brain_spec = self.aux_brain_spec(feat_b_distinct)
        pred_audio_spec = self.aux_audio_spec(feat_a_distinct)
        loss_brain_spec = self.criterion(pred_brain_spec, label)
        loss_audio_spec = self.criterion(pred_audio_spec, label)
        aux_loss = loss_brain_spec + loss_audio_spec

        loss1 = self.cent_loss(feat_b_shared, label)
        loss2 = self.cent_loss(feat_a_shared, label)
        loss_align = loss1 + loss2

        h1 = feat_a_shared
        h2 = feat_b_shared
        if len(h1.shape) == 1:
            h1 = h1.unsqueeze(0)
        if len(h2.shape) == 1:
            h2 = h2.unsqueeze(0)
        term1 = torch.stack([h1 + h2, h1 + h2, h1, h2], dim=2)
        term2 = torch.stack([torch.zeros_like(h1), torch.zeros_like(h1), h1, h2], dim=2)
        feat_avg_shared = torch.logsumexp(term1, dim=2) - torch.logsumexp(term2, dim=2)

        jsd = self.jsd(feat_b_shared.sigmoid(), feat_a_shared.sigmoid())

        if len(feat_avg_shared.shape) == 1:
            feat_avg_shared = feat_avg_shared.unsqueeze(0)

        pred_final = self.classifier(feat_avg_shared)

        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, device=device)
        if label.dim() == 0:
            label = label.unsqueeze(0)

        loss_final = self.criterion(pred_final, label)

        return (
            pred_final,
            loss_sim_a + loss_entropy_a,
            loss_sim_b + loss_entropy_b,
            jsd,
            loss_final,
            loss_align,
            constraint_loss,
            aux_loss
        )


class MLA(nn.Module):
    def __init__(self, model, hidden_dim, lr, weight_decay):
        super(MLA, self).__init__()
        self.model = model
        self.total_optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def forward(self, x, a, labels, mode=True, save_masked_pcc=False):
        pred, loss_a, loss_b, loss_shared, loss_final, loss_align, loss_contrast, aux_loss  = self.model(
            x, a, labels, save_masked_pcc
        )
        loss_total = 0.001 * loss_a + 0.1 * loss_b + loss_final + loss_align * 0.001 + loss_contrast + loss_shared + aux_loss

        if mode:
            loss_total.backward(retain_graph=True)
            self.total_optimizer.step()
            self.total_optimizer.zero_grad()

        return pred, loss_a, loss_b, loss_final, loss_total, loss_align, loss_contrast


def setup_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    data_cfg = config["data"]
    train_cfg = config["training"]
    model_cfg = config["model"]
    roi_mask_cfg = model_cfg["roi_mask"]

    fmri_data = np.load(data_cfg["fmri_path"])
    audio_data = np.load(data_cfg["audio_path"]).reshape(-1, data_cfg["speech_channels"], data_cfg["speech_length"])
    labels = np.load(data_cfg["label_path"])

    train_data = fmri_data
    audio_feature = audio_data
    train_label = labels

    print(f"fMRI数据形状: {train_data.shape}")
    print(f"语音数据形状: {audio_feature.shape}")
    print(f"标签形状: {train_label.shape}")

    # K折交叉验证
    kf = KFold(n_splits=train_cfg["n_splits"], random_state=train_cfg["random_seed"], shuffle=True)
    all_results = []
    kfold_index = 0

    for train_idx, test_idx in kf.split(train_data):

        setup_seed(train_cfg["random_seed"])
        print(f"\n{'=' * 60}")
        print(f"开始第 {kfold_index + 1} 折交叉验证")
        print(f"{'=' * 60}")

        X_train, X_test = train_data[train_idx], train_data[test_idx]
        A_train, A_test = audio_feature[train_idx], audio_feature[test_idx]
        Y_train, Y_test = train_label[train_idx], train_label[test_idx]

        align = RegNet_lite()
        model = FUSION(
            n_brain_regions=data_cfg["n_brain_regions"],
            n_speech_features=data_cfg["n_speech_features"],
            speech_channels=data_cfg["speech_channels"],
            hidden_size=model_cfg["hidden_size"],
            num_classes=model_cfg["num_classes"],
            align=align,
            roi_mask_kwargs=roi_mask_cfg
        ).to(device)

        mla = MLA(model, hidden_dim=model_cfg["hidden_size"],
                  lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
        mla.to(device)

        total_params = sum(p.numel() for p in mla.parameters())
        trainable_params = sum(p.numel() for p in mla.parameters() if p.requires_grad)
        print(f"模型参数量 - 总数: {total_params}, 可训练: {trainable_params}")

        for epoch in range(train_cfg["num_epochs"]):
            mla.train()
            indices = np.random.permutation(len(X_train))
            total_batches = int(np.ceil(len(X_train) / train_cfg["batch_size"]))
            batch_losses = []

            for i in range(total_batches):
                start = i * train_cfg["batch_size"]
                end = min((i + 1) * train_cfg["batch_size"], len(X_train))
                batch_idx = indices[start:end]

                brain_batch = torch.tensor(X_train[batch_idx]).float().to(device)
                audio_batch = torch.tensor(A_train[batch_idx]).float().to(device)
                label_batch = torch.tensor(Y_train[batch_idx]).long().to(device)

                pred, loss_a, loss_b, loss_final, loss_total, loss_align, loss_contrast = mla(
                    brain_batch, audio_batch, label_batch
                )
                batch_losses.append(loss_total.item())

            if epoch % 5 == 0:
                mla.eval()
                train_acc = 0
                train_cnt = 0
                for i in range(total_batches):
                    start = i * train_cfg["batch_size"]
                    end = min((i + 1) * train_cfg["batch_size"], len(X_train))
                    batch_idx = indices[start:end]

                    brain_batch = torch.tensor(X_train[batch_idx]).float().to(device)
                    audio_batch = torch.tensor(A_train[batch_idx]).float().to(device)
                    label_batch = torch.tensor(Y_train[batch_idx]).long().to(device)

                    pred, *_ = mla(brain_batch, audio_batch, label_batch, mode=False)
                    _, preds = torch.max(pred, 1)
                    train_acc += metrics.accuracy_score(label_batch.cpu(), preds.cpu())
                    train_cnt += 1
                train_acc /= train_cnt if train_cnt > 0 else 1

                brain_test = torch.tensor(X_test).float().to(device)
                audio_test = torch.tensor(A_test).float().to(device)
                label_test = torch.tensor(Y_test).long().to(device)

                pred, *_ = mla(brain_test, audio_test, label_test, mode=False)
                _, preds = torch.max(pred, 1)
                acc = metrics.accuracy_score(Y_test, preds.cpu())
                precision = metrics.precision_score(Y_test, preds.cpu(), average='weighted', zero_division=0)
                recall = metrics.recall_score(Y_test, preds.cpu(), average='weighted')
                f1 = metrics.f1_score(Y_test, preds.cpu(), average='weighted')

                all_results.append([epoch, acc, precision, recall, f1])

                if epoch == train_cfg["num_epochs"] - 1:
                    model.enable_save_masked_pcc()
                    with torch.no_grad():
                        _, *_ = mla(brain_test, audio_test, label_test, mode=False, save_masked_pcc=True)

                print(f'Epoch {epoch:3d}: Train Acc={train_acc:.4f}, Test Acc={acc:.4f}, F1={f1:.4f}')

        del model, mla
        torch.cuda.empty_cache()
        kfold_index += 1

    if all_results:
        print("\n最终结果汇总：")
        df = pd.DataFrame(all_results, columns=['epoch', 'acc', 'precision', 'recall', 'f1'])
        print(df.groupby('epoch').mean())


if __name__ == "__main__":
    main()