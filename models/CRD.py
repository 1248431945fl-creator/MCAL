import matplotlib
matplotlib.use('Agg')
import networkx as nx
from models.GradNDAG import GranDAGModule
import torch
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CRD(nn.Module):
    def __init__(self,
                 n_brain_regions, n_speech_features, speech_channels, roi_sparsity,
                 smooth_lambda, lambda_p, damping_factor, grandag_hidden,
                 grandag_layers, grandag_lambda_a, grandag_rho_init, grandag_rho_factor, hidden_size):
        super().__init__()
        self.n_brain_regions = n_brain_regions
        self.n_speech_features = n_speech_features
        self.speech_channels = speech_channels

        # Causal discovery
        self.grandag = GranDAGModule(num_vars=n_brain_regions + n_speech_features,          # Required: input feature dimension
                                     num_layers=grandag_layers,          # Number of MLP layers, 2-3 usually sufficient
                                     hid_dim=grandag_hidden,            # Hidden layer dimension, 10-20 usually appropriate
                                     nonlin="leaky-relu",   # Activation function
                                     lambda_init=grandag_lambda_a,       # Lagrange multiplier initial value
                                     mu=0.001,              # Penalty coefficient, key parameter!
                                     norm_prod='paths',     # Adjacency matrix normalization method
                                     square_prod=False)
        self.roi_sparsity = roi_sparsity
        self.smooth_lambda = smooth_lambda
        self.lambda_p = lambda_p
        self.damping_factor = damping_factor
        # Temporary storage
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

    def forward(self, fmri, speech_compressed):
        N = self.n_brain_regions
        S = self.n_speech_features

        # Brain compression
        diag_mask = ~torch.eye(N, dtype=torch.bool, device=device)
        node_strength = (fmri * diag_mask).sum(dim=2) / (N - 1)

        # Causal discovery
        combined = torch.cat([node_strength, speech_compressed], dim=1)
        L_ca, recon_loss, h_val, adj = self.grandag(combined)
        masked_adj = self.grandag.apply_fixed_mask(adj, N, S)
        # Direct, indirect effects and importance matrix M
        node_imp, direct, indirect = self._compute_node_importance(masked_adj)
        M = self._create_causal_mask(node_imp)
        M = torch.sigmoid(M)
        # Apply mask
        masked_pcc = fmri * M.unsqueeze(0)
        # Loss calculation
        L_sparse = self._compute_sparsity_loss(M)
        L_smooth = self._compute_smoothness_loss(M)
        total_loss = L_sparse + L_smooth + L_ca
        if self.save_intermediate_results:
            self.importance_matrix_history.append(M.detach().cpu().numpy())
            self.node_importance_history.append(node_imp.detach().cpu().numpy())
            self.adjacency_matrix_history.append(masked_adj.detach().cpu().numpy())

        return masked_pcc, total_loss, node_imp, M
    # -------------------------------------------------------------------
    def enable_save_intermediate(self):
        self.save_intermediate_results = True
    def disable_save_intermediate(self):
        self.save_intermediate_results = False
    # -------------------------------------------------------------------