"""
GraN-DAG PyTorch Module Version
Can be directly embedded into your neural network

Usage Example:
    self.grandag = GranDAGModule(num_vars=10, num_layers=2, hid_dim=16)
    L_ca, recon_loss, h_val, adj = self.grandag(combined)
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import expm


class TrExpScipy(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        with torch.no_grad():
            expm_input = expm(input.detach().cpu().numpy())
            expm_input = torch.as_tensor(expm_input, dtype=input.dtype, device=input.device)
            ctx.save_for_backward(expm_input)
            return torch.trace(expm_input)

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():
            expm_input, = ctx.saved_tensors
            return expm_input.t() * grad_output


class GranDAGModule(nn.Module):
    """
    GraN-DAG as PyTorch Module
    Can be directly embedded into your model for end-to-end training
    """
    def __init__(self, num_vars, num_layers=2, hid_dim=10, nonlin="leaky-relu", lambda_init=0.0, mu=0.001, norm_prod='paths', square_prod=False):
        """
        Args:
            num_vars: Number of variables (input feature dimension)
            num_layers: Number of hidden layers
            hid_dim: Hidden layer dimension
            nonlin: Activation function type ('leaky-relu' or 'sigmoid')
            lambda_init: Lagrange multiplier initial value
            mu: Penalty coefficient
            norm_prod: Normalization method for weighted adjacency matrix ('paths' or 'none')
            square_prod: Whether to square the weights
        """
        super(GranDAGModule, self).__init__()

        self.num_vars = num_vars
        self.num_layers = num_layers
        self.hid_dim = hid_dim
        self.nonlin = nonlin
        self.norm_prod = norm_prod
        self.square_prod = square_prod

        # Lagrangian parameters (learnable or fixed)
        self.register_buffer('lambda_val', torch.tensor(lambda_init))
        self.register_buffer('mu', torch.tensor(mu))

        # Initialize adjacency matrix (allow all edges except self-loops)
        self.register_buffer('adjacency', torch.ones((num_vars, num_vars)) - torch.eye(num_vars))

        # Initialize network weights and biases
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        for i in range(num_layers + 1):
            in_dim = hid_dim if i > 0 else num_vars
            out_dim = hid_dim if i < num_layers else 2  # Output mean and log_std

            self.weights.append(nn.Parameter(torch.zeros(num_vars, out_dim, in_dim)))
            self.biases.append(nn.Parameter(torch.zeros(num_vars, out_dim)))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters"""
        with torch.no_grad():
            for node in range(self.num_vars):
                for w in self.weights:
                    nn.init.xavier_uniform_(w[node], gain=nn.init.calculate_gain('leaky_relu'))
                for b in self.biases:
                    b[node].zero_()

    def forward_mlp(self, x):
        """
        MLP forward propagation
        Args:
            x: (batch_size, num_vars) input data
        Returns:
            means: (batch_size, num_vars) conditional means
            log_stds: (batch_size, num_vars) conditional log standard deviations
        """
        # Through multi-layer MLP
        for k in range(self.num_layers + 1):
            if k == 0:
                # First layer: apply adjacency matrix
                adj = self.adjacency.unsqueeze(0)
                x = torch.einsum("tij,ljt,bj->bti", self.weights[k], adj, x) + self.biases[k]
            else:
                x = torch.einsum("tij,btj->bti", self.weights[k], x) + self.biases[k]

            # Apply activation function (except for last layer)
            if k != self.num_layers:
                x = F.leaky_relu(x) if self.nonlin == "leaky-relu" else torch.sigmoid(x)

        # x shape: (batch_size, num_vars, 2)
        means = x[:, :, 0]
        log_stds = x[:, :, 1]

        return means, log_stds

    def compute_log_likelihood(self, x):
        """
        Compute log likelihood
        Args:
            x: (batch_size, num_vars)
        Returns:
            log_likelihood: (batch_size,) log likelihood for each sample
        """
        means, log_stds = self.forward_mlp(x)
        stds = torch.exp(log_stds)

        # Compute Gaussian log likelihood
        log_probs = -0.5 * torch.log(torch.tensor(2 * 3.14159265359)) - log_stds - 0.5 * ((x - means) / stds) ** 2

        return torch.sum(log_probs, dim=1)  # Sum over all variables

    def get_weighted_adjacency(self):
        """Get weighted adjacency matrix (based on absolute values of weights)"""
        w_adj = torch.eye(self.num_vars, device=self.weights[0].device)

        for i, w in enumerate(self.weights):
            if self.square_prod:
                w_abs = w ** 2
            else:
                w_abs = torch.abs(w)

            if i == 0:
                w_adj = torch.einsum("tij,ljt,jk->tik", w_abs, self.adjacency.unsqueeze(0), w_adj)
            else:
                w_adj = torch.einsum("tij,tjk->tik", w_abs, w_adj)

        # Sum over output dimensions
        w_adj = torch.sum(w_adj, dim=1)

        if self.norm_prod == 'paths':
            # Path normalization
            prod_norm = torch.eye(self.num_vars, device=w_adj.device)
            for i, w in enumerate(self.weights):
                if i == 0:
                    tmp = 1. - torch.eye(self.num_vars, device=w_adj.device).unsqueeze(0)
                    prod_norm = torch.einsum("tij,ljt,jk->tik", torch.ones_like(w).detach(), tmp, prod_norm)
                else:
                    prod_norm = torch.einsum("tij,tjk->tik", torch.ones_like(w).detach(), prod_norm)
            prod_norm = torch.sum(prod_norm, dim=1)
            denominator = prod_norm + torch.eye(self.num_vars, device=w_adj.device)
            w_adj = w_adj / denominator

        return w_adj.t()

    def compute_dag_constraint(self, w_adj):
        """Compute DAG constraint h(W) = tr(e^(W ∘ W)) - d"""
        h = TrExpScipy.apply(w_adj) - self.num_vars
        return h

    def forward(self, x, return_adj=True, compute_loss=True):
        """
        Forward propagation - main interface

        Args:
            x: (batch_size, num_vars) input features
            return_adj: whether to return adjacency matrix
            compute_loss: whether to compute loss

        Returns:
            L_ca: causal loss (augmented Lagrangian)
            recon_loss: reconstruction loss (negative log likelihood)
            h_val: DAG constraint value
            adj: weighted adjacency matrix (if return_adj=True)
        """
        # Compute negative log likelihood
        log_likelihood = self.compute_log_likelihood(x)
        recon_loss = -torch.mean(log_likelihood)

        # Get weighted adjacency matrix
        w_adj = self.get_weighted_adjacency()

        # Compute DAG constraint
        h_val = self.compute_dag_constraint(w_adj)

        # Compute augmented Lagrangian loss
        if compute_loss:
            L_ca = recon_loss + 0.5 * self.mu * h_val * h_val + self.lambda_val * h_val
        else:
            L_ca = None

        if return_adj:
            return L_ca, recon_loss, h_val, w_adj
        else:
            return L_ca, recon_loss, h_val

    def update_lagrangian_params(self, h_val, h_tol=1e-8, mu_factor=0.1):
        """
        Update Lagrange multiplier and penalty coefficient
        Recommended to call periodically during training (e.g., every epoch)

        Args:
            h_val: current DAG constraint value
            h_tol: tolerance
            mu_factor: penalty coefficient growth factor
        """
        with torch.no_grad():
            if h_val > 0.25 * h_tol:
                self.mu *= mu_factor

            self.lambda_val += self.mu * h_val

    def get_binary_adjacency(self, threshold=0.3):
        """
        Get binary adjacency matrix

        Args:
            threshold: threshold value

        Returns:
            binary_adj: (num_vars, num_vars) binary adjacency matrix
        """
        with torch.no_grad():
            w_adj = self.get_weighted_adjacency()
            binary_adj = (w_adj > threshold).float()
        return binary_adj

    def prune_edges(self, threshold=0.1):
        """
        Prune edges based on threshold (update adjacency matrix)

        Args:
            threshold: pruning threshold
        """
        with torch.no_grad():
            w_adj = self.get_weighted_adjacency()
            self.adjacency *= (w_adj.t() > threshold).float()

    def apply_fixed_mask(self, adjacency_matrix, n_brain, n_speech):
        n_total = adjacency_matrix.shape[0]
        assert n_total == n_brain + n_speech
        with torch.no_grad():
            mask = torch.zeros((n_total, n_total)).cuda()
            for i in range(n_brain):
                for j in range(n_brain, n_total):
                    mask[i, j] = 1.0
        return adjacency_matrix * mask


# ============================================================================
# Usage Examples
# ============================================================================

class YourModel(nn.Module):
    """Example: How to use GranDAG in your model"""

    def __init__(self, input_dim, output_dim, num_causal_vars):
        super(YourModel, self).__init__()

        # Your other network layers
        self.encoder = nn.Linear(input_dim, num_causal_vars)
        self.decoder = nn.Linear(num_causal_vars, output_dim)

        # GranDAG module
        self.grandag = GranDAGModule(
            num_vars=num_causal_vars,
            num_layers=2,
            hid_dim=16,
            mu=0.001
        )

    def forward(self, x):
        # Extract causal features
        causal_features = self.encoder(x)  # (batch, num_causal_vars)

        # Use GranDAG
        L_ca, recon_loss, h_val, adj = self.grandag(causal_features)

        # Continue with your other operations
        output = self.decoder(causal_features)

        return output, L_ca, recon_loss, h_val, adj

    def compute_total_loss(self, x, y):
        """Compute total loss"""
        output, L_ca, recon_loss, h_val, adj = self.forward(x)

        # Your main task loss
        task_loss = F.mse_loss(output, y)

        # Total loss = main task loss + causal loss
        total_loss = task_loss + 0.1 * L_ca  # 0.1 is weight coefficient, adjustable

        return total_loss, {
            'task_loss': task_loss.item(),
            'causal_loss': L_ca.item(),
            'recon_loss': recon_loss.item(),
            'h_constraint': h_val.item()
        }


if __name__ == "__main__":
    print("="*70)
    print("GranDAG Module Test")
    print("="*70)

    # Test 1: Use GranDAG module independently
    print("\nTest 1: Use GranDAG module independently")
    print("-"*70)

    batch_size = 32
    num_vars = 5

    # Create GranDAG module
    grandag = GranDAGModule(num_vars=num_vars, num_layers=2, hid_dim=10)

    # Create test data
    x = torch.randn(batch_size, num_vars)

    # Forward propagation
    L_ca, recon_loss, h_val, adj = grandag(x)

    print(f"Input shape: {x.shape}")
    print(f"Causal loss L_ca: {L_ca.item():.4f}")
    print(f"Reconstruction loss recon_loss: {recon_loss.item():.4f}")
    print(f"DAG constraint h_val: {h_val.item():.6f}")
    print(f"Adjacency matrix shape: {adj.shape}")
    print(f"\nWeighted adjacency matrix:\n{adj.detach().cpu().numpy()}")

    # Test 2: Use in custom model
    print("\n" + "="*70)
    print("Test 2: Use in custom model")
    print("-"*70)

    model = YourModel(input_dim=20, output_dim=10, num_causal_vars=5)

    x_input = torch.randn(batch_size, 20)
    y_target = torch.randn(batch_size, 10)

    total_loss, metrics = model.compute_total_loss(x_input, y_target)

    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Metrics: {metrics}")

    # Test 3: Simulate training process
    print("\n" + "="*70)
    print("Test 3: Simulate training process")
    print("-"*70)

    optimizer = torch.optim.Adam(grandag.parameters(), lr=0.01)

    for epoch in range(5):
        x = torch.randn(batch_size, num_vars)

        L_ca, recon_loss, h_val, adj = grandag(x)

        optimizer.zero_grad()
        L_ca.backward()
        optimizer.step()

        # Update Lagrangian parameters every epoch
        grandag.update_lagrangian_params(h_val.item())

        print(f"Epoch {epoch+1}: L_ca={L_ca.item():.4f}, h={h_val.item():.6f}, "
              f"mu={grandag.mu.item():.6f}")

    print("\n" + "="*70)
    print("Test completed!")
    print("="*70)