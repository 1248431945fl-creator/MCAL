import torch
import torch.nn as nn

class E2E(nn.Module):
    def __init__(self, in_channel, out_channel, input_shape, **kwargs):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.d = input_shape[0]  # 脑区数量N
        self.conv1xd = nn.Conv2d(in_channel, out_channel, (self.d, 1))
        self.convdx1 = nn.Conv2d(in_channel, out_channel, (1, self.d))

    def forward(self, A):
        # A shape: (batch, 1, N, N)  or (batch, in_channel, N, N)
        A = A.view(-1, self.in_channel, self.d, self.d)
        a = self.conv1xd(A)          # (batch, out_channel, 1, N)
        b = self.convdx1(A)          # (batch, out_channel, N, 1)
        concat1 = torch.cat([a] * self.d, 2)   # 沿行方向复制
        concat2 = torch.cat([b] * self.d, 3)   # 沿列方向复制
        return concat1 + concat2


class BrainEncoder(nn.Module):
    def __init__(self, n_brain_regions: int, hidden_channels: int = 64):
        super().__init__()

        self.shared = nn.Sequential(
            E2E(1, 8, (n_brain_regions, n_brain_regions)),
            nn.LeakyReLU(0.25),
            E2E(8, 8, (n_brain_regions, n_brain_regions)),
            nn.LeakyReLU(0.25),
            nn.Conv2d(8, hidden_channels, (1, n_brain_regions)),
            nn.LeakyReLU(0.33),
        )

        self.distinct = nn.Sequential(
            E2E(1, 8, (n_brain_regions, n_brain_regions)),
            nn.LeakyReLU(0.25),
            E2E(8, 8, (n_brain_regions, n_brain_regions)),
            nn.LeakyReLU(0.25),
            nn.Conv2d(8, hidden_channels, (1, n_brain_regions)),
            nn.LeakyReLU(0.33),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        shared = self.shared(x).squeeze()      # (batch, hidden_channels)
        distinct = self.distinct(x).squeeze()
        return shared, distinct