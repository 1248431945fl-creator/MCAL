import torch
import torch.nn as nn

class E2E(nn.Module):
    def __init__(self, in_channel, out_channel, input_shape, **kwargs):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.d = input_shape[0]  # Number of brain regions N
        self.conv1xd = nn.Conv2d(in_channel, out_channel, (self.d, 1))
        self.convdx1 = nn.Conv2d(in_channel, out_channel, (1, self.d))

    def forward(self, A):
        # A shape: (batch, 1, N, N) or (batch, in_channel, N, N)
        A = A.view(-1, self.in_channel, self.d, self.d)
        a = self.conv1xd(A)          # (batch, out_channel, 1, N)
        b = self.convdx1(A)          # (batch, out_channel, N, 1)
        concat1 = torch.cat([a] * self.d, 2)   # Replicate along row dimension
        concat2 = torch.cat([b] * self.d, 3)   # Replicate along column dimension
        return concat1 + concat2


class BrainEncoder(nn.Module):
    def __init__(self, dropout=0.25, num_class=2, hidden=64):
        super().__init__()

        # Shared shallow feature extraction
        self.shared_e2e = nn.Sequential(
            E2E(1, 16, (128, 128)),
            nn.Dropout2d(dropout),
            nn.LeakyReLU(0.1),
        )
        self.pool = nn.AdaptiveAvgPool2d((64, 64))

        # Common branch deep features
        self.common_e2e = E2E(16, 16, (64, 64))
        self.common_e2n2g = nn.Sequential(
            nn.Conv2d(16, 32, (1, 64)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, (64, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
        )

        # Specific branch deep features
        self.specific_e2e = E2E(16, 16, (64, 64))
        self.specific_e2n2g = nn.Sequential(
            nn.Conv2d(16, 32, (1, 64)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, (64, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Individual classification heads
        self.commonLinear = self._make_head(64, hidden, dropout)
        self.specificLinear = self._make_head(64, hidden, dropout)

        self._initialize_weights()

    def _make_head(self, in_features, hidden, dropout):
        """Helper function to create classification head"""
        return nn.Sequential(
            nn.Linear(in_features, hidden, bias=True),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden, bias=True)
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Shared shallow features
        x = self.shared_e2e(x)
        x = self.pool(x)

        # Branch processing
        common_x = self.common_e2e(x)
        common_x = self.common_e2n2g(common_x)
        common_x = self.global_pool(common_x).view(common_x.size(0), -1)

        specific_x = self.specific_e2e(x)
        specific_x = self.specific_e2n2g(specific_x)
        specific_x = self.global_pool(specific_x).view(specific_x.size(0), -1)

        common = self.commonLinear(common_x)
        specific = self.specificLinear(specific_x)

        return common, specific