
import torch
import torch.nn as nn


# Cell
class ResBlock(nn.Module):
    def __init__(self, ni, nf, kss=[3, 3, 3]):
        super().__init__()
        # 卷积块1：Conv + BN + ReLU
        self.convblock1 = nn.Sequential(
            nn.Conv1d(ni, nf, kernel_size=kss[0], padding=kss[0]//2),
            nn.BatchNorm1d(nf),
            nn.ReLU()
        )
        # 卷积块2：Conv + BN + ReLU
        self.convblock2 = nn.Sequential(
            nn.Conv1d(nf, nf, kernel_size=kss[1], padding=kss[1]//2),
            nn.BatchNorm1d(nf),
            nn.ReLU()
        )
        # 卷积块3：Conv + BN（无激活）
        self.convblock3 = nn.Sequential(
            nn.Conv1d(nf, nf, kernel_size=kss[2], padding=kss[2]//2),
            nn.BatchNorm1d(nf)
        )
        # shortcut：1x1 Conv + BN
        if ni == nf:
            self.shortcut = nn.BatchNorm1d(ni)   # 原代码中 BN1d(ni) 仅 BN，无卷积
        else:
            self.shortcut = nn.Sequential(
                nn.Conv1d(ni, nf, kernel_size=1),
                nn.BatchNorm1d(nf)
            )
        self.act = nn.LeakyReLU(0.2)

        # 初始化（保持原风格）
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res = x
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = x + self.shortcut(res)   # 直接加法，无需 Add 类
        x = self.act(x)
        return x


class ResNet(torch.nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        # 33 47
        nf = 47  # 根据merge后的大小设置通道
        self.resblock1 = ResBlock(c_in, nf)
        self.resblock2 = ResBlock(nf, nf * 2)
        self.resblock3 = ResBlock(nf * 2, nf * 2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(nf * 2, c_out)
        self.dropout = nn.Dropout(0.25)  # 防止over-fitting

    def forward(self, x):
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.gap(x).squeeze(-1)
        x = self.fc(x)
        x = self.dropout(x)
        return x

