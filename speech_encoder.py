import torch
import torch.nn as nn
import torch.nn.functional as F
from ResNet import ResBlock  # 假设ResBlock从该路径导入


class AudioInfoCollect(nn.Module):
    def __init__(self, output_channels=64, hidden_channels=64, skip_channels=64,
                 n_layers=5, n_blocks=1, dilation=2, kernel_size=3,
                 filter_num=64, frame_num=640):
        super().__init__()
        self.output_channels = output_channels
        self.hidden_channels = hidden_channels
        self.skip_channels = skip_channels
        self.n_layers = n_layers
        self.n_blocks = n_blocks
        self.dilation = dilation
        self.filter_num = filter_num
        self.frame_num = frame_num
        self.dilations = [self.dilation ** i for i in range(self.n_layers)] * self.n_blocks
        self.kernel_size = kernel_size
        self.relu = nn.LeakyReLU(0.2)

        self.conv = nn.ModuleList()
        self.skip = nn.ModuleList()

        hidden_channels = self.hidden_channels

        for idx, d in enumerate(self.dilations):
            skip_tmp = nn.Sequential(
                nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=1, bias=False),
            )
            self.skip.append(skip_tmp)
            conv_tmp = nn.Sequential(
                nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels,
                          kernel_size=self.kernel_size, bias=False, dilation=d,
                          padding=d * (self.kernel_size - 1) // 2, groups=hidden_channels),
            )
            self.conv.append(conv_tmp)
            if (idx + 1) % self.n_layers == 0:
                hidden_channels = self.hidden_channels // (2 ** ((idx + 1) // self.n_layers))

    def forward(self, inputs):
        output = inputs
        skip_connections = []

        for idx, (dilation, conv, skip) in enumerate(zip(self.dilations, self.conv, self.skip)):
            shortcut = output
            output = conv(output)
            output = self.relu(output)
            skip_outputs = skip(output)
            skip_connections.append(skip_outputs)
            output = output + shortcut[:, :, -output.size(2):]

            if (idx + 1) % self.n_layers == 0 and idx < len(self.dilations) - 1:
                # 合并 skip_connections 并继续，但不下采样
                sum_output = sum([s[:, :, -output.size(2):] for s in skip_connections])
                output = output + sum_output  # 简单相加
                skip_connections = []

        return output


class SpeechEncoder(nn.Module):
    def __init__(self, speech_channels: int, hidden_size: int):
        super().__init__()
        self.audio_info = AudioInfoCollect(output_channels=speech_channels)
        # 两个ResBlock，将通道数从 speech_channels 转换到 hidden_size
        self.res_blocks = nn.Sequential(
            ResBlock(speech_channels, hidden_size),
            ResBlock(hidden_size, hidden_size)
        )
        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.audio_info(x)                # (batch, speech_channels, T')
        x = self.res_blocks(x)                 # (batch, hidden_size, T')
        x = self.gap(x).squeeze(-1)            # (batch, hidden_size)
        return x