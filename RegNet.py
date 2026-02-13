import numpy as np
import torch
from scipy.io import loadmat
import torch.nn.functional as F
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.autograd import Variable
from sklearn import metrics
from monai.losses.ssim_loss import SSIMLoss
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
class E2E(nn.Module):

    def __init__(self, in_channel, out_channel, input_shape, **kwargs):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.d = input_shape[0]
        self.conv1xd = nn.Conv2d(in_channel, out_channel, (self.d, 1))
        self.convdx1 = nn.Conv2d(in_channel, out_channel, (1, self.d))

    def forward(self, A):
        #         print(A.shape)
        #A = A.view(-1, self.in_channel, 200, 200)#############
        A = A.view(-1, self.in_channel, 90, 90)

        a = self.conv1xd(A)
        b = self.convdx1(A)

        concat1 = torch.cat([a] * self.d, 2)
        concat2 = torch.cat([b] * self.d, 3)

        # A = torch.mean(concat1+concat2, 1)
        # print('e2e', (concat1+concat2).shape)
        return concat1 + concat2

class ClassConsistencyLoss(nn.Module):
    def __init__(self, temperature=1.0):
        """
        类别一致性损失的实现。
        :param temperature: 温度参数，用于控制softmax的锐度。
        """
        super(ClassConsistencyLoss, self).__init__()
        self.temperature = temperature

    def forward(self, logits_weak, logits_strong):
        """
        计算类别一致性损失。
        :param logits_weak: 弱增强（或未增强）的输入的logits。
        :param logits_strong: 强增强的输入的logits。
        :return: 类别一致性损失。
        """
        # 计算softmax概率
        probs_weak = F.softmax(logits_weak / self.temperature, dim=1)
        log_probs_strong = F.log_softmax(logits_strong / self.temperature, dim=1)

        # 计算KL散度
        loss = F.kl_div(log_probs_strong, probs_weak, reduction='batchmean')
        return loss


class AttentionAlignment(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionAlignment, self).__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.layer_norm = nn.LayerNorm(feature_dim)

    def forward(self, f1, f2):
        # Compute attention scores
        query = self.query(f1)
        key = self.key(f2)
        attention_scores = torch.matmul(query, key.permute(0,2,1)) / torch.sqrt(torch.tensor(f1.shape[-1], dtype=torch.float32))
        attention_weights = self.softmax(attention_scores)

        # Apply attention weights
        aligned_f2 = torch.matmul(attention_weights, f2)
        # Residual connection and layer normalization
        residual = aligned_f2 + f2  # 残差连接
        output = self.layer_norm(residual)  # 层归一化
        return output
class CrossAttention(nn.Module):
    def __init__(self, in_channel):
        super(CrossAttention, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1, stride=1),
                                   # nn.InstanceNorm2d(in_channel),
                                   nn.ReLU(),
                                   )
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1, stride=1),
                                   # nn.InstanceNorm2d(in_channel),
                                   nn.ReLU(),
                                   )
    def forward(self, f1, f2):
        f1_hat = f1
        f1 = self.conv1(f1)
        f2 = self.conv2(f2)
        att_map = f1 * f2
        att_shape = att_map.shape
        att_map = torch.reshape(att_map, [att_shape[0], att_shape[1], -1])
        att_map = F.softmax(att_map, dim=2)
        att_map = torch.reshape(att_map, att_shape)
        f1 = f1 * att_map
        f1 = f1 + f1_hat
        return f1


class Decoder(nn.Module):
    def __init__(self, channels):
        super(Decoder, self).__init__()
        self.up1 = nn.Linear(channels[0], channels[1])
        self.up2 = nn.Linear(channels[1], channels[2])
        self.up3 = nn.Linear(channels[2], channels[3])
        self.up4 = nn.Linear(channels[3], channels[4])

class RegNet_lite(nn.Module):
    def __init__(self):
        super(RegNet_lite, self).__init__()
        self.channels = [128, 64, 16, 8, 1]
        # self.channels = [64, 16, 1]
        self.attention_align0 = AttentionAlignment(self.channels[0])
        self.attention_align1 = AttentionAlignment(self.channels[1])
        self.attention_align2 = AttentionAlignment(self.channels[2])
        self.attention_align3 = AttentionAlignment(self.channels[3])
        self.attention_align4 = AttentionAlignment(self.channels[4])
        self.decoder = Decoder(self.channels)

        self.conv_f1_1 = nn.Conv2d(8, 64, kernel_size=1, stride=1, padding=0)
        self.upsample = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=True)
        self.conv_f1_2 = nn.Conv2d(64, 128, kernel_size=1)
        self.conv_f2 = nn.Conv2d(64, 128, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 128))
        self.brain_re = nn.Sequential(
            # E2E(1, 8, (200, 200)),
            E2E(1, 8, (90, 90)),
            nn.LeakyReLU(0.25),
            # E2E(8, 8, (200, 200)),
            E2E(8, 8, (90, 90)),
            nn.LeakyReLU(0.25),

            nn.Conv2d(8, 64, (1, 90)),
            nn.LeakyReLU(0.33),
        )
        self.liner_b = nn.Linear(90,128)
        self.liner_a = nn.Linear(640, 128)

    def forward(self, f1, f2):
        # 调整 f1 的通道数
        # f1 = self.brain_re(f1).squeeze()

        # re channel interpolate
        f1 = f1.unsqueeze(1)
        f2 = f2.unsqueeze(1)
        f1 = F.interpolate(f1, size=(128, 90), mode='bilinear', align_corners=True)
        f2 = F.interpolate(f2, size=(128, 640), mode='bilinear', align_corners=True)
        f1 = f1.squeeze(1)
        f2 = f2.squeeze(1)
        # re channel conv
        # f1 = f1.unsqueeze(2)
        # f2 = f2.unsqueeze(2)
        # f1 = self.conv_f1_2(f1)
        # f2 = self.conv_f2(f2)
        # f1 = f1.squeeze()
        # f2 = f2.squeeze()

        brain_features = self.liner_b(f1)
        audio_features = self.liner_a(f2)

        # Align audio features to brain features using attention
        brain_features = self.attention_align0(brain_features, audio_features)
        audio_features = self.attention_align0(audio_features, brain_features)

        # Decode features to higher resolution
        brain_features = self.decoder.up1(brain_features)
        audio_features = self.decoder.up1(audio_features)

        # Align audio features to brain features using attention
        brain_features = self.attention_align1(brain_features, audio_features)
        audio_features = self.attention_align1(audio_features, brain_features)

        # Decode features to higher resolution
        brain_features = self.decoder.up2(brain_features)
        audio_features = self.decoder.up2(audio_features)

        # Align audio features to brain features using attention
        brain_features = self.attention_align2(brain_features, audio_features)
        audio_features = self.attention_align2(audio_features, brain_features)

        # Decode features to higher resolution
        brain_features = self.decoder.up3(brain_features)
        audio_features = self.decoder.up3(audio_features)
        # Align audio features to brain features using attention
        brain_features = self.attention_align3(brain_features, audio_features)
        audio_features = self.attention_align3(audio_features, brain_features)

        # Decode features to higher resolution
        brain_features = self.decoder.up4(brain_features)
        audio_features = self.decoder.up4(audio_features)

        # Align audio features to brain features using attention
        # brain_features = self.attention_align4(brain_features, audio_features)
        # audio_features = self.attention_align4(audio_features, brain_features)
        return brain_features, audio_features

def global_average_pooling(x):
    return F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
