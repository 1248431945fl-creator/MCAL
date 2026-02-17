import torch
import torch.nn.functional as F
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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


class Decoder(nn.Module):
    def __init__(self, channels, dropout=0.1):
        super(Decoder, self).__init__()
        self.up1 =  nn.Sequential(
            nn.Linear(channels[0]//4, channels[0]//2, bias=True),
            nn.LayerNorm(channels[0]//2),
            nn.LeakyReLU(0.1),
            nn.Linear(channels[0]//2, channels[0], bias=True)
        )
        self.up2 =  nn.Sequential(
            nn.Linear(channels[1]//4, channels[1]//2, bias=True),
            nn.LayerNorm(channels[1]//2),
            nn.LeakyReLU(0.1),
            nn.Linear(channels[1]//2, channels[1], bias=True)
        )
        self.up3 =  nn.Sequential(
            nn.Linear(channels[2]//4, channels[2]//2, bias=True),
            nn.LayerNorm(channels[2]//2),
            nn.LeakyReLU(0.1),
            nn.Linear(channels[2]//2, channels[2], bias=True)
        )
        self.up4 =  nn.Sequential(
            nn.Linear(channels[3]//4, channels[3]//2, bias=True),
            nn.LayerNorm(channels[3]//2),
            nn.LeakyReLU(0.1),
            nn.Linear(channels[3]//2, channels[3], bias=True)
        )
        self.up5 =  nn.Sequential(
                    nn.Linear(channels[3]//4, channels[3]//2, bias=True),
                    nn.LayerNorm(channels[3]//2),
                    nn.LeakyReLU(0.1),
                    nn.Linear(channels[3]//2, channels[3], bias=True)
                )
class Projector(nn.Module):
    def __init__(self, channels, dropout=0.1):
        super(Projector, self).__init__()

        self.projector1 = nn.Sequential(
            nn.Linear(channels[0], channels[0]//2, bias=True),
            nn.LayerNorm(channels[0]//2),
            nn.LeakyReLU(0.1),
            # nn.Dropout(dropout),
            nn.Linear(channels[0]//2, channels[0]//4, bias=True)
        )
        self.projector2 = nn.Sequential(
            nn.Linear(channels[1], channels[1]//2, bias=True),
            nn.LayerNorm(channels[1]//2),
            nn.LeakyReLU(0.1),
            # nn.Dropout(dropout),
            nn.Linear(channels[1]//2, channels[1]//4, bias=True)
        )
        self.projector3 = nn.Sequential(
            nn.Linear(channels[2], channels[2]//2, bias=True),
            nn.LayerNorm(channels[2]//2),
            nn.LeakyReLU(0.1),
            # nn.Dropout(dropout),
            nn.Linear(channels[2]//2, channels[2]//4, bias=True)
        )
        self.projector4 = nn.Sequential(
            nn.Linear(channels[3], channels[3]//2, bias=True),
            nn.LayerNorm(channels[3]//2),
            nn.LeakyReLU(0.1),
            # nn.Dropout(dropout),
            nn.Linear(channels[3]//2, channels[3]//4, bias=True)
        )
        self.projector5 = nn.Sequential(
            nn.Linear(channels[3], channels[3] // 2, bias=True),
            nn.LayerNorm(channels[3] // 2),
            nn.LeakyReLU(0.1),
            # nn.Dropout(dropout),
            nn.Linear(channels[3] // 2, channels[3] // 4, bias=True)
        )


class ClassWiseDistributionAlignment(nn.Module):
    def __init__(self):
        super(ClassWiseDistributionAlignment, self).__init__()
        self.channels = [128, 128, 128, 128, 128]
        # self.de_channels = 128
        self.pro = Projector(self.channels, dropout=0.0)
        self.attention_align0 = AttentionAlignment(self.channels[0]//4)
        self.attention_align1 = AttentionAlignment(self.channels[1]//4)
        self.attention_align2 = AttentionAlignment(self.channels[2]//4)
        self.attention_align3 = AttentionAlignment(self.channels[3]//4)
        self.attention_align4 = AttentionAlignment(self.channels[4]//4)
        self.decoder = Decoder(self.channels, dropout=0.0)


    def forward(self, f1, f2):


        brain_features = self.pro.projector1(f1)
        audio_features = self.pro.projector1(f2)
        # Align audio features to brain features using attention
        brain_features = self.attention_align0(brain_features, audio_features)
        audio_features = self.attention_align0(audio_features, brain_features)
        # Decode features to higher resolution
        brain_features = self.decoder.up1(brain_features)
        audio_features = self.decoder.up1(audio_features)
        # -----------------------------------------------------------------------

        brain_features = self.pro.projector2(brain_features)
        audio_features = self.pro.projector2(audio_features)
        # Align audio features to brain features using attention
        brain_features = self.attention_align1(brain_features, audio_features)
        audio_features = self.attention_align1(audio_features, brain_features)
        # Decode features to higher resolution
        brain_features = self.decoder.up2(brain_features)
        audio_features = self.decoder.up2(audio_features)

        # -----------------------------------------------------------------------
        brain_features = self.pro.projector3(brain_features)
        audio_features = self.pro.projector3(audio_features)
        # Align audio features to brain features using attention
        brain_features = self.attention_align2(brain_features, audio_features)
        audio_features = self.attention_align2(audio_features, brain_features)
        # Decode features to higher resolution
        brain_features = self.decoder.up3(brain_features)
        audio_features = self.decoder.up3(audio_features)

        # -----------------------------------------------------------------------
        brain_features = self.pro.projector4(brain_features)
        audio_features = self.pro.projector4(audio_features)
        # Align audio features to brain features using attention
        brain_features = self.attention_align3(brain_features, audio_features)
        audio_features = self.attention_align3(audio_features, brain_features)
        # Decode features to higher resolution
        brain_features = self.decoder.up4(brain_features)
        audio_features = self.decoder.up4(audio_features)

        brain_features = self.pro.projector5(brain_features)
        audio_features = self.pro.projector5(audio_features)
        # Align audio features to brain features using attention
        brain_features = self.attention_align4(brain_features, audio_features)
        audio_features = self.attention_align4(audio_features, brain_features)
        brain_features = self.decoder.up5(brain_features)
        audio_features = self.decoder.up5(audio_features)

        return brain_features, audio_features

def global_average_pooling(x):
    return F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
