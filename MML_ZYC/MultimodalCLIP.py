import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math


class PositionalEncoding(nn.Module):
    """Transformer位置编码层"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class EEGEncoder(nn.Module):
    """EEG信号编码器"""

    def __init__(self, in_channels=32, time_length=585, feat_dim=128):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, feat_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(feat_dim),
            nn.GELU(),
            nn.MaxPool1d(2),
        )
        self.time_length = time_length // 4  # 经过两次池化
        self.proj = nn.Linear(feat_dim * self.time_length, feat_dim)

    def forward(self, x):
        x = self.conv_net(x)  # [B, C, T]
        x = x.flatten(1)  # [B, C*T]
        return F.normalize(self.proj(x), dim=-1)


class TabularEncoder(nn.Module):
    """表格数据编码器（眼动和生理信号）"""

    def __init__(self, input_dim, feat_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, feat_dim)
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class MultimodalCLIP(nn.Module):
    """基于CLIP思想的多模态情绪识别模型"""

    def __init__(self, num_classes=3, temperature=0.07):
        super().__init__()
        # 各模态编码器
        self.eeg_encoder = EEGEncoder()
        self.eye_encoder = TabularEncoder(38)  # 眼动特征维度
        self.pps_encoder = TabularEncoder(230)  # 生理信号维度

        # 共享投影头
        self.proj = nn.Sequential(
            nn.Linear(128, 128),
            nn.GELU(),
            nn.LayerNorm(128)
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        # 温度参数（可学习）
        self.temperature = nn.Parameter(torch.tensor(temperature))

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, eeg, eye, pps):
        # 特征提取
        eeg_feat = self.eeg_encoder(eeg)  # [B,128]
        eye_feat = self.eye_encoder(eye)  # [B,128]
        pps_feat = self.pps_encoder(pps)  # [B,128]

        # 共享空间投影
        eeg_proj = self.proj(eeg_feat)
        eye_proj = self.proj(eye_feat)
        pps_proj = self.proj(pps_feat)

        # 分类特征融合
        fused = torch.cat([eeg_feat, eye_feat, pps_feat], dim=1)
        logits = self.classifier(fused)

        return {
            'logits': logits,
            'eeg_proj': eeg_proj,
            'eye_proj': eye_proj,
            'pps_proj': pps_proj,
            'temperature': torch.clamp(self.temperature, 0.01, 0.5)
        }


class ContrastiveLoss(nn.Module):
    """多模态对比学习损失"""

    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha  # 模态间损失权重

    def forward(self, features_dict, labels):
        eeg = features_dict['eeg_proj']
        eye = features_dict['eye_proj']
        pps = features_dict['pps_proj']
        temp = features_dict['temperature']

        batch_size = eeg.size(0)

        # 模态内对比损失
        def intra_modal_loss(feat):
            logits = torch.mm(feat, feat.t()) / temp
            targets = torch.arange(batch_size, device=feat.device)
            return F.cross_entropy(logits, targets)

        # 模态间对比损失
        def inter_modal_loss(feat1, feat2):
            sim = torch.mm(feat1, feat2.t()) / temp
            pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)


            # 计算InfoNCE损失
            exp_sim = torch.exp(sim)
            # 计算每个样本的正样本相似度指数之和
            pos_exp_sum = (exp_sim * pos_mask.float()).sum(1)
            total_exp_sum = exp_sim.sum(1)
            pos_loss = -torch.log(pos_exp_sum / total_exp_sum)
            return pos_loss.mean()

        # 计算各项损失
        loss = 0
        # 模态内损失（三种模态）
        loss += intra_modal_loss(eeg)
        loss += intra_modal_loss(eye)
        loss += intra_modal_loss(pps)

        # 模态间损失（三组配对）
        loss += self.alpha * inter_modal_loss(eeg, eye)
        loss += self.alpha * inter_modal_loss(eeg, pps)
        loss += self.alpha * inter_modal_loss(eye, pps)

        return loss / (3 + 3 * self.alpha)  # 加权平均


class MultimodalModel(nn.Module):
    """完整的多模态模型（包含分类和对比学习）"""

    def __init__(self, num_classes=3):
        super().__init__()
        self.clip_model = MultimodalCLIP(num_classes)
        self.contrast_loss = ContrastiveLoss()

    def forward(self, eeg, eye, pps, labels=None):
        outputs = self.clip_model(eeg, eye, pps)

        if labels is not None:
            cls_loss = F.cross_entropy(outputs['logits'], labels)
            contrast_loss = self.contrast_loss(outputs, labels)
            outputs['total_loss'] = cls_loss + 0.3 * contrast_loss  # 可调整权重

        return outputs


# 示例使用
if __name__ == "__main__":
    # 模拟输入数据
    batch_size = 32
    eeg = torch.randn(batch_size, 32, 585)
    eye = torch.randn(batch_size, 38)
    pps = torch.randn(batch_size, 230)
    labels = torch.randint(0, 3, (batch_size,))

    # 初始化模型
    model = MultimodalModel(num_classes=3)

    # 前向传播
    outputs = model(eeg, eye, pps, labels)

    print("模型输出结构:")
    for key, value in outputs.items():
        print(f"{key}: {value.shape if hasattr(value, 'shape') else value}")