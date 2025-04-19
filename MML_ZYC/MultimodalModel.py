import math
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class EEGTransformer(nn.Module):
    def __init__(self, in_channels=32, time_length=585, feat_dim=128, nhead=2, num_layers=4):  # 增加层数
        super().__init__()
        self.conv_proj = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64, eps=1e-5),
            nn.GELU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, feat_dim, kernel_size=5, padding=2),  # 减小卷积核大小
            nn.BatchNorm1d(feat_dim, eps=1e-5),
            nn.GELU(),
            nn.MaxPool1d(2),
        )

        conv_time = time_length // 2 // 2   # 更新时间维度
        self.pos_encoder = PositionalEncoding(feat_dim, conv_time)

        encoder_layers = TransformerEncoderLayer(
            d_model=feat_dim,
            nhead=nhead,
            dim_feedforward=feat_dim * 3,
            dropout=0.3,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layers, num_layers)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(feat_dim, feat_dim)
        self.dropout = nn.Dropout(0.3)  # 添加 Dropout

    def forward(self, x):
        x = self.conv_proj(x)
        x = x.transpose(1, 2)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.transpose(1, 2)
        x = self.global_pool(x).squeeze(-1)
        return self.fc(self.dropout(x))  # 应用 Dropout



class TransformerSubnetwork(nn.Module):
    def __init__(self, input_dim, feat_dim=128, num_layers=2, nhead=2):
        super().__init__()
        self.proj = nn.Linear(input_dim, feat_dim)
        self.pos_encoder = PositionalEncoding(feat_dim, max_len=100)

        encoder_layers = TransformerEncoderLayer(
            d_model=feat_dim,
            nhead=nhead,
            dim_feedforward=feat_dim * 3,
            dropout=0.3,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layers, num_layers)
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, x):
        # x shape: [batch, input_dim]

        x = self.proj(x).unsqueeze(1)  # [batch, 1, feat_dim]
        x = self.pos_encoder(x)
        x = self.transformer(x)  # [batch, 1, feat_dim]
        return self.norm(x.squeeze(1))  # [batch, feat_dim]


class CrossModalTransformer(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query, key, value):
        # 输入形状: [batch, embed_dim]
        query = query.unsqueeze(1)  # [batch, 1, embed_dim]
        key = key.unsqueeze(1)  # [batch, 1, embed_dim]
        value = value.unsqueeze(1)  # [batch, 1, embed_dim]

        attn_output, _ = self.multihead_attn(
            query=query,
            key=key,
            value=value
        )
        output = self.norm(query + attn_output)
        return output.squeeze(1)  # [batch, embed_dim]


class MultimodalTransformerModel(nn.Module):
    def __init__(self, num_classes=3, temperature=0.01):
        super().__init__()
        # 输入维度定义
        self.eeg_channels = 32
        self.eeg_time = 585
        self.eye_dim = 38
        self.pps_dim = 230

        # 子模态特征提取
        self.eeg_net = EEGTransformer()
        self.eye_net = TransformerSubnetwork(self.eye_dim)
        self.pps_net = TransformerSubnetwork(self.pps_dim)

        # 跨模态注意力
        self.cross_attn_e2p = CrossModalTransformer()
        self.cross_attn_p2e = CrossModalTransformer()

        # 模态融合权重
        self.attention_weights = nn.Sequential(
            nn.Linear(128 * 3, 64),
            nn.GELU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=1)
        )

        # 融合网络
        self.fusion = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.BatchNorm1d(256, eps=1e-5),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128, eps=1e-5),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        # 修改输出头部分
        self.arousal_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(128, num_classes)
        )
        # 新增Valence分类头
        self.valence_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(64, num_classes)
        )

        # 添加可学习的对比损失权重参数
        self.contrastive_weight = nn.Parameter(torch.ones(1))
        # 温度参数
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def compute_contrastive_loss(self, feat1, feat2, labels):
        # Normalize features
        feat1 = F.normalize(feat1, dim=1)
        feat2 = F.normalize(feat2, dim=1)
        # 计算特征之间的相似度矩阵，除以温度参数
        sim = torch.mm(feat1, feat2.t()) / self.temperature

        # 创建正样本掩码，用于标识哪些样本对是正样本对
        pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)

        # 减去每行的最大值，避免指数运算时数值溢出
        # 这是LogSumExp技巧的一部分，不影响最终的概率计算
        sim = sim - torch.max(sim, dim=1, keepdim=True)[0]

        # 计算指数相似度
        exp_sim = torch.exp(sim)

        # 计算正样本的指数相似度之和
        pos_sim = (exp_sim * pos_mask).sum(1)

        # 计算所有样本的指数相似度之和
        all_sim = exp_sim.sum(1)

        # 计算InfoNCE损失，使用对数概率避免数值问题
        loss = -torch.log(pos_sim / all_sim)

        # 计算平均损失
        return loss.mean()

    def forward(self, eeg, eye, pps, labels=None):
        # 特征提取
        eeg_feat = self.eeg_net(eeg)  # [batch, 128]
        eye_feat = self.eye_net(eye)  # [batch, 128]
        pps_feat = self.pps_net(pps)  # [batch, 128]

        contrastive_loss = 0
        if labels is not None:
            arousal_labels = labels[0]
            valence_labels = labels[1]  # 新增valence标签
            contrastive_loss += self.compute_contrastive_loss(eeg_feat, eye_feat, arousal_labels)
            contrastive_loss += self.compute_contrastive_loss(eeg_feat, pps_feat, arousal_labels)
            contrastive_loss += self.compute_contrastive_loss(eye_feat, pps_feat, arousal_labels)

        # 双向跨模态注意力
        eye_enhanced = self.cross_attn_e2p(
            query=eye_feat,
            key=eeg_feat,
            value=eeg_feat
        )

        pps_enhanced = self.cross_attn_p2e(
            query=pps_feat,
            key=eeg_feat,
            value=eeg_feat
        )

        # 动态加权融合
        weights = self.attention_weights(
            torch.cat([eeg_feat, eye_feat, pps_feat], dim=1)
        )
        fused = torch.cat([
            eeg_feat * weights[:, 0:1],
            eye_enhanced * weights[:, 1:2],
            pps_enhanced * weights[:, 2:3]
        ], dim=1)

        # 最终特征
        fused = self.fusion(fused)

        # 修改输出部分
        arousal = self.arousal_head(fused)
        valence = self.valence_head(fused)  # 新增输出

        contrastive_loss = self.contrastive_weight * contrastive_loss
        if labels is None:
            return arousal, valence  # 修改返回格式
        else:
            return arousal, valence, contrastive_loss  # 修改返回格式
