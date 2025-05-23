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


# ================== 改进的EEG处理模块 ==================
class EEGMultiScaleNet(nn.Module):
    def __init__(self, in_channels=32, time_len=585, feat_dim=256):
        super().__init__()
        # 多尺度时序卷积（参考网页1/6）
        self.temp_conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, 15, padding=7),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.4),  # 添加 Dropout 层
            nn.MaxPool1d(4),
            nn.Conv1d(64, feat_dim, 5, padding=2),
            nn.BatchNorm1d(feat_dim),
            nn.GELU(),
            nn.Dropout(0.4),  # 添加 Dropout 层
            nn.MaxPool1d(2)
        )

        # 频域处理分支（参考网页4）
        self.freq_branch = nn.Sequential(
            nn.Linear(time_len, 128),
            nn.GELU(),
            nn.Linear(128, 64)
        )

        # BiLSTM时序建模（参考网页1）
        self.bilstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=feat_dim // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim + 64, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU()
        )

    def forward(self, x):
        # 时序特征 [B, C, T] -> [B, F, T']
        temp_feat = self.temp_conv(x)

        # 频域特征 [B, C, T] -> [B, F]
        freq_feat = self.freq_branch(x.mean(1))

        # BiLSTM处理 [B, F, T'] -> [B, T', F]
        temp_feat = temp_feat.permute(0, 2, 1)
        lstm_out, _ = self.bilstm(temp_feat)

        # 时空特征聚合
        temp_feat = lstm_out.mean(1)  # [B, F]

        # 特征融合
        fused = self.fusion(torch.cat([temp_feat, freq_feat], dim=1))
        return fused


class Subnetwork(nn.Module):
    def __init__(self, input_dim, feat_dim=256, num_layers=2, nhead=4):
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
    def __init__(self, embed_dim=256, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim  # 存储嵌入维度，方便后续使用
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True  # 确保批次维度是第一个维度
        )
        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),  # 输入是两个嵌入向量的拼接
            nn.Sigmoid()  # 使用Sigmoid函数进行门控
        )
        self.norm = nn.LayerNorm(embed_dim)  # 层归一化

    def forward(self, query, key, value):
        # 输入形状: [batch, embed_dim]
        # query = query.unsqueeze(1)  # [batch, 1, embed_dim]  # 移除unsqueeze，因为可能不需要
        # key = key.unsqueeze(1)  # [batch, 1, embed_dim]    # 移除unsqueeze，因为可能不需要
        # value = value.unsqueeze(1)  # [batch, 1, embed_dim]  # 移除unsqueeze，因为可能不需要

        # 确保输入具有 MultiheadAttention 期望的形状 (batch, seq_len, embed_dim)
        # 如果输入是 (batch, embed_dim)，则将其reshape为 (batch, 1, embed_dim)
        if query.ndim == 2:
            query = query.unsqueeze(1)
        if key.ndim == 2:
            key = key.unsqueeze(1)
        if value.ndim == 2:
            value = value.unsqueeze(1)

        attn_output, _ = self.multihead_attn(
            query=query,
            key=key,
            value=value
        )
        attn_output = attn_output.squeeze(1)  # 移除序列长度维度

        # 门控融合
        gate = self.gate(torch.cat([query.squeeze(1), attn_output], dim=1))  # 在拼接之前squeeze query
        output = gate * query.squeeze(1) + (1 - gate) * attn_output  # 在相乘之前squeeze query
        return self.norm(output)


class MultimodalTransformerModel(nn.Module):
    def __init__(self, num_classes=3, temperature=0.01):
        super().__init__()
        # 输入维度定义
        self.eeg_channels = 32
        self.eeg_time = 585
        self.eye_dim = 38
        self.pps_dim = 230

        # 子模态特征提取
        self.eeg_net = EEGMultiScaleNet()
        self.eye_net = Subnetwork(self.eye_dim)
        self.pps_net = Subnetwork(self.pps_dim)

        # 跨模态注意力
        self.cross_attn_e2p = CrossModalTransformer()
        self.cross_attn_p2e = CrossModalTransformer()

        # 模态融合权重
        self.attention_weights = nn.Sequential(
            nn.Linear(256 * 3, 64),
            nn.GELU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=1)
        )

        # 融合网络
        self.fusion = nn.Sequential(
            nn.Linear(256 * 3, 256),
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
        self.valence_head = nn.Sequential(
            # 增加输入层宽度
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),

            # 新增隐藏层
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3),

            # 保持原有的中间层
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
        pos_mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
        pos_mask.fill_diagonal_(0)  # 排除自身样本

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
        loss = -torch.log((pos_sim + 1e-12) / (all_sim + 1e-12))

        # 计算平均损失
        return loss.mean()

    def forward(self, eeg, eye, pps, labels=None):
        # 特征提取
        eeg_feat = self.eeg_net(eeg)
        eye_feat = self.eye_net(eye)
        pps_feat = self.pps_net(pps)

        # Stack features as sequence of length 3 for attention: shape (3, batch, feat_dim)
        feats = torch.stack([eeg_feat, eye_feat, pps_feat], dim=0)

        eeg_contrastive_loss = 0
        if labels is not None:
            arousal_labels = labels[0]
            eeg_contrastive_loss += self.compute_contrastive_loss(eeg_feat, eeg_feat, arousal_labels)

        eye_contrastive_loss = 0
        if labels is not None:
            arousal_labels = labels[0]
            eye_contrastive_loss += self.compute_contrastive_loss(eye_feat, eye_feat, arousal_labels)

        pps_contrastive_loss = 0
        if labels is not None:
            arousal_labels = labels[0]
            pps_contrastive_loss += self.compute_contrastive_loss(pps_feat, pps_feat, arousal_labels)

        # 双向跨模态注意力
        eye_enhanced = self.cross_attn_e2p(
            query=eeg_feat,
            key=eye_feat,
            value=eye_feat
        )

        pps_enhanced = self.cross_attn_p2e(
            query=eeg_feat,
            key=pps_feat,
            value=pps_feat
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

        eeg_contrastive_loss = self.contrastive_weight * eeg_contrastive_loss
        eye_contrastive_loss = self.contrastive_weight * eye_contrastive_loss
        pps_contrastive_loss = self.contrastive_weight * pps_contrastive_loss

        if labels is None:
            return arousal, valence  # 修改返回格式
        else:
            return arousal, valence, eeg_contrastive_loss, eye_contrastive_loss, pps_contrastive_loss  # 修改返回格式


class EyeMLPNet(nn.Module):
    def __init__(self, input_dim=38, feat_dim=256):
        super(EyeMLPNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, feat_dim),
            nn.ReLU(),
            nn.BatchNorm1d(feat_dim)
        )

    def forward(self, x):
        return self.net(x)


class PPSMLPNet(nn.Module):
    def __init__(self, input_dim=230, feat_dim=256):
        super(PPSMLPNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, feat_dim),
            nn.ReLU(),
            nn.BatchNorm1d(feat_dim)
        )

    def forward(self, x):
        return self.net(x)


class MultiModalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入维度定义
        self.eeg_channels = 32
        self.eeg_time = 585
        self.eye_dim = 38
        self.pps_dim = 230
        self.feat_dim = 256
        self.num_heads = 8

        # 子模态特征提取
        self.eeg_net = EEGMultiScaleNet()
        self.eye_net = EyeMLPNet(self.eye_dim, self.feat_dim)
        self.pps_net = PPSMLPNet(self.pps_dim, self.feat_dim)

        # Multi-head Attention for feature fusion (8 heads as in related work&#8203;:contentReference[oaicite:4]{index=4})
        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.feat_dim, num_heads=self.num_heads,
                                                    batch_first=False)
        # 融合后的小MLP，进一步处理
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.feat_dim)
        )

    def forward(self, eeg, eye, pps, labels=None):
        # 特征提取
        # eeg_feat = self.eeg_net(eeg)
        # eye_feat = self.eye_net(eye)
        # pps_feat = self.pps_net(pps)
        eeg_feat = F.normalize(self.eeg_net(eeg), dim=-1)
        eye_feat = F.normalize(self.eye_net(eye), dim=-1)
        pps_feat = F.normalize(self.pps_net(pps), dim=-1)

        # Stack features and reshape for attention: shape (batch, 3, feat_dim)
        feats = torch.stack([eeg_feat, eye_feat, pps_feat], dim=1)  # Changed dim=0 to dim=1

        # Perform attention (note: batch_first=False in MultiheadAttention init)
        feats = feats.transpose(0, 1)  # Convert to (3, batch, feat_dim) for attention
        attn_out, _ = self.multihead_attn(feats, feats, feats)
        attn_out = attn_out.transpose(0, 1)  # Convert back to (batch, 3, feat_dim)

        # Max pooling over modality dimension (dim=1) to get (batch, feat_dim)
        fused = attn_out.max(dim=1)[0]  # Using [0] to get values from max

        # Final MLP processing
        fused = self.fusion_mlp(fused)

        return fused


class ProjectionHead(nn.Module):
    """
    MLP projection head for contrastive learning (e.g., SimCLR-style).
    """

    def __init__(self, in_dim=256, hidden_dim=256, out_dim=128):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_dim),
            nn.Dropout(0.5),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class Classifier(nn.Module):
    """
    Classifier head for fine-tuning: outputs two logits (arousal, valence).
    """

    def __init__(self, in_dim=256, hidden_dim=128):
        super(Classifier, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.fc_arousal = nn.Linear(hidden_dim, 3)  # binary arousal (2 classes)
        self.fc_valence = nn.Linear(hidden_dim, 3)  # binary valence (2 classes)

    def forward(self, x):
        h = self.shared(x)
        out_a = self.fc_arousal(h)
        out_v = self.fc_valence(h)
        return out_a, out_v
