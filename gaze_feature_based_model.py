import torch.nn as nn
import torch


class SimpleDNN(nn.Module):
    def __init__(self):
        super(SimpleDNN, self).__init__()

        self.fc_layers = nn.Sequential(
            nn.Linear(81, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


class ResNet(nn.Module):
    def __init__(self, input_size=81, output_size=2):
        super(ResNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.out = nn.Linear(32, output_size)

    def forward(self, x):
        # residual = x  # 保存输入作为残差
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # 残差连接
        x = torch.relu(x)
        x = self.out(x)
        return x


# ===================== GazeMoE 模型组件 =====================

class CrossNet(nn.Module):
    """显式特征交叉网络，捕捉特征间二阶及以上交互。"""

    def __init__(self, input_dim, num_cross_layers=1):
        super(CrossNet, self).__init__()
        self.num_cross_layers = num_cross_layers
        self.w = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim, 1) * 0.01)
            for _ in range(num_cross_layers)
        ])
        self.b = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim))
            for _ in range(num_cross_layers)
        ])

    def forward(self, x):
        x0 = x
        xl = x
        for i in range(self.num_cross_layers):
            xl = x0 * (xl @ self.w[i]) + self.b[i] + xl
        return xl


class ResidualBlock(nn.Module):
    """带 BatchNorm + Dropout 的残差块。"""

    def __init__(self, in_dim, out_dim, dropout=0.3):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # 维度不匹配时用线性投影做 shortcut
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.fc(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out + residual


class GazeMoE(nn.Module):
    """GazeMoE —— 门控混合专家视线模型（Mixture of Experts for Gaze）。

    架构（两个完全同构的专家）：
      - Iris 专家：全 124 维 → CrossNet + 4 ResBlocks → y_iris（虹膜偏转贡献量）
      - Head 专家：全 124 维 → CrossNet + 4 ResBlocks → y_head（头部姿态贡献量，base 预测）
      - Gate 路由：基于 ||rel|| 物理先验的二值化门控（STE）
      - 输出 = y_head + gate * y_iris
    """

    # 特征向量中虹膜相对眼眶中心偏移的索引（124 维特征，对齐 Mac 版）
    # 索引 8: eye_l_rel_x, 9: eye_l_rel_y
    # 索引 16: eye_r_rel_x, 17: eye_r_rel_y
    GATE_FEATURE_INDICES = (8, 9, 16, 17)

    # —— 门控参数（Otsu + EMA 自适应阈值）——
    OTSU_N_BINS = 50
    EMA_MOMENTUM = 0.99
    OTSU_MIN_SAMPLES = 16
    K_SHARP = 5.0
    GATE_RESIDUAL_SCALE = 0.0

    def __init__(self, input_size=124, output_size=2, dropout=0.3,
                 num_cross_layers=1, hidden_dim=128):
        super(GazeMoE, self).__init__()
        self.input_size = input_size

        # ============ 输入特征标准化参数（作为 buffer）============
        self.register_buffer("feat_mean", torch.zeros(input_size))
        self.register_buffer("feat_std", torch.ones(input_size))
        self.register_buffer("feat_norm_ready", torch.tensor(0, dtype=torch.long))

        block_in = hidden_dim + input_size  # 128 + 124 = 252

        # ============ IRIS 专家 ============
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.cross_net = CrossNet(input_size, num_cross_layers=num_cross_layers)
        self.block1 = ResidualBlock(block_in, hidden_dim, dropout)
        self.block2 = ResidualBlock(hidden_dim, 64, dropout)
        self.block3 = ResidualBlock(64, 32, dropout)
        self.block4 = ResidualBlock(32, 16, dropout)
        # 主干 16 维 + y_head 2 维（detach 拼接 anchor）→ y_iris
        self.output_layer = nn.Linear(16 + output_size, output_size)

        # ============ HEAD 专家（与 IRIS 完全同构）============
        self.head_input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head_cross_net = CrossNet(input_size, num_cross_layers=num_cross_layers)
        self.head_block1 = ResidualBlock(block_in, hidden_dim, dropout)
        self.head_block2 = ResidualBlock(hidden_dim, 64, dropout)
        self.head_block3 = ResidualBlock(64, 32, dropout)
        self.head_block4 = ResidualBlock(32, 16, dropout)
        self.head_output_layer = nn.Linear(16, output_size)

        # ============ Gate 层 ============
        gate_in = len(self.GATE_FEATURE_INDICES) * 2
        self.gate_residual = nn.Sequential(
            nn.Linear(gate_in, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh(),
        )
        # rel BatchNorm 去 DC
        self.rel_norm = nn.BatchNorm1d(len(self.GATE_FEATURE_INDICES), affine=False)
        # Otsu + EMA buffers
        self.register_buffer("threshold_ema", torch.tensor(1.5))
        self.register_buffer("offset_ema_std", torch.tensor(0.5))
        # gate 索引 buffer
        self.register_buffer(
            "_gate_idx_buf",
            torch.tensor(self.GATE_FEATURE_INDICES, dtype=torch.long),
            persistent=False,
        )

    def _normalize_input(self, x):
        """对输入特征做 Z-score 标准化（训练时更新 EMA，推理时用 buffer）。"""
        if self.training:
            with torch.no_grad():
                batch_mean = x.mean(dim=0)
                batch_var = x.var(dim=0, unbiased=False)
                batch_std = torch.sqrt(batch_var + 1e-8)
                if self.feat_norm_ready == 0:
                    self.feat_mean.copy_(batch_mean)
                    self.feat_std.copy_(batch_std)
                    self.feat_norm_ready.fill_(1)
                else:
                    self.feat_mean.mul_(self.EMA_MOMENTUM).add_(batch_mean, alpha=1 - self.EMA_MOMENTUM)
                    self.feat_std.mul_(self.EMA_MOMENTUM).add_(batch_std, alpha=1 - self.EMA_MOMENTUM)
        std = self.feat_std.clamp(min=1e-8)
        return (x - self.feat_mean) / std

    def _compute_gate(self, x):
        """基于虹膜偏移 ||rel|| 的物理先验门控（STE 二值化）。

        使用原始特征 x（未标准化）以保留物理尺度。
        """
        B = x.shape[0]
        idx = self._gate_idx_buf  # (4,)
        rel = x[:, idx]           # (B, 4)
        # 左右眼偏移量平方和
        rel_sq = rel.pow(2)
        left_norm2 = rel_sq[:, 0:2].sum(dim=1, keepdim=True)   # (B, 1)
        right_norm2 = rel_sq[:, 2:4].sum(dim=1, keepdim=True)  # (B, 1)
        rel_norm = torch.sqrt(torch.cat([left_norm2, right_norm2], dim=1))  # (B, 2)
        # 取左右眼最大偏移量作为门控信号
        rel_max = rel_norm.max(dim=1, keepdim=True)[0]  # (B, 1)

        # —— Otsu 自适应阈值（EMA 平滑）——
        if self.training and B >= self.OTSU_MIN_SAMPLES:
            with torch.no_grad():
                thresh = self._otsu_threshold(rel_max.squeeze())
                self.threshold_ema.mul_(self.EMA_MOMENTUM).add_(thresh, alpha=1 - self.EMA_MOMENTUM)
                self.offset_ema_std.mul_(self.EMA_MOMENTUM).add_(
                    rel_max.std(), alpha=1 - self.EMA_MOMENTUM)

        threshold = self.threshold_ema
        offset_std = self.offset_ema_std.clamp(min=1e-6)

        # sigmoid 锐化门控
        prior_logit = self.K_SHARP * (rel_max - threshold) / offset_std
        prior_gate = torch.sigmoid(prior_logit)  # (B, 1)

        # 可学习微调残差
        rel_normed = self.rel_norm(rel)  # (B, 4)
        residual = self.gate_residual(rel_normed) * self.GATE_RESIDUAL_SCALE  # (B, 1)

        gate_soft = prior_gate + residual

        # —— STE 二值化（训练时 hard，推理时 soft）——
        if self.training:
            gate_hard = (gate_soft > 0.5).float()
            gate = gate_hard - gate_soft.detach() + gate_soft  # STE
        else:
            gate = gate_soft

        return gate, prior_gate

    def _otsu_threshold(self, values):
        """Otsu 二值化阈值计算。"""
        values = values.detach()
        vmin, vmax = values.min(), values.max()
        if vmax - vmin < 1e-6:
            return vmin
        bins = torch.linspace(vmin, vmax, steps=self.OTSU_N_BINS + 1, device=values.device)
        hist = torch.histc(values, bins=self.OTSU_N_BINS, min=vmin, max=vmax)
        hist = hist / hist.sum()
        max_var, best_thresh = 0.0, vmin
        w0 = 0.0
        sum_all = (torch.arange(self.OTSU_N_BINS, device=values.device, dtype=torch.float) * hist).sum()
        sum0 = 0.0
        for t in range(self.OTSU_N_BINS):
            w0 += hist[t].item()
            if w0 == 0 or w0 == 1:
                continue
            sum0 += t * hist[t].item()
            w1 = 1.0 - w0
            mean0 = sum0 / w0
            mean1 = (sum_all - sum0) / w1
            var_between = w0 * w1 * (mean0 - mean1) ** 2
            if var_between > max_var:
                max_var = var_between
                best_thresh = bins[t]
        return best_thresh

    def forward(self, x):
        x0 = x                          # 原始特征：供门控（Otsu 物理尺度）与 anchor 使用
        xn = self._normalize_input(x0)  # 标准化特征：供两个专家分支使用

        # ============ HEAD 专家 ============
        h1 = self.head_input_layer(xn)
        h2 = self.head_cross_net(xn)
        h_main = torch.cat([h1, h2], dim=1)
        h_main = self.head_block1(h_main)
        h_main = self.head_block2(h_main)
        h_main = self.head_block3(h_main)
        h_main = self.head_block4(h_main)           # (B, 16)
        y_head = self.head_output_layer(h_main)     # (B, 2)

        # ============ IRIS 专家 ============
        x1 = self.input_layer(xn)
        x2 = self.cross_net(xn)
        x_main = torch.cat([x1, x2], dim=1)
        x_main = self.block1(x_main)
        x_main = self.block2(x_main)
        x_main = self.block3(x_main)
        x_main = self.block4(x_main)  # (B, 16)
        # 把 y_head 作为 anchor 拼接到 iris 主干特征（detach 避免污染 y_head 的语义）
        x_with_head = torch.cat([x_main, y_head.detach()], dim=1)   # (B, 18)
        y_iris = self.output_layer(x_with_head)     # (B, 2)

        # 门控融合
        gate, prior_gate = self._compute_gate(x0)
        y_final = y_head + gate * y_iris

        if self.training:
            return y_final, y_iris, y_head, gate, prior_gate
        return y_final