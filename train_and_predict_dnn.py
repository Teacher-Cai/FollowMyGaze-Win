import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_process_dnn import GazeDataset
from gaze_feature_based_model import ResNet, GazeMoE
from global_info import GlobalInfo


class GazeController:
    # 多任务监督权重
    LOSS_W_FINAL = 1.0       # 最终预测的主 loss
    LOSS_W_HEAD_AUX = 0.8    # 仅在虹膜居中时生效，适度保留
    LOSS_W_IRIS_AUX = 0.5    # 提高权重，让 y_iris 获得足够梯度
    LOSS_W_IRIS_ZERO_LAMBDA = 1.0  # iris_aux 内归零项的相对弱化系数

    def __init__(self, ui_callback=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GazeMoE().to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.ui_callback = ui_callback
        self.training = False
        self.is_predict = False
        # EMA 平滑状态
        self._ema_x = None
        self._ema_y = None
        self._ema_alpha = 0.9  # 平滑系数：越小越平滑但延迟越大
        # Load existing model if exists
        model_path = os.path.join(GlobalInfo.path_dir, "gaze_moe_model.pth")
        if os.path.exists(model_path):
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device),
                strict=False,
            )

    def _compute_loss(self, outputs, targets):
        """多任务监督损失（gate 关系修正版）。

        语义：
        - y_head : 头部姿态贡献的注视点位置（base 预测）
        - y_iris : 虹膜偏转贡献的位移量
        - y_final = y_head + gate * y_iris

        三个 loss 的 gate 加权逻辑（按物理直觉）：
        - gate ≈ 0 (虹膜居中)：label ≈ 头部姿态决定的位置
                · 应强监督 y_head ≈ label（这是干净标签）
                · 应强制 y_iris ≈ 0（虹膜没动，不该有贡献）
        - gate ≈ 1 (虹膜偏转)：label 包含虹膜贡献
                · 应弱监督 y_head（避免污染）
                · 应让 y_iris 学到完整的虹膜贡献量
        - loss_final    : MSE(y_head + gate*y_iris, label)              — 主目标
        - loss_head_aux : (1-gate.detach()) * (y_head - label)^2        — 虹膜居中时监督 y_head
        - loss_iris_aux : gate.detach() * (y_head.detach()+y_iris-label)^2
                    + (1-gate.detach()) * y_iris^2
                    — 虹膜偏转时学贡献量，虹膜居中时归零
        """
        y_final, y_iris, y_head, gate, prior_gate = outputs

        loss_final = self.criterion(y_final, targets)

        gate_d = gate.detach()                                   # (B, 1)
        prior_d = prior_gate.detach()                            # (B, 1)
        residual_abs = (gate_d - prior_d).abs()                  # 检查 residual 贡献

        # head_aux：仅在 gate 小（虹膜居中）时强监督 y_head
        head_err = (y_head - targets).pow(2)                     # (B, 2)
        loss_head_aux = ((1.0 - gate_d) * head_err + gate_d * head_err * 0.5).mean()

        # iris_aux：gate 大时学贡献量，gate 小时归零
        err_contrib = (y_head.detach() + y_iris - targets).pow(2)
        err_zero = y_iris.pow(2)
        loss_iris_aux = (gate_d * err_contrib
                         + self.LOSS_W_IRIS_ZERO_LAMBDA * (1.0 - gate_d) * err_zero).mean()

        total = (self.LOSS_W_FINAL * loss_final
                 + self.LOSS_W_HEAD_AUX * loss_head_aux
                 + self.LOSS_W_IRIS_AUX * loss_iris_aux)

        return total, {
            'final': loss_final.item(),
            'head_aux': loss_head_aux.item(),
            'iris_aux': loss_iris_aux.item(),
            'gate_mean': gate_d.mean().item(),
            'gate_min': gate_d.min().item(),
            'gate_max': gate_d.max().item(),
            'gate_q25': gate_d.quantile(0.25).item(),
            'gate_q75': gate_d.quantile(0.75).item(),
            'prior_gate_mean': prior_d.mean().item(),
            'gate_residual_abs_mean': residual_abs.mean().item(),
            'y_iris_abs_mean': y_iris.detach().abs().mean().item(),
        }

    def train_model(self, is_online_mode=True, clear_after=True):
        if self.training:
            return
        self.training = True
        if self.ui_callback:
            self.ui_callback("training_started")
        print("Starting model training...")
        dataset = GlobalInfo.train_data
        if is_online_mode:
            dataset.use_current_samples()
        else:
            dataset.load_all_samples()
        training_batch_size = GlobalInfo.online_training_batchSize if is_online_mode else GlobalInfo.offline_training_batchSize
        dataloader = DataLoader(dataset, batch_size=training_batch_size, shuffle=True)

        # —— 影子模型 shadow：训练不动 self.model，避免与推理线程抢模型状态 ——
        shadow = GazeMoE().to(self.device)
        shadow.load_state_dict(self.model.state_dict())
        shadow.train()
        shadow_optim = optim.Adam(shadow.parameters(),
                                  lr=self.optimizer.param_groups[0]['lr'])

        training_epoch = GlobalInfo.online_training_epoch if is_online_mode else GlobalInfo.offline_training_epoch
        for epoch in range(training_epoch):
            total_loss = 0
            for images, targets in dataloader:
                images, targets = images.to(self.device), targets.to(self.device)
                targets = targets.float()
                shadow_optim.zero_grad()
                images = torch.FloatTensor(images.float())
                outputs = shadow(images)
                loss, loss_info = self._compute_loss(outputs, targets)
                loss.backward()
                shadow_optim.step()
                total_loss += loss.item()
            print(f"Epoch [{epoch + 1}/{training_epoch}], Loss: {total_loss / len(dataloader):.4f}")
            if self.ui_callback:
                self.ui_callback("training_progress", epoch + 1, total_loss / len(dataloader))

        # 训练完成：切 eval，持久化，然后原子替换推理模型
        shadow.eval()
        model_path = os.path.join(GlobalInfo.path_dir, "gaze_moe_model.pth")
        torch.save(shadow.state_dict(), model_path)
        # Python 属性赋值本身是原子的：推理下一帧就会用新模型
        self.model = shadow
        self.optimizer = shadow_optim
        print("Model training completed.")
        self.training = False
        if self.ui_callback:
            self.ui_callback("training_completed")
        if clear_after:
            dataset.clear_samples()

    def predict_gaze(self, features):
        if self.is_predict:
            return
        self.is_predict = True
        try:
            with torch.no_grad():
                self.model.eval()
                feature_tensor = torch.FloatTensor(features.reshape(1, -1))
                image = feature_tensor.to(self.device)
                output = self.model(image)
                coords = output.cpu().numpy()[0]
                raw_x = float(coords[0])
                raw_y = float(coords[1])
                # EMA 平滑：减少帧间抖动
                if self._ema_x is None:
                    self._ema_x, self._ema_y = raw_x, raw_y
                else:
                    self._ema_x = self._ema_alpha * raw_x + (1 - self._ema_alpha) * self._ema_x
                    self._ema_y = self._ema_alpha * raw_y + (1 - self._ema_alpha) * self._ema_y
                x = int(self._ema_x)
                y = int(self._ema_y)
                return x, y
        finally:
            self.is_predict = False


if __name__ == "__main__":
    GlobalInfo.train_data = GazeDataset()
    aaa = GazeController()
    aaa.train_model(is_online_mode=False)