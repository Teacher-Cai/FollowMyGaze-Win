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
