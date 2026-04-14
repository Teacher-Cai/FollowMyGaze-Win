import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data_process_dnn import GazeDataset
from gaze_feature_based_model import ResNet
from global_info import GlobalInfo


class GazeController:
    def __init__(self, ui_callback=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ResNet().to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.ui_callback = ui_callback
        self.training = False
        self.is_predict = False

        # Load existing model if exists
        if os.path.exists("gaze_model_resnet.pth"):
            self.model.load_state_dict(torch.load("gaze_model_resnet.pth", map_location=self.device))

    def train_model(self, is_online_mode=True):
        if self.training:
            return
        self.training = True
        if self.ui_callback:
            self.ui_callback("training_started")
        print("Starting model training...")
        dataset = GlobalInfo.train_data

        dataset.load_all_samples()

        training_batch_size = GlobalInfo.online_training_batchSize if is_online_mode else GlobalInfo.offline_training_batchSize

        dataloader = DataLoader(dataset, batch_size=training_batch_size, shuffle=True)
        self.model.train()

        training_epoch = GlobalInfo.online_training_epoch if is_online_mode else GlobalInfo.offline_training_epoch
        for epoch in range(training_epoch):
            total_loss = 0
            for images, targets in dataloader:
                images, targets = images.to(self.device), targets.to(self.device)
                targets = targets.float()

                self.optimizer.zero_grad()
                images = torch.FloatTensor(images.float())
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch [{epoch + 1}/{training_epoch}], Loss: {total_loss / len(dataloader):.4f}")
            if self.ui_callback:
                self.ui_callback("training_progress", epoch + 1, total_loss / len(dataloader))

        # Save trained model
        torch.save(self.model.state_dict(), "gaze_model_resnet.pth")
        self.model.eval()
        print("Model training completed.")
        self.training = False
        dataset.clear_samples()

    def predict_gaze(self, features):
        if self.is_predict:
            return
        self.is_predict = True
        with torch.no_grad():
            feature_tensor = torch.FloatTensor(features.reshape(1, -1))
            image = feature_tensor
            image = image.to(self.device)
            output = self.model(image)
            coords = output.cpu().numpy()[0]

            x = int(coords[0])
            y = int(coords[1])

            self.is_predict = False

            return max(0, min(x, GlobalInfo.screen_width)), max(0, min(y, GlobalInfo.screen_height))


if __name__ == "__main__":
    GlobalInfo.train_data = GazeDataset()
    aaa = GazeController()
    aaa.train_model(is_online_mode=False)
