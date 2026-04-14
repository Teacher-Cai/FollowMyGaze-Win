import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from global_info import GlobalInfo

# 全局变量
SAMPLE_DIR = "samples"
os.makedirs(SAMPLE_DIR, exist_ok=True)


class GazeDataset(Dataset):
    def __init__(self, samples_dir=SAMPLE_DIR, ui_callback=None):
        self.samples_dir = samples_dir
        self.ui_callback = ui_callback
        self.features = None
        self.labels = None
        self.sample_count = 0
        self.features_new = []
        self.labels_new = []
        self.df = pd.DataFrame()

    def __len__(self):
        return self.sample_count

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def load_all_samples(self):
        if not os.path.exists(self.samples_dir):
            return
        features = []
        labels = []
        # 读取标注数据
        annotations_file = os.path.join(self.samples_dir, 'samples.pkl')
        if not os.path.exists(annotations_file):
            print("Annotations file not found")
            return

        df = pd.read_pickle(annotations_file)

        for index, row in df.iterrows():
            features.append(row['features'])
            labels.append([row['gaze_x'], row['gaze_y']])

        self.features = np.array(features)
        self.labels = np.array(labels)
        self.sample_count = len(self.features)

    def save_sample(self, features, coords):

        self.features_new.append(features)
        self.labels_new.append(coords)

        self.sample_count = len(self.features_new)

        new_row = pd.DataFrame({'features': [features],
                                'gaze_x': coords[0],
                                'gaze_y': coords[1]})

        print(new_row)
        self.df = pd.concat([self.df, new_row], axis=0)  # 沿行方向拼接

        if self.ui_callback:
            self.ui_callback("sample_saved", self.sample_count)
        print(f"Sample saved. Total samples: {self.sample_count}")

    def check_sample_upper_limit(self, df):
        samples_size = len(df)
        GlobalInfo.total_samples_num_int.set(samples_size)
        if samples_size > GlobalInfo.sample_upper_limit:
            df = df[-GlobalInfo.sample_upper_limit:]
            print("over max samples value, remove oldest samples!")

    def use_current_samples(self):
        self.features = np.array(self.features_new)
        self.labels = np.array(self.labels_new)
        print(f"Created {self.sample_count} sample images and annotations")

    def clear_samples(self):
        self.features = None
        self.labels = None
        self.sample_count = 0
        self.features_new.clear()
        self.labels_new.clear()

    def save_current_sample_to_local(self):
        # load local samples first
        if not os.path.exists(self.samples_dir):
            return
        annotations_file = os.path.join(self.samples_dir, 'samples.pkl')
        if not os.path.exists(annotations_file):
            print("Annotations file not found, new one")
            df_local = pd.DataFrame()
        else:
            df_local = pd.read_pickle(annotations_file)

        df_local = pd.concat([df_local, self.df], axis=0)  # 沿行方向拼接
        self.check_sample_upper_limit(df_local)
        df_local.to_pickle(os.path.join(self.samples_dir, 'samples.pkl'))
        print("save sample completed, total samples:", len(df_local))
