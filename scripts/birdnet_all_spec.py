import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

batch_size = 16

specdata = np.load("datasets/specdata.npz", allow_pickle=True)

pop_classes = specdata["categories"]

sav_path = "D://Birdnet_all_files_27class_images"


class SpecDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, df, class_names, sav_folder=None):
        """
        Arguments:
            df (pd.DataFrame): Data with annotations.
            root_dir (string): Directory with all the images.
        """
        self.df = df
        self.class_names = class_names
        self.sav_folder = sav_folder
        if sav_folder and not os.path.exists(sav_folder):
            os.makedirs(sav_folder)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.df.iloc[idx]
        wav_name = row.file_path
        # print(row.file_name)
        sav_name = None
        if self.sav_folder:
            sav_name = os.path.join(
                self.sav_folder,
                f'{row["file_name"]}_{int(row["begin_time"])}_{int(row["begin_time"])}.png',
            )
        if sav_name and os.path.exists(sav_name):
            return [], []
        elif os.path.exists(wav_name):
            wav, sr = librosa.core.load(wav_name, sr=None)
            wav_len = 6
            wav_sub = wav[int(int(row["begin_time"]) * sr) : int((int(row["begin_time"]) + wav_len) * sr)]
            if len(wav_sub) / sr != wav_len:
                wav_sub = wav[int((int(row["end_time"]) - wav_len) * sr) : int(int(row["end_time"]) * sr)]
            nfft = 512

            spec = librosa.feature.melspectrogram(y=wav_sub, sr=sr, n_mels=256, hop_length=int(0.75 * nfft))
            spec = librosa.power_to_db(spec, ref=np.max)[:, :256]

            lab = list(self.class_names).index(row.common_name)
            if self.sav_folder:
                plt.imsave(sav_name, spec)
                return [], []
            spec = np.expand_dims(spec, 0)  # channel first

            return spec, lab
        else:
            return None, None


df_filtered = pd.read_csv("C:/Users/Anthony/Downloads/birdnet_df_filtered.csv")

loader = DataLoader(
    dataset=SpecDataset(df_filtered, pop_classes, sav_folder=sav_path),
    batch_size=batch_size,
    shuffle=False,
)

for i, data in enumerate(loader):
    print(f"{i}/{len(loader)}")
