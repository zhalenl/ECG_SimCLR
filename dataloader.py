import os

import pandas as pd
import numpy as np
from torchvision.io import read_image
from PIL import Image
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import torchvision.transforms.functional as TF
from multiprocessing import Pool
import pickle
import matplotlib.pyplot as plt
from scipy import signal


class ECGDataset(Dataset):
    def __init__(
        self,
        root="./",
        metadata_dir=f"/ssd-shared/mimiciv-ecg-echonotes/physionet.org/files/mimic-iv-ecg/1.0/processed/machine_measurements_abnormality_v2.csv",
        npy_file_dir="",
        split="Train",
        from_numpy=True,
        ecg_transform=None,
        range=[0, 1, 2],
    ):
        self.from_numpy = from_numpy
        self.range = range
        self.meta_data = (
            pd.read_csv(metadata_dir)
            # .sample(frac=1, random_state=42)
            # .reset_index(drop=True)
        )
        # TODO: Determining whether we need to iloc a subset
        # self.meta_data = self.meta_data.iloc[:55000]
        print(self.meta_data)

        if self.from_numpy:
            self.npy_data = np.load(npy_file_dir)

            self.meta_data = self.meta_data[self.meta_data["data_warning"] == 0]
            self.meta_data = self.meta_data[self.meta_data["valid_bit"] == 1]
            self.meta_data = self.meta_data[
                self.meta_data["abnormality_label"] != "other"
            ]

            # self.meta_data = self.meta_data[
            #     (self.meta_data["k_val"] < 12) & (self.meta_data["k_val"] > 0.5)
            # ]
            # self.meta_data = self.meta_data[self.meta_data["k_class"].isin(self.range)]

            if split == "Train":
                self.meta_data = self.meta_data.iloc[: int(0.8 * len(self.meta_data))]
                self.npy_data = self.npy_data[self.meta_data.index]
                self.meta_data = self.meta_data.reset_index(drop=True)

            else:
                self.meta_data = self.meta_data.iloc[int(0.8 * len(self.meta_data)) :]
                self.npy_data = self.npy_data[self.meta_data.index]
                self.meta_data = self.meta_data.reset_index(drop=True)

        else:
            self.meta_data = self.meta_data[
                self.meta_data["data_warning"] == 0
            ].reset_index(drop=True)
            self.meta_data = self.meta_data[
                self.meta_data["valid_bit"] == 1
            ].reset_index(drop=True)
            self.meta_data = self.meta_data[
                self.meta_data["abnormality_label"] != "other"
            ].reset_index(drop=True)
            # self.meta_data = self.meta_data[
            #     (self.meta_data["k_val"] < 12) & (self.meta_data["k_val"] > 0.5)
            # ].reset_index(drop=True)

            # self.meta_data = self.meta_data.iloc[:1000].reset_index(drop=True)

            # First 80 train, last 20 test
            # TODO: Simutanously loading both train and test data
            if split == "Train":
                self.meta_data = self.meta_data.iloc[
                    : int(0.8 * len(self.meta_data))
                ].reset_index(drop=True)
                # self.meta_data = self.meta_data[
                #     self.meta_data["k_class"].isin(self.range)
                # ].reset_index(drop=True)

            else:
                self.meta_data = self.meta_data.iloc[
                    int(0.8 * len(self.meta_data)) :
                ].reset_index(drop=True)
                # self.meta_data = self.meta_data[
                #     self.meta_data["k_class"].isin(self.range)
                # ].reset_index(drop=True)

        # self.meta_data["normalized_k"] = (
        #     np.array(self.meta_data["k_val"]) - min(self.meta_data["k_val"])
        # ) / (max(self.meta_data["k_val"]) - min(self.meta_data["k_val"]))
        # self.meta_data = self.meta_data[self.meta_data['k_class'] == 2].reset_index(drop = True)

        # self.meta_data = self.meta_data.iloc[:1000]

        # if flags.shuffle_data:
        #     shuffled_indices = np.arange(len(X_data))
        #     np.random.shuffle(shuffled_indices)
        #     X_data = X_data[shuffled_indices]
        #     y_data = y_data[shuffled_indices]
        #     print(pretty_progress_bar("shuffling data from train/test split"))
        # # Getting to train test split
        # x_train = X_data[0 : int(len(X_data) * flags.training_size)]
        # y_train = y_data[0 : int(len(y_data) * flags.training_size)]
        # x_test = X_data[int(len(X_data) * flags.training_size) :]
        # y_test = y_data[int(len(y_data) * flags.training_size) :]

        # self.meta_data = self.meta_data.iloc[:1000]
        # gender_dict = {'M' : 1, 'F' : 0}
        # self.meta_data = self.meta_data.replace({"Gender": gender_dict})
        self.ecg = [None] * len(self.meta_data)
        # self.label = [None] * len(self.meta_data)
        # self.gender = [None] * len(self.meta_data)
        self.subject_id = [None] * len(self.meta_data)
        self.study_id = [None] * len(self.meta_data)
        self.abnormality_label = [None] * len(self.meta_data)
        self.root = root
        self.get_data()

    def get_data(self):
        for idx in tqdm(range(len(self))):
            # for idx in range(5):
            if self.from_numpy:
                self.ecg[idx] = self.npy_data[idx]
            else:
                self.ecg[idx] = np.load(
                    # f"{self.meta_data.iloc[idx]['segment_path']}{self.meta_data.iloc[idx]['segment_path'][-9:-1]}_ECG_II.npy"
                    f"/ssd-shared/mimiciv-ecg-echonotes/physionet.org/files/mimic-iv-ecg/1.0/{self.meta_data.iloc[idx]['path']}_ECG_II.npy"
                )
            # self.ecg[idx] = np.array( (self.ecg[idx]-min(self.ecg[idx]))/(max(self.ecg[idx])- min(self.ecg[idx])))
            if not np.isfinite(self.ecg[idx]).all():
                print("here")
            #     continue
            #     self.ecg[idx] = np.zeros(1)

            # print(self.ecg[idx])
            # print(self.ecg[idx] is None)

            # self.gender[idx] = self.meta_data.iloc[idx]['Gender']
            self.subject_id[idx] = self.meta_data.iloc[idx]["subject_id"]
            self.study_id[idx] = self.meta_data.iloc[idx]["study_id"]

            # self.k[idx] = self.meta_data.iloc[idx]["normalized_k"]
            self.abnormality_label[idx] = self.meta_data.iloc[idx]["abnormality_label"]

            # if self.meta_data.iloc[idx]['Gender']=='M':
            #     self.gender[idx] = 1
            # else:
            #     self.gender[idx] = 0

    def __getitem__(self, index):
        # if self.ecg[index] is None:
        #     return
        # if self.ecg[index] is None:
        #     return
        # ecg = np.array(self.ecg[index]).reshape(1,len(self.ecg[index]))
        # ecg = self.ecg[index]

        # ecg = np.array(
        #     (self.ecg[index] - min(self.ecg[index]))
        #     / (max(self.ecg[index]) - min(self.ecg[index]))
        # )
        ecg = signal.resample(self.ecg[index], 128).reshape(1, 128)
        # ecg = self.ecg[index].reshape(1, 5000)

        # if not np.isfinite(ecg).all():
        #     print("here")
        #     return
        # ecg = np.zeros(len(ecg))
        # print("INF!")
        # print(max(self.ecg[index])- min(self.ecg[index]))

        # if np.isnan(ecg).all():
        #     print("NAN!")
        # ecg = self.ecg[index]
        subject_id = self.subject_id[index]
        k_class = self.study_id[index]
        abnormality_label = self.abnormality_label[index]
        if self.abnormality_label[index] == "sinus_rhythm":
            abnormality_label = 0
        elif self.abnormality_label[index] == "sinus_bradycardia":
            abnormality_label = 1
        elif self.abnormality_label[index] == "sinus_tachycardia":
            abnormality_label = 2
        elif self.abnormality_label[index] == "atrial_fibrillation":
            abnormality_label = 3
        else:
            print("here")
            abnormality_label = 0
        # gender = self.gender[index]

        return (ecg, abnormality_label)

    def collate(self, batch):
        (
            ecg,
            # gender,
            abnormality_label,
        ) = zip(*batch)

        ecg = torch.stack(ecg, dim=0)
        # k = torch.stack(k, dim=0)
        return (ecg, abnormality_label)

    def __len__(self):
        return len(self.meta_data)
