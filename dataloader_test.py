from dataloader_contrastive import ECGDataset
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pdb


# metadata_dir = (
#     f"/ssd-shared/mimiciv-ecg-echonotes/physionet.org/files/mimic-iv-ecg/1.0/processed/machine_measurements_abnormality_v2.csv",
# )

# Getting the train and test datasets
train_dataset = ECGDataset(
    split="Train",
    # npy_file_dir="/home/grads/z/zhale/MIMIC_ECG/MIMICIV_ECG_Processing/EDA/ecg_all.npy",
    from_numpy=False,
)
test_dataset = ECGDataset(
    split="Test",
    # npy_file_dir="/home/grads/z/zhale/MIMIC_ECG/MIMICIV_ECG_Processing/EDA/ecg_all.npy",
    from_numpy=False,
)
# Loading the train and test datasets into dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=False,
)
# test_loader = DataLoader(
#     test_dataset,
#     batch_size=64,
#     shuffle=False,
# )

# Initializes vallabels (validation labels)
vallabels = np.array(0)
print("vallabels.shape", vallabels.shape, vallabels)
# Initializes feats (collection of features)
feats = np.array([])  # .reshape((0,164))
print("feats.shape", feats.shape)

# Populating vallabels and feats
# TODO: Explain what are ecg1 and ecg2 variables
# TODO: Explain what are abnormality_class variables (there are 64 classes)?
for i, (ecg1, ecg2, abnormality_class) in enumerate(train_loader):
    print(i)
    print(abnormality_class.shape)
    print(abnormality_class[0])
    print("*" * 100)
    pdb.set_trace()
    # x1 = x1.squeeze().to(device = 'cuda:0', dtype = torch.float)
    # out = model(x1)
    # out = out.cpu().data.numpy()#.reshape((1,-1))
    # feats = np.append(feats,out,axis = 0)
    vallabels = np.concatenate(
        (vallabels, abnormality_class.detach().cpu().numpy()), axis=None
    )
    print("vallabels.shape", vallabels.shape, vallabels)
    # plt.figure()
    # print(ecg1[j])
    # print(ecg2[j])
    # plt.plot(ecg1[j][0], label="III")
    # plt.plot(ecg2[j][0], label="II")
    # plt.legend()
    # plt.savefig(f"./test_figs/ecg{j}.jpeg")
    # plt.show()
    # print(abnormality_class)
    # print("*" * 100)
    # if i == 2:
    #     break
