from dataloader_contrastive import ECGDataset
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# metadata_dir = (
#     f"/ssd-shared/mimiciv-ecg-echonotes/physionet.org/files/mimic-iv-ecg/1.0/processed/machine_measurements_abnormality_v2.csv",
# )

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

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
vallabels = np.array(0)
print("vallabels.shape", vallabels.shape, vallabels)

feats = np.array([])#.reshape((0,164))
print("feats.shape", feats.shape)

for i, (ecg1, ecg2, abnormality_class) in enumerate(train_loader):
    print(i)
    print(abnormality_class.shape)
    print(abnormality_class[0])
    print("*"* 100)





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
