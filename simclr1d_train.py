#importing the libraries 

import numpy as np
import pandas as pd
import shutil, time, os, requests, random, copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dataloader_contrastive import ECGDataset
from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt
# %matplotlib inline

from sklearn.manifold import TSNE
import wandb
from simclr_model import LARS, SimCLR_Loss, PreModel
from utils import save_model, plot_features

def set_seed(seed = 16):
    np.random.seed(seed)
    torch.manual_seed(seed)


batch_size = 256

# if is_debug:
#     writer = SummaryWriter('/nethome/shong375/log/resnet1d/challenge2017/debug')
# else:
#     writer = SummaryWriter('/nethome/shong375/log/resnext1d/challenge2017/layer98')
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

train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True)


os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device_str = "cuda"
device = torch.device(device_str if torch.cuda.is_available() else "cpu")

# MODEL INITILIZER
model = PreModel().to(device).double()

#OPTMIZER
optimizer = LARS(
    [params for params in model.parameters() if params.requires_grad],
    lr=0.2,
    weight_decay=1e-6,
    exclude_from_weight_decay=["batch_normalization", "bias"],
)

# "decay the learning rate with the cosine decay schedule without restarts"
#SCHEDULER OR LINEAR EWARMUP
warmupscheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch : (epoch+1)/10.0, verbose = True)

#SCHEDULER FOR COSINE DECAY
mainscheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 500, eta_min=0.05, last_epoch=-1, verbose = True)

#LOSS FUNCTION
criterion = SimCLR_Loss(batch_size = batch_size, temperature = 0.5)

wandb.init(
    project="MIMICIV_ECG_abnormality",
    reinit=True,
    # tags=[flags.wandb_tag, f"{flags.sel_subject}_{flags.subject}"],
    tags=["resampled_rate = 1000", "bs = 256", "base_filters=64", "data_percentage = 1p", "resnet = 1024", "temperature = 0.1","condition = normal"],  # {args.batch_size}"],
    name=f"selfsupervised resnet+simclr",
)

nr = 0
current_epoch = 0
epochs = 100
tr_loss = []
val_loss = []

for epoch in range(epochs):
        
    print(f"Epoch [{epoch}/{epochs}]\t")
    stime = time.time()

    model.train()
    tr_loss_epoch = 0
    
    for step, (x_i, x_j, _) in enumerate(train_loader):

        x_i = x_i.to(device)
        x_j = x_j.to(device)
        # x_i = x_i.reshape(x_i.shape[1], x_i.shape[0], x_i.shape[2])
        # x_j = x_j.reshape(x_j.shape[1], x_j.shape[0], x_j.shape[2])

        optimizer.zero_grad()
        # x_i = x_i.squeeze().to('cuda:0').float()
        # x_j = x_j.squeeze().to('cuda:0').float()

        # positive pair, with encoding
        z_i = model(x_i)
        z_j = model(x_j)

        loss = criterion(z_i, z_j)
        loss.backward()

        optimizer.step()
        
        if nr == 0 and step % 50 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {round(loss.item(), 5)}")
            wandb.log(
                {
                    # "training step": step/len(train_loader),
                    "training Loss": round(loss.item(), 5),
                }
            )

        tr_loss_epoch += loss.item()

    if nr == 0 and epoch < 10:
        warmupscheduler.step()
    if nr == 0 and epoch >= 10:
        mainscheduler.step()
    
    lr = optimizer.param_groups[0]["lr"]

    if nr == 0 and (epoch+1) % 50 == 0:
        save_model(model, optimizer, mainscheduler, current_epoch,f"SimCLR_CIFAR10_RN50_P128_LR0P2_LWup10_Cos500_T0p5_B128_checkpoint_{epoch}_260621.pt")

    model.eval()
    with torch.no_grad():
        val_loss_epoch = 0
        for step, (x_i, x_j, _) in enumerate(test_loader):
        

          x_i = x_i.to(device)
          x_j = x_j.to(device)

          # positive pair, with encoding
          z_i = model(x_i)
          z_j = model(x_j)

          loss = criterion(z_i, z_j)

          if nr == 0 and step % 50 == 0:
              print(f"Step [{step}/{len(test_loader)}]\t Loss: {round(loss.item(),5)}")

          val_loss_epoch += loss.item()

    if nr == 0:
        tr_loss.append(tr_loss_epoch / len(train_loader))
        val_loss.append(val_loss_epoch / len(test_loader))
        print(f"Epoch [{epoch}/{epochs}]\t Training Loss: {tr_loss_epoch / len(train_loader)}\t lr: {round(lr, 5)}")
        print(f"Epoch [{epoch}/{epochs}]\t Validation Loss: {val_loss_epoch / len(test_loader)}\t lr: {round(lr, 5)}")
        wandb.log(
            {
                "Epoch": epoch,
                "Training Loss": tr_loss_epoch / len(train_loader),
                "Validation Loss": val_loss_epoch / len(test_loader),
                "lr": round(lr, 5),
            }
        )
        current_epoch += 1

    # dg.on_epoch_end()

    time_taken = (time.time()-stime)/60
    print(f"Epoch [{epoch}/{epochs}]\t Time Taken: {time_taken} minutes")

    # if (epoch+1)%10==0:
    plot_features(model.pretrained, test_loader,epoch, 4, 1024, batch_size)

save_model(model, optimizer, mainscheduler, current_epoch, f"SimCLR_MIMICIV_RN50_P128_LR0P2_LWup10_Cos500_T0p5_B128_checkpoint_{epoch}_260621.pt")

