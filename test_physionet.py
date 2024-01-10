"""
test on physionet data

Shenda Hong, Nov 2019
"""

import numpy as np
import pandas as pd
from collections import Counter
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import OneHotEncoder

from physionet_utils import (
    read_data_physionet_2,
    read_data_physionet_4,
    preprocess_physionet,
)
from resnet1d import ResNet1D, MyDataset
from dataloader import ECGDataset

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torcheval.metrics import MulticlassAUROC
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
import wandb
from sklearn import metrics


# from torchsummary import summary

if __name__ == "__main__":
    is_debug = False

    batch_size = 512
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

    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=64,
    #     shuffle=False,
    # )
    # test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=64,
    #     shuffle=False,
    # )
    wandb.init(
        project="MIMICIV_ECG_abnormality",
        reinit=True,
        # tags=[flags.wandb_tag, f"{flags.sel_subject}_{flags.subject}"],
        tags=["resampled_rate = 256", "bs = 512", "base_filters=64"],  # {args.batch_size}"],
        name=f"supervised resnet",
    )
    # make data
    # preprocess_physionet() ## run this if you have no preprocessed data yet
    # X_train, X_test, Y_train, Y_test, pid_test = read_data_physionet_4()
    # print(X_train.shape, Y_train.shape)
    # dataset = MyDataset(X_train, Y_train)
    # dataset_test = MyDataset(X_test, Y_test)
    dataloader = DataLoader(train_dataset, batch_size=batch_size)
    dataloader_test = DataLoader(test_dataset, batch_size=batch_size, drop_last=False)

    # make model
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device_str = "cuda"
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    kernel_size = 16
    stride = 2
    n_block = 48
    downsample_gap = 6
    increasefilter_gap = 12
    model = ResNet1D(
        in_channels=1,
        base_filters=64,  # 64 for ResNet1D, 352 for ResNeXt1D
        kernel_size=kernel_size,
        stride=stride,
        groups=32,
        n_block=n_block,
        n_classes=4,
        downsample_gap=downsample_gap,
        increasefilter_gap=increasefilter_gap,
        use_do=True,
    )
    model.double().to(device)

    # summary(model, (X_train.shape[1], X_train.shape[2]), device=device_str)
    # exit()

    # train and test
    model.verbose = False
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=10
    )
    loss_func = torch.nn.CrossEntropyLoss()

    n_epoch = 50
    step = 0

    for _ in tqdm(range(n_epoch), desc="epoch", leave=False):
        # train
        model.train()
        loss_all = 0
        acc_all = 0
        prog_iter = tqdm(dataloader, desc="Training", leave=False)
        for batch_idx, batch in enumerate(prog_iter):
            input_x, input_y = tuple(t.to(device) for t in batch)
            pred = model(input_x)
            loss = loss_func(pred, input_y)
            acc = (torch.argmax(pred, 1) == input_y).cpu().numpy().mean()
            pred_label = torch.argmax(pred, 1).cpu().numpy()
            print("plotting...")
            print(confusion_matrix(input_y.cpu().numpy(), pred_label))
            
            # enc = OneHotEncoder()
            # labels = input_y.detach().cpu().numpy().reshape(1, len(input_y))
            # print(labels)
            # enc.fit(labels)  
            
            # s = pd.Series(list(labels.reshape(1, len(input_y))))
            # print("s", s)
            # print("khhh",np.array(pd.get_dummies(s)))
            metric = MulticlassAUROC(num_classes=4, average='macro')
            metric.update(pred, input_y)
            print("AUC ROC", metric.compute())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            loss_all += loss.item()
            acc_all += acc * 100
            # writer.add_scalar("Loss/train", loss.item(), step)
            print(f"Loss = {loss.item()} accuracy = {acc*100} AUCROC = {metric.compute()} at step = {step}")
            wandb.log(
                {
                    "loss per iteration": loss.item(),
                    "accuracy per iteration": acc * 100,
                    # "AUC ROC class 1 per iteration": metric.compute()[0],
                    # "AUC ROC class 2 per iteration": metric.compute()[1],
                    # "AUC ROC class 3 per iteration": metric.compute()[2],
                    # "AUC ROC class 4 per iteration": metric.compute()[3],
                    "Macro AUC ROC iteration": metric.compute(),
                }
            )
            if is_debug:
                break

        scheduler.step(_)
        wandb.log(
            {
                "loss per epoch": loss_all / batch_idx,
                "accuracy per epoch": acc_all / batch_idx,
            }
        )

        # test

        model.eval()
        prog_iter_test = tqdm(dataloader_test, desc="Testing", leave=False)
        all_pred_prob = []
        all_acc = 0
        all_auc = np.array([0.0,0.0,0.0,0.0])
        with torch.no_grad():
            for batch_idx, batch in enumerate(prog_iter_test):
                input_x, input_y = tuple(t.to(device) for t in batch)
                pred = model(input_x)
                all_acc += (torch.argmax(pred, 1) == input_y).cpu().numpy().mean() * 100
                all_pred_prob.append(pred.cpu().data.numpy())
                metric = MulticlassAUROC(num_classes=4, average='macro')
                metric.update(pred, input_y)
                all_auc += metric.compute().cpu().numpy()
                print("AUC ROC", metric.compute())
        all_pred_prob = np.concatenate(all_pred_prob)
        all_pred = np.argmax(all_pred_prob, axis=1)

        ## vote most common
        final_pred = []
        final_gt = []
        # for i_pid in np.unique(pid_test):
        #     tmp_pred = all_pred[pid_test == i_pid]
        #     tmp_gt = Y_test[pid_test == i_pid]
        #     final_pred.append(Counter(tmp_pred).most_common(1)[0][0])
        #     final_gt.append(Counter(tmp_gt).most_common(1)[0][0])

        print("test_accuracy = ", all_acc / batch_idx)
        wandb.log(
            {
                "test accuracy per epoch": all_acc / batch_idx,
                # "test auc class 1 per epoch": all_auc[0] / batch_idx,
                # "test auc class 2 per epoch": all_auc[1] / batch_idx,
                # "test auc class 3 per epoch": all_auc[2] / batch_idx,
                # "test auc class 4 per epoch": all_auc[3] / batch_idx,
                "test auc macro per epoch": all_auc / batch_idx,
            }
        )
    ## classification report
    # tmp_report = classification_report(final_gt, final_pred, output_dict=True)
    # print(confusion_matrix(final_gt, final_pred))
    # f1_score = (
    #     tmp_report["0"]["f1-score"]
    #     + tmp_report["1"]["f1-score"]
    #     + tmp_report["2"]["f1-score"]
    #     + tmp_report["3"]["f1-score"]
    #     + tmp_report["4"]["f1-score"]
    # ) / 4

    # writer.add_scalar("F1/f1_score", f1_score, _)
    # writer.add_scalar("F1/label_0", tmp_report["0"]["f1-score"], _)
    # writer.add_scalar("F1/label_1", tmp_report["1"]["f1-score"], _)
    # writer.add_scalar("F1/label_2", tmp_report["2"]["f1-score"], _)
    # writer.add_scalar("F1/label_3", tmp_report["3"]["f1-score"], _)
