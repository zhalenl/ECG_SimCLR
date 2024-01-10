import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def save_model(model, optimizer, scheduler, current_epoch, name):
    out = os.path.join('./saved_models/',name.format(current_epoch))

    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict':scheduler.state_dict()}, out)
    
# trimages = images[:40000]
# valimages = images[40000:]
# trlabels = labels[:40000]
# vallabels = labels[40000:]


def plot_features(model, vdl, epoch, num_classes, num_feats, batch_size):
    preds = np.array([]).reshape((0,1))
    gt = np.array([]).reshape((0,1))
    vallabels = np.array(0)
    feats = np.array([]).reshape((0,num_feats))
    model.eval()
    plt.figure()
    with torch.no_grad():
        for x1,_, label in vdl:
            x1 = x1.to("cuda") 
            out = model(x1)
            out = out.cpu().data.numpy()#.reshape((1,-1))
            feats = np.append(feats,out,axis = 0)
            vallabels = np.concatenate(
                (vallabels, label.detach().cpu().numpy()), axis=None
            )        
    tsne = TSNE(n_components = 2, perplexity = 50)
    x_feats = tsne.fit_transform(feats)
    num_samples = int(batch_size*(vallabels.shape[0]//batch_size))#(len(val_df)
    
    for i in range(num_classes):
        plt.scatter(x_feats[vallabels[:num_samples]==i,1],x_feats[vallabels[:num_samples]==i,0])
    
    plt.legend([str(i) for i in range(num_classes)])
    plt.show()
    plt.savefig(f"./FIGS/embedding_{epoch}.jpeg")