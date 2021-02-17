#! python

# name
#$ -N c_conc_eval

# execute from current directory
#$ -cwd

# Preserve environment variables
#$ -V

# Provide path to python executable
#$ -S /home/stenger/smaxxhome/anaconda3/envs/torch/bin/python

# Merge error and out
#$ -j yes

# Path for output
#$ -o /home/stenger/smaxxhome/outputs

import sys, os
from tqdm import tqdm
sys.path.append('../../src')
sys.path.append('../../models')
from architecture import modules
from data.datasets import BarkleyDataset
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error as mae
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import cv2
import ot
from scipy.stats import wasserstein_distance
from omegaconf import DictConfig, OmegaConf

def sliced_wasserstein(X, Y, num_proj=256):
    dim = X.shape[1]
    ests = []
    for _ in range(num_proj):
        # sample uniformly from the unit sphere
        dir = np.random.rand(dim)
        dir /= np.linalg.norm(dir)

        # project the data
        X_proj = X @ dir
        Y_proj = Y @ dir

        # compute 1d wasserstein
        ests.append(wasserstein_distance(X_proj, Y_proj))
    return np.mean(ests)

def multiple_wasserstein(Xs, Ys, num_proj=2048*4):
    ests = []
    for X, Y in tqdm(zip(Xs, Ys), total=len(Xs)):
        est = sliced_wasserstein(X, Y)
        ests.append(est)
    print(np.mean(ests))
    return np.mean(ests)

def mae_b(img1, img2):
    return mae(img1.reshape((-1,)), img2.reshape((-1,)))

@torch.no_grad()
def getYTruePredPairs(model, dataset, batch_size=4, depth=2, time_steps=1):
    dataloader = DataLoader(dataset, batch_size)
    
    y_trues, y_preds = [], []
    for i, (X,y) in tqdm(enumerate(dataloader), total=len(dataloader)):
        X = X[:,:time_steps].to(device) # [bs,t,d,120,120]
        y = y[:,:,:depth].cpu().detach().numpy()
        
        y_pred = model(X, max_depth=depth).cpu().detach().numpy()
        
        y_preds.append(y_pred)
        y_trues.append(y)
    y_preds = np.concatenate(y_preds, 0)
    y_trues = np.concatenate(y_trues, 0) 
        
    return y_trues, y_preds

def getLossPerDepth(y_trues, y_preds, criterion=sliced_wasserstein, max_depth=32, num_examples=None): 
    losses = []
    
    for depth in range(max_depth):
        print(f'Depth: {depth}')
        loss = criterion(y_trues[:,0,depth], y_preds[:,0,depth])
        losses.append(loss)
        
    return losses

def getModel(model_architecture, folder=f'../../models/weights/concentric', model_file="CLSTM"):
    data_dir = os.path.abspath(folder) + '/'
    model_architecture.load_state_dict(torch.load(data_dir + model_file, map_location=device), strict=True)
    return model_architecture

def make(cfg, t, d):
    dataset = BarkleyDataset(root=f'../../data/concentric/processed/', train=False, depth=32, time_steps=32)
    
    if cfg.model == "CLSTM":
        model = getModel(nn.DataParallel(modules.CLSTM(1,64, directional=1)),
                        folder=f'../../models/weights/{cfg.mode}',
                        model_file=cfg.model + f'_t{t}_d{d}').to(device)
    elif cfg.model == "STLSTM":
        model = getModel(nn.DataParallel(modules.STLSTM(1,64, directional=1)),
                        folder=f'../../models/weights/{cfg.mode}',
                        model_file=cfg.model + f'_t{t}_d{d}').to(device)
        
    if cfg.emd:
        criterion = multiple_wasserstein
    else:
        criterion = None
        
    return model, dataset, criterion

if __name__=='__main__':
    
    cfg = {
        "mode":"concentric", # "concentric", "chaotic"
        "d_max":32,
        "ts":[32,30,28,25,20,16,8,4,2,1],
        "model": "CLSTM", # "CLSTM", "STLSTM"
        "emd":True
    }
    
    cfg = OmegaConf.create(cfg)

    
    losses_mae = []
    losses_emd = []
    
    with torch.no_grad():
        for t in cfg.ts:
            model, dataset, criterion = make(cfg, t, cfg.d_max)
            y_trues, y_preds = getYTruePredPairs(model, dataset, batch_size=4, depth=cfg.d_max, time_steps=t)
    
            
            loss_mae = getLossPerDepth(y_trues, y_preds, criterion = mae_b, max_depth=cfg.d_max)
            losses_mae.append(loss_mae)
            
            if criterion != None:
                loss_emd = getLossPerDepth(y_trues, y_preds, criterion = multiple_wasserstein, max_depth=cfg.d_max)
            losses_emd.append(loss_emd)

    np.save(f'mae_{cfg.mode}_{cfg.model}.npy', losses_mae)
    np.save(f'emd_{cfg.mode}_{cfg.model}.npy', losses_emd)






