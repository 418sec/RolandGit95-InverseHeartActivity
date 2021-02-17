# execute with qsub evaluation.py

#! python

# name
#$ -N conc_var_T

# execute from current directory
#$ -cwd

# Preserve environment variables
#$ -V

# Provide path to python executable
#$ -S /home/stenger/smaxxhome/anaconda3/envs/torch/bin/python

# Merge error and out
#$ -j yes

# Path for output
#$ -o /home/stenger/smaxxhome/Masterthesis/UnderSurface/outputs

# %%

import sys, os
from tqdm import tqdm
sys.path.append('../../src')
sys.path.append('../../models')

from architecture import modules
from data.datasets import BarkleyDataset
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

import itertools


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Found', torch.cuda.device_count(), 'GPUs')


from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import sklearn.metrics as metrics
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

import cv2

# %% 

def loadModel(model_architecture, folder='../../models/weights/', model_file='model'):
    data_dir = os.path.abspath(folder) + '/'
    model_architecture.load_state_dict(torch.load(data_dir + model_file, map_location=device), strict=True)
    return model_architecture

@torch.no_grad()
def getYTruePredPairs(model, dataset, batch_size=4, depth=2, time_steps=1):
    dataloader = DataLoader(dataset, batch_size)
    
    y_trues, y_preds = [], []
    for i, (X,y) in tqdm(enumerate(dataloader), total=len(dataloader)):
        X = X[:,:time_steps].to(device)
        y = y[:,:,:depth].cpu().detach().numpy()
        
        y_pred = model(X, max_depth=depth).cpu().detach().numpy()
        
        y_preds.append(y_pred)
        y_trues.append(y)
    y_preds = np.concatenate(y_preds, 0)
    y_trues = np.concatenate(y_trues, 0) 
        
    return y_trues, y_preds

# mse per layer
def getLossPerDepth(y_trues, y_preds, criterion=mse, max_depth=32, num_examples=None):    
    return [criterion(y_trues[:,:,depth], y_preds[:,:,depth]) for depth in range(max_depth)]

def ssim_b(img1, img2):
    img1 = np.swapaxes(img1, -3,-1)
    img2 = np.swapaxes(img2, -3,-1)
    ds = len(np.shape(img1))
    #print(img1.shape)
    if ds > 4:
        keep_shape = [img1.shape[i-1] for i in np.flip(range(ds, ds-3,-1))]
        return ssim(img1.reshape((-1, *keep_shape)), img2.reshape((-1, *keep_shape)), multichannel=True)
    else:
        return ssim(img1, img2, multichannel=True)
    
def psnr_b(img1, img2):
    img1 = np.swapaxes(img1, -3,-1)
    img2 = np.swapaxes(img2, -3,-1)
    ds = len(np.shape(img1))
    #print(img1.shape)
    if ds > 4:
        keep_shape = [img1.shape[i-1] for i in np.flip(range(ds, ds-3,-1))]
        return ssim(img1.reshape((-1, *keep_shape)), img2.reshape((-1, *keep_shape)))
    else:
        return psnr(img1, img2)
    
def rmse_b(img1, img2):
    return mse(img1.reshape((-1,)), img2.reshape((-1,)), squared=False)
    
def mae_b(img1, img2):
    return mae(img1.reshape((-1,)), img2.reshape((-1,)))

def psnr_b(img1, img2):
    img1 = np.swapaxes(img1, -3,-1)
    img2 = np.swapaxes(img2, -3,-1)
    ds = len(np.shape(img1))
    #print(img1.shape)
    if ds > 4:
        keep_shape = [img1.shape[i-1] for i in np.flip(range(ds, ds-3,-1))]
        return ssim(img1.reshape((-1, *keep_shape)), img2.reshape((-1, *keep_shape)))

def emd(img1, img2):
    # convert to (?,h*w)
    # get only subsample if image is too big:
    h, w = img1.shape[-2:]
    max_x_off, max_y_off = max(0,w - 60), max(0,h - 60)   
    sigs = []
    s=[60,60]
    _s = np.shape(img1)[:-2]
    for _s_ in _s:
        s.append(_s_)
    #print(list(s))
    img1 = np.swapaxes(np.swapaxes(img1, 0,-2), 1,-1)
    img2 = np.swapaxes(np.swapaxes(img2, 0,-2), 1,-1)
    
    new_img1, new_img2 = np.zeros(s), np.zeros(s)
    for i in range(len(new_img1[0,0,:])):
        x_start, y_start = np.random.randint(0,max_x_off), np.random.randint(0,max_y_off)
        x_end, y_end = min(x_start+60, w), min(y_start+60, h)
        #np.swapaxes(img1, 0,-1)
        new_img1[:,:,i] = img1[y_start:y_end, x_start:x_end,i]
        new_img2[:,:,i] = img2[y_start:y_end, x_start:x_end,i]

    new_img1 = np.swapaxes(new_img1, 0,-1)
    new_img2 = np.swapaxes(new_img2, 0,-1)
    print('Shapes:',new_img1.shape,new_img2.shape)
        
    for img in [new_img1, new_img2]:
        
        h, w = img.shape[-2:]
        #print(h, w)
        
        ds = len(np.shape(img))
        if ds > 2:
            keep_shape = [img.shape[i] for i in np.flip(range(ds, ds-2,-1))-1]
            img = img.reshape(-1, *keep_shape).reshape(-1,h*w)  
            
        sig = np.empty((len(img), h*w, 3))
        #print(sig.shape)
        inds = np.array(list(itertools.product(np.arange(0,h,1), np.arange(0,w,1)))).T

        for n in range(len(img)):
            #print('HO',img[n].shape)
            sig[n] = np.array([img[n], inds[0], inds[1]]).T
        sigs.append(cv2.normalize(sig, None, 1, 0, cv2.NORM_MINMAX, cv2.CV_32FC1))
        
    dists = []
    for i in tqdm(range(len(sigs[0]))):
        dist, _, _= cv2.EMD(sigs[0][i], sigs[1][i], 2)
        dists.append(dist)
        #print(dist)
    print(np.mean(dists))
    return np.mean(dists)

def emd_b(imgs1, imgs2):
    #print(imgs1.shape)
    e = [emd(np.array([img1]), np.array([img2])) for img1, img2 in zip(imgs1[:,0], imgs2[:,0])]
    return np.mean(e)

# %%

def getValidation(data, modelname, ts, d=32):
    dataset = BarkleyDataset(root=f'../../data/{data}/processed/', chaotic=False, train=False,
                             depth=32, time_steps=32)

    
    losses_mae = np.zeros((d, len(ts)))
    losses_mse = np.zeros((d, len(ts)))
    losses_ssim = np.zeros((d, len(ts)))
    
    for j, t in enumerate(ts):
        print('Depth:', d, 'Time-step:', t)
        model = loadModel(nn.DataParallel(modules.CLSTM(1,64, directional=1)),
                          folder=f'../../models/weights/{data}',
                          model_file=f'{modelname}_t{t}_d{32}').to(device)
        y_trues, y_preds = getYTruePredPairs(model, dataset, batch_size=4, depth=d, time_steps=t)
    
        #loss_emd = getLossPerDepth(y_trues, y_preds, criterion=emd_b, max_depth=d)

        loss_mae = getLossPerDepth(y_trues, y_preds, criterion=mae_b, max_depth=d)
        loss_mse = getLossPerDepth(y_trues, y_preds, criterion=rmse_b, max_depth=d)
        loss_ssim = getLossPerDepth(y_trues, y_preds, criterion=ssim_b, max_depth=d)
    
        losses_mae[:,j] = loss_mae
        losses_mse[:,j] = loss_mse
        losses_ssim[:,j] = loss_ssim
        
    losses = np.array([losses_mae, losses_mse, losses_ssim])
    
    np.save(f'losses_{data}_{modelname}.npy', losses)
  
def getEMD(data, modelname, ts, d=32):
    dataset = BarkleyDataset(root=f'../../data/{data}/processed/', chaotic=False, train=False,
                             depth=32, time_steps=32)

    
    losses = np.zeros((d, len(ts)))
    
    for j, t in enumerate(ts):
        print('Depth:', d, 'Time-step:', t)
        model = loadModel(nn.DataParallel(modules.CLSTM(1,64, directional=1)),
                          folder=f'../../models/weights/{data}',
                          model_file=f'{modelname}_t{t}_d{32}').to(device)
        y_trues, y_preds = getYTruePredPairs(model, dataset, batch_size=8, depth=d, time_steps=t)
    
        loss_emd = getLossPerDepth(y_trues, y_preds, criterion=emd_b, max_depth=d)

        losses[:,j] = loss_emd

    np.save(f'emd_{data}_{modelname}.npy', losses)

# %%
    
if __name__=='__main__':
    #getValidation('chaotic', 'CLSTM', [1], d=2)
    #getEMD('chaotic', 'CLSTM', [32,30,28,25,20,16,8,4,2,1])
    #getEMD('concentric', 'CLSTM', [32,30,28,25,20,16,8,4,2,1])
    #getEMD('chaotic', 'STLSTM', [32,30,28,25,20,16,8,4,2,1])
    #getEMD('concentric', 'STLSTM', [32,30,28,25,20,16,8,4,2,1])

    #getValidation('chaotic', 'CLSTM', [32,30,28,25,20,16,8,4,2,1])
    #getValidation('concentric', 'CLSTM', [32,30,28,25,20,16,8,4,2,1])
    #getValidation('chaotic', 'STLSTM', [32,30,28,25,20,16,8,4,2,1])
    #getValidation('concentric', 'STLSTM', [32,30,28,25,20,16,8,4,2,1])

    getValidation('concentric', 'CLSTM', [20,1])




