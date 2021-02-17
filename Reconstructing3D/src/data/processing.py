import glob, re, os, json, time
from tqdm import tqdm
import numpy as np
import torch

dirname = os.path.dirname(os.path.realpath(__file__)) + '/'
os.chdir(dirname)

# %% Processing Function

def getProcessedData(directory, target_directory, start=0, stop=64):
    max_depth = 32
    
    X, y = [],[]
    files = glob.glob(directory + '*')
    
    if stop>len(files):
        stop = len(files)
        
    for file in tqdm(files[start:stop]):
        #print(f'Load file {i+1 + 12*n}/{len(files)}', file, end="\r")
        d = np.load(file)
        #print(np.shape(d))
                
        data = []
        data.append(np.swapaxes(d,1,1)[:,:max_depth,:,:])
        data.append(np.swapaxes(d,1,2)[:,:max_depth,:,:])
        data.append(np.swapaxes(d,1,3)[:,:max_depth,:,:])
                    
        data.append(np.flip(np.swapaxes(d,1,1), axis=1)[:,:max_depth,:,:])
        data.append(np.flip(np.swapaxes(d,1,2), axis=1)[:,:max_depth,:,:])
        data.append(np.flip(np.swapaxes(d,1,3), axis=1)[:,:max_depth,:,:])
                
        for j in range(len(data)):
            k = np.random.randint(0,4)
            data[j] = np.rot90(data[j], k=k, axes=(2,3))
            
        data = np.array(data) 
                
        n_ts = np.shape(data)[1]
                
        _X = data[:,:,:1]
        #_y = data[:,n_ts//2:n_ts//2+1]
        _y = data[:,:1]
                
        X.append(_X)
        y.append(_y)
                        
        del data, _X, _y
    #[print(x.shape) for x in X]
            
    X = np.concatenate(X, axis=0)
    X = np.swapaxes(X, 0,1)
    X = np.array(X)
    
    y = np.concatenate(y, axis=0)
    y = np.swapaxes(y, 0,1)
    y = np.array(y)
    
    
    time_id = str(int(time.time()*10**6))[8:]
    savename = 'X' + time_id
    
    np.save(target_directory + 'X' + time_id, X)
    np.save(target_directory + 'y' + time_id, y)
    
    del X, y
    
# %% Merging after processing
def mergeData(directory):
    filesX = glob.glob(directory + 'X*')
    filesY = glob.glob(directory + 'y*')
    
    filesX.sort()
    filesY.sort()
    
    #print(filesX, filesY)
    
    X = []
    for fileX in filesX:
        _X = np.load(fileX)
        #print(_X.shape)
        X.append(_X)
        
    X = np.concatenate(X, axis=1)
    X = np.array(X)
    np.save(directory + 'X', X)
    #del X
    
    y = []
    for fileY in filesY:
        _y = np.load(fileY)
        #print(_y.shape)
        y.append(_y)
        
    y = np.concatenate(y, axis=1)
    y = np.array(y)
    np.save(directory + 'y', y)
    #del y
    
    return X,y
    
def toPytorch(X,y, target_dir):
    X_train = torch.Tensor(X[:,:-512]).permute(1,0,2,3,4).type(torch.int8)
    y_train = torch.Tensor(y[:,:-512]).permute(1,0,2,3,4).type(torch.int8)
    X_test = torch.Tensor(X[:,-512:]).permute(1,0,2,3,4).type(torch.int8)
    y_test = torch.Tensor(y[:,-512:]).permute(1,0,2,3,4).type(torch.int8)
    
    torch.save((X_train, y_train), f=f'{target_dir}train.pt')
    del X_train, y_train
    
    torch.save((X_test, y_test), f=f'{target_dir}test.pt')

    
    
# %% main

if __name__=='__main__':
    
    #dir_chaos = '../../data/chaotic/raw/'
    #dir_chaos_target = '../../data/chaotic/processed/'
    #dir_chaos_target = '../../data/chaotic/processed/'

    raw_dir = '../../data/winfree/raw/'
    processed_dir = '../../data/winfree/processed/'

    a = np.arange(0,256,32,dtype=np.int)
    for start,stop in zip(a,a+32):
        print(start, stop)
        
        #getProcessedData(raw_dir, processed_dir, start, stop)
    mergeData(processed_dir)
    X, y = mergeData(processed_dir)
    toPytorch(X, y, processed_dir)
    
    
    
    
    
    
    
    
    
    
    
    
    