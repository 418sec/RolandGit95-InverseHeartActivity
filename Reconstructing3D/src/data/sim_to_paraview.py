import os
import numpy as np
from pyevtk.hl import gridToVTK

dirname = os.path.dirname(os.path.realpath(__file__)) + '/'
os.chdir(dirname)

# %% Processing Function

def getRawData(file):
    data = np.load(file)
    return data
# %% main

if __name__=='__main__':
    
    x = np.arange(0, 121)
    y = np.arange(0, 121)
    z = np.arange(0, 121)

    data_chaotic = getRawData('../../data/chaotic/raw/X134201610.npy')
    #data_concentric = getRawData('../../data/concentric/raw/X839049931.npy')
    
    gridToVTK("data_chaotic", x, y, z, cellData = {'u': data_chaotic[-1]})
    #gridToVTK("data_concentric", x, y, z, cellData = {'u': data_concentric[-1]})

    
    
    
    
    
    
    
    
    
    
    