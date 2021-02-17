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

    #data_chaotic = getRawData('../../data/chaotic/raw/X134201610.npy')
    #data_concentric = getRawData('../../data/concentric/raw/X839049931.npy')
    data_chaotic = getRawData('/home/roland/Projekte/Masterthesis/Reconstructing3D/src/simulation/vis.npy')

    folder = '/home/roland/Projekte/Masterthesis/Reconstructing3D/data/visualization/chaotic/sim1/'
    for i in range(len(data_chaotic)):
        gridToVTK(f"{folder}frame{i:04}", x, y, z, cellData = {'u': data_chaotic[i]})
    #gridToVTK("data_concentric", x, y, z, cellData = {'u': data_concentric[-1]})

    
    
    
    
    
    
    
    
    
    
    