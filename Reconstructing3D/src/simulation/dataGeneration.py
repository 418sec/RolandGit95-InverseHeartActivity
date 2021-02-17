import numpy as np
import matplotlib.pyplot as plt 
#import sys
import time, json
#from sklearn.preprocessing import MinMaxScaler
#from pyevtk.hl import gridToVTK

from BarkleySimulation import BarkleySimluation3D

# %%

def np_to_vtkFrame(X, x=120, y=120, z=120, save_name='X'):
    x, y, z = np.arange(0, x+1), np.arange(0, y+1), np.arange(0, z+1)
    
    #gridToVTK(save_name, x, y, z, cellData = {'u': X[:,:,:].astype(np.float64)})
    
def simulateData(num_sims = 1, name='X', init_steps=512, considered_range=1024, dt=32, show_plots=False,
                 a=1, b=0.15, epsilon=0.02, deltaT=0.01, deltaX=0.1, D=0.02, sim_range=2048, target_dir='', chaos=True):

    
#    mms = MinMaxScaler(feature_range=(-127,128))
    
    time_id = str(int(time.time()*10**6))[8:]
    savename = name + time_id
    print('Speichere Sim unter', savename)
    
    s=BarkleySimluation3D(a=a, b=b, epsilon=epsilon, deltaT=deltaT, deltaX=deltaX, D=D)
    s.initialize_random(size=128, n_boxes=12)#42)  

    # initialise steps
    for i in range(init_steps):
        if i%32==0:
            print(i)
            if show_plots:
                plt.imshow(s.u[:,:,0]), plt.show()
            
        s.explicit_step(chaotic=chaos)
                
    # main simulation
        
    number = 0
    for i in range(sim_range):
        s.explicit_step()
        if i%64==0:
            print(i)
            if show_plots:
                plt.imshow(s.u[:,:,0], vmin=0, vmax=1)
                plt.show()
                
        sub_sim_num = 0
        
        if i%256==0:
            print(f'Start recording the next {considered_range} steps')
            x = []
            for j in range(considered_range):
                s.explicit_step()
                if j%dt==0:
                    x.append(s.u)
                    
                    if len(x)>=128:
                        save_data = np.array(x) * 255 -127
                        np.save(target_dir + savename + f'{sub_sim_num:03}', save_data.astype(np.int8))  
                        sub_sim_num += 1
                        del x, save_data
                        x = []
                        
                if j%64==0:
                    print(i)
                    if show_plots:
                        plt.imshow(s.u[:,:,0], vmin=0, vmax=1)
                        plt.show()
                        
            save_data = np.array(x) * 255 -127
            #np.save(target_dir + savename + str(number), save_data.astype(np.int8))
            
            del x, save_data
            
            number += 1 

# %%
if __name__=='__main__':   
    with open('metadata.json', 'r') as f:
        kwargs = json.load(f)
    
    dir_chaos_target = kwargs['paths']['raw']['chaotic']
    dir_concentric_target = kwargs['paths']['raw']['concentric']

    #kwargs_chaos = {'a':0.94, 'b':0.05, 'epsilon':0.02, 'deltaT':0.01, 'deltaX':0.1, 'D':0.02,
    #                'num_sims':16, 'name':'X', 'init_steps':1024, 'show_plots':True, 'sim_range':1024*3,
    #                'considered_range':512, 'dt':16, 'chaos':True,
    #                'target_dir':dir_chaos_target}
    
    kwargs_chaos = {'a':1, 'b':0.15, 'epsilon':0.02, 'deltaT':0.01, 'deltaX':0.1, 'D':0.02,
                    'num_sims':16, 'name':'X', 'init_steps':2048, 'show_plots':1, 'sim_range':1024*3,
                    'considered_range':512, 'dt':16, 'chaos':True,
                    'target_dir':dir_chaos_target}
    
    kwargs_concentric = {'a':0.6, 'b':0.01, 'epsilon':0.02, 'deltaT':0.01, 'deltaX':0.1, 'D':0.02,
                         'num_sims':16, 'name':'X', 'init_steps':1024, 'show_plots':True, 'sim_range':1024,
                         'considered_range':512, 'dt':16, 'chaos':False,
                         'target_dir':dir_concentric_target}   

    kwargs_concentric_vis = {'a':0.6, 'b':0.01, 'epsilon':0.02, 'deltaT':0.01, 'deltaX':0.1, 'D':0.02,
                             'num_sims':1, 'name':'visualize_concentric', 'init_steps':1024, 'show_plots':True, 'sim_range':1024,
                             'considered_range':2048*16, 'dt':4, 'chaos':False,
                             'target_dir':''}   

    kwargs_chaos_vis = {'a':1, 'b':0.15, 'epsilon':0.02, 'deltaT':0.01, 'deltaX':0.1, 'D':0.02,
                             'num_sims':1, 'name':'visualize_chaotic', 'init_steps':2048, 'show_plots':False, 'sim_range':1024,
                             'considered_range':2048*16, 'dt':4, 'chaos':False,
                             'target_dir':''}   
    
    for _ in range(1):
        #simulateData(**kwargs_chaos)
        #simulateData(**kwargs_concentric)
        #kwargs['config']['chaotic']['show_plots'] = True
        #simulateData(**kwargs['config']['chaotic'])
        #simulateData(**kwargs_concentric_vis)
        simulateData(**kwargs_chaos_vis)

        
    #s = simulatePlotData(init_steps=512, num_steps=4096)
    
    