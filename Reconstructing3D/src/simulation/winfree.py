import numpy as np
import matplotlib.pyplot as plt 
import time
from BarkleySimulation import BarkleySimluation3D

# %%
def long(num_sims = 1, name='vis', init_steps=512, considered_range=1024, dt=32, show_plots=False,
                 a=1, b=0.15, epsilon=0.02, deltaT=0.01, deltaX=0.1, D=0.02, 
                 sim_range=2048, target_dir='', chaos=False):
    
    time_id = str(int(time.time()*10**6))[8:]
    savename = name + time_id
    print('Speichere Sim unter', savename)
    
    s=BarkleySimluation3D(a=a, b=b, epsilon=epsilon, deltaT=deltaT, deltaX=deltaX, D=D)
    s.initialize_random(n_boxes=(12,12,12), size=(120,120,120))#42)  
    #s.initialize_one_spiral(120,120,10)           
    
    number, sub_sim_num = 0, 0
    max_len=512
    x = []
    for i in range(sim_range):
        s.explicit_step()
        
        if i%64==0:
            print(i)                
            plt.imshow(s.u[:,:,0], vmin=0, vmax=1)
            plt.show()
                
        if i%dt==0:
            x.append(s.u)
                    
        if len(x)>=max_len:
            print('save data')
            save_data = np.array(x) * 255 -127
            np.save(target_dir + savename + f'{sub_sim_num:03}', save_data.astype(np.int8))  
            sub_sim_num += 1
            del x, save_data
            x = []
            #np.save(target_dir + savename + str(number), save_data.astype(np.int8))
            number += 1 
# %%
    
def test(num_sims = 1, name='X', init_steps=512, considered_range=1024, dt=32, show_plots=False,
                 a=1, b=0.15, epsilon=0.02, deltaT=0.01, deltaX=0.1, D=0.02, 
                 sim_range=2048, target_dir='', chaos=False):
    
    time_id = str(int(time.time()*10**6))[8:]
    savename = name + time_id
    print('Speichere Sim unter', savename)
    
    s=BarkleySimluation3D(a=a, b=b, epsilon=epsilon, deltaT=deltaT, deltaX=deltaX, D=D)
    s.initialize_random(n_boxes=(12,12,12), size=(120,120,120))#42)  
    #s.initialize_one_spiral(120,120,10)           
    
    
    for i in range(init_steps):
        s.explicit_step()
        
        if i%64==0:
            print(i)
            if show_plots:
                plt.imshow(s.u[:,:,0], vmin=0, vmax=1)
                plt.show()
                
    number, sub_sim_num = 0, 0
    x = []
    for i in range(sim_range):
        s.explicit_step()
        
        if i%64==0:
            print(i)
            if show_plots:
                plt.imshow(s.u[:,:,0], vmin=0, vmax=1)
                plt.show()
                
        
        if i%dt==0:
            x.append(s.u)
                    
        if len(x)>=considered_range//dt:
            print('save data')
            save_data = np.array(x) * 255 -127
            np.save(target_dir + savename + f'{sub_sim_num:03}', save_data.astype(np.int8))  
            sub_sim_num += 1
            del x, save_data
            x = []
            #np.save(target_dir + savename + str(number), save_data.astype(np.int8))

            number += 1 
    


# %%
if __name__=='__main__':   
    dir_target = ''


    kwargs_winfree = {'a':0.75, 'b':0.06, 'epsilon':0.08, 'deltaT':0.01, 'deltaX':0.1, 'D':0.02,
                      'num_sims':16, 'name':'X', 'init_steps':1024, 'show_plots':0, 'sim_range':4196*8,
                      'considered_range':512, 'dt':16, 'chaos':True,
                      'target_dir':dir_target}
    
    kwargs_long = {'a':0.75, 'b':0.06, 'epsilon':0.08, 'deltaT':0.01, 'deltaX':0.1, 'D':0.02,
                      'num_sims':16, 'name':'vis', 'init_steps':1024, 'show_plots':0, 'sim_range':4196*4,
                      'considered_range':512, 'dt':2, 'chaos':True,
                      'target_dir':dir_target}
 
 
    kwargs_long_conc= {'a':0.6, 'b':0.01, 'epsilon':0.02, 'deltaT':0.01, 'deltaX':0.1, 'D':0.02,
                      'num_sims':16, 'name':'vis_conc', 'init_steps':1024, 'show_plots':0, 'sim_range':4196*4,
                      'considered_range':512, 'dt':2, 'chaos':True,
                      'target_dir':dir_target}
    
    for _ in range(1):
        #test(**kwargs_winfree)
        long(**kwargs_long)
        #long(**kwargs_long_conc)


        
    #s = simulatePlotData(init_steps=512, num_steps=4096)
    
    