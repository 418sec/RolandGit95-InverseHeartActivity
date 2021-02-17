import numpy as np
import matplotlib.pyplot as plt
import sys

class BarkleySimluation3D:
    def __init__(self, a=0.6, b=0.01, epsilon=0.02, deltaT=0.01, deltaX=0.1, D=0.02, boundary_mode='noflux'):
        self.a = a
        self.b = b
        self.epsilon = epsilon
        self.boundary_mode = boundary_mode # Neumann
        self.deltaT = deltaT
        self.deltaX = deltaX
        self.h = D/self.deltaX**2
    
    def initialize_random(self, n_boxes=(20,20,20), size=(120,120,120), seed=None):
        np.random.seed(seed)

        tmp = np.random.rand(*n_boxes)
        
        rpt = size[0]//n_boxes[0], size[1]//n_boxes[1], size[2]//n_boxes[2]
        
        tmp = np.repeat(tmp, np.ones(n_boxes[0], dtype=int)*rpt[0], axis=0)
        tmp = np.repeat(tmp, np.ones(n_boxes[1], dtype=int)*rpt[1], axis=1)
        tmp = np.repeat(tmp, np.ones(n_boxes[2], dtype=int)*rpt[2], axis=2)
        
        self.u = tmp
        
        self.v = self.u.copy()
        self.v[self.v<0.4] = 0.0
        self.v[self.v>0.4] = 1.0
        
    def initialize_one_spiral(self, Nx, Ny, Nz):
        self.u = np.zeros((Nx, Ny, Nz))
        self.v = np.zeros((Nx, Ny, Nz))

        #for one spiral
        for i in range(Nx):
            for j in range(Ny):
                if i >= Nx//2:
                    self.u[i,j,:] = 1.0
                if j >= Ny//2:
                    self.v[i,j,:] = self.a/2.0
                    
                    
    def set_boundaries(self, oldFields):
        if self.boundary_mode == "noflux":
            for (field, oldField) in zip((self.u, self.v), oldFields):
                field[:,:,0] = oldField[:,:,1]
                field[:,:,-1] = oldField[:,:,-2]
                
                field[:,0,:] = oldField[:,1,:]
                field[:,-1,:] = oldField[:,-2,:]
                
                field[0,:,:] = oldField[1,:,:]
                field[-1,:,:] = oldField[-2,:,:]
        
    def explicit_step(self, chaotic=True):
        uOld = self.u.copy()
        vOld = self.v.copy()

        f = 1/self.epsilon * self.u * (1 - self.u) * (self.u - (self.v+self.b)/self.a)

        #f = 1/self.epsilon * self.u * (1 - self.u) * (self.u - (self.v+self.b)/self.a)
        
        laplace = -6*self.u.copy()

        laplace += np.roll(self.u, +1, axis=0)
        laplace += np.roll(self.u, -1, axis=0)
        laplace += np.roll(self.u, +1, axis=1)
        laplace += np.roll(self.u, -1, axis=1)
        laplace += np.roll(self.u, +1, axis=2)
        laplace += np.roll(self.u, -1, axis=2)

        self.u = self.u + self.deltaT * (f + self.h * laplace)

        if chaotic:
            self.v = self.v + self.deltaT * (np.power(uOld, 3) - self.v)
        else:
            self.v = self.v + self.deltaT * (np.power(uOld, 1) - self.v)

        self.set_boundaries((uOld, vOld))