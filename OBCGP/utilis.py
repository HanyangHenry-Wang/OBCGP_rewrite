from typing import Any
import torch
from botorch.utils.transforms import unnormalize,normalize
import torch
from botorch.utils.sampling import draw_sobol_samples
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double


class obj_fun:
    def __init__(self,botorch_fun) -> None:
        self.fun = botorch_fun
        self.dim = botorch_fun.dim
        self.bounds = botorch_fun.bounds
   
    
    def __call__(self, x) -> Any:  # x is a numpy
        x = torch.tensor(x)
        x_unnormal = unnormalize(x, self.bounds).reshape(-1,self.dim)  
        res = self.fun(x_unnormal)  

        return np.atleast_2d(res.numpy()).reshape(-1,1)
    



def get_initial_points(bounds,num,device,dtype,seed=0):
    
        train_x = draw_sobol_samples(
        bounds=bounds, n=num, q=1,seed=seed).reshape(num,-1).to(device, dtype=dtype)
        
        return train_x