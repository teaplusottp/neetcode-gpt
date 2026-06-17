import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        
        #setup variable
        e=1e-5
        running_mean = np.array(running_mean).reshape(-1, 1)
        running_var = np.array(running_var).reshape(-1, 1)
        reshape_X = np.array(x, dtype=float).T

        gamma = np.array(gamma).reshape(-1, 1)
        beta  = np.array(beta).reshape(-1, 1)
        
        if(training):
            #mean and var
            mean_X=np.mean(reshape_X,axis=1).reshape(-1,1)
            var_X=np.var(reshape_X,axis=1).reshape(-1,1)

            running_mean=(1-momentum)*running_mean+momentum*mean_X
            running_var=(1-momentum)*running_var+momentum*var_X
        else:
            mean_X=running_mean
            var_X=running_var
        
        x_hat=(reshape_X-mean_X)/np.sqrt(var_X+e)
        y=gamma*x_hat+beta
        y=y.T

        res = (
            np.round(y, 4),
            np.round(running_mean.T[0], 4),
            np.round(running_var.T[0], 4)
        )
        return res

