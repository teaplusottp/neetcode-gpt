import torch
import torch.nn as nn
from typing import List
import numpy as np

class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        # Forward pass through the model.
        # After each ReLU layer, compute the fraction of neurons that are dead.
        # A neuron is dead if it outputs 0 for ALL samples in the batch.
        # Return a list of dead fractions (one per ReLU layer), rounded to 4 decimals.
        res=[]
        # print(x.shape)
        for layer in model:
            x=layer(x)
            if (isinstance(layer,nn.ReLU)):
                fraction=((x==0).all(dim=0)).sum()/len(x[0])
                print(fraction)
                res.append(np.round(fraction,4))
        return res

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        # Given dead fractions per ReLU layer, suggest a fix.
        # Check in this order:
        # 1. 'use_leaky_relu' if any layer has dead fraction > 0.5
        # 2. 'reinitialize' if the first layer has dead fraction > 0.3
        # 3. 'reduce_learning_rate' if dead fraction strictly increases
        #    with depth AND the last layer's fraction > 0.1
        # 4. 'healthy' if max dead fraction < 0.1
        # 5. 'healthy' otherwise
        prev=0
        increase=True
        M=0
        for index,item in enumerate(dead_fractions):
            if(item>0.5):
                return "use_leaky_relu"
            if(item>0.3 and index==0):
                return "reinitialize"
            if(prev>item):
                increase=False
            M=max(item,M)
            prev=item
        if(increase and dead_fractions[-1]>0.1):
            return "reduce_learning_rate"
        
        return "healthy"
