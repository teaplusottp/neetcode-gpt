import torch
import torch.nn as nn
import math
from typing import List
import numpy as np

class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std=math.sqrt(2/(fan_in+fan_out))
        W = torch.randn(fan_out,fan_in)
        return (W * std).round(decimals=4).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std=math.sqrt(2/fan_in)
        W = torch.randn(fan_out,fan_in)
        return (W * std).round(decimals=4).tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        torch.manual_seed(0)
        
        # 1. Khởi tạo toàn bộ ma trận trọng số (Weights) TRƯỚC
        weights = []
        for i in range(num_layers):
            f_in = input_dim if i == 0 else hidden_dim
            f_out = hidden_dim
            
            if init_type == "xavier":
                std = math.sqrt(2 / (f_in + f_out))
            elif init_type == "kaiming":
                std = math.sqrt(2 / f_in)
            else:
                std = 1.0
            
            W = torch.randn(f_out, f_in) * std
            weights.append(W)
            
        # 2. Khởi tạo random input SAU KHI đã tạo xong weights
        x = torch.randn(1, input_dim)
        activations = []
        
        # 3. Tiến hành Forward pass qua từng layer
        for W in weights:
            x = torch.relu(x @ W.T)
            activations.append(round(x.std().item(), 2))
            
        return activations