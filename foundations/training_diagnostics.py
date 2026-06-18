import torch
import torch.nn as nn
from typing import List, Dict
import numpy as np

class Solution:
    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        res = []
        for layer in model:
            # Lưu ý: Không tự ý gọi relu(x) vì nó làm thay đổi giá trị x cho lớp tiếp theo
            # Nếu model đã có sẵn layer ReLU thì model đã tự xử lý rồi
            with torch.no_grad():
                x = layer(x) 
                if isinstance(layer, nn.Linear):
                    is_dead = (x <= 0).all(dim=0)
                    dead_fraction = is_dead.float().mean().item()
                    res.append({
                        "mean": round(x.mean().item(), 4),
                        "std": round(x.std().item(), 4),
                        "dead_fraction": round(dead_fraction, 4)
                    })
        return res

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        criterion = nn.MSELoss()
        model.zero_grad()
        y_pred=model(x)
        
        Loss=criterion(y_pred,y)
        Loss.backward()
        
        res=[]
        
        for layer in model:
            if(isinstance(layer,nn.Linear)):

                grad=layer.weight.grad
                res.append({
                    "mean": round(grad.mean().item(), 4),
                    "std": round(grad.std().item(), 4),
                    "norm": round(grad.norm().item(), 4)
                })

        return res


    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        # 1
        for item in activation_stats:
            if item["dead_fraction"] > 0.5:
                return "dead_neurons"
        # 2
        for item in gradient_stats:
            if item["norm"]>1000:
                return "exploding_gradients"
        # 3      
        if gradient_stats[-1]["norm"]<1e-5:
            return "vanishing_gradients"
        # 4         
        for item in activation_stats:
            if item["std"]<0.1:
                return "vanishing_gradients"
            if item["std"]>10:
                return "exploding_gradients"     
        # 5
        return "healthy"
