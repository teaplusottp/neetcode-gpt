import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.deptrai1=nn.Linear(784,512)
        self.deptrai2=nn.ReLU()
        self.deptrai3=nn.Dropout(0.2)
        self.deptrai4=nn.Linear(512,10)
        self.deptrai5=nn.Sigmoid()
        # Architecture: Linear(784, 512) -> ReLU -> Dropout(0.2) -> Linear(512, 10) -> Sigmoid

    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        # images shape: (batch_size, 784)
        # Return the model's prediction to 4 decimal places
        x=self.deptrai1(images)
        x=self.deptrai2(x)
        x=self.deptrai3(x)
        x=self.deptrai4(x)
        x=self.deptrai5(x)

        return x
