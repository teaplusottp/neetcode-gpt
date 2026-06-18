import torch
import torch.nn as nn
from torchtyping import TensorType
import numpy as np

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        # Layers: Embedding(vocabulary_size, 16) -> Linear(16, 1) -> Sigmoid
        self.vocab_size=vocabulary_size
        self.emd=nn.Embedding(self.vocab_size,16)
        self.linear=nn.Linear(16,1)
        self.sigmoid=nn.Sigmoid()
    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # Hint: The embedding layer outputs a B, T, embed_dim tensor
        # but you should average it into a B, embed_dim tensor before using the Linear layer

        # Return a B, 1 tensor and round to 4 decimal places
        res=[]
        for sentence in x:
            vector=[]
            for word in sentence:
                vector.append(self.emd(word))
            
            vector=torch.stack(vector)
            vector=vector.mean(axis=0)
       
            vector=self.linear(vector)
            vector=self.sigmoid(vector)

            print("vector value:",vector)
            res.append([vector.squeeze().item()])
        return np.round(res,4)

