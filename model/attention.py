import torch
import torch.nn as nn
from torchtyping import TensorType
from math import sqrt
import numpy as np

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)

        self.Wk = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.Wq = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, attention_dim, bias=False)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        seqlen=embedded.shape[1]
        # 1. Project input through K, Q, V linear layers
        Q = self.Wq(embedded)
        K = self.Wk(embedded)
        V = self.Wv(embedded)
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        score=(Q @ K.transpose(-2,-1)) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        mask= torch.tril(torch.ones((seqlen,seqlen)))
        #    then masked_fill positions where mask == 0 with float('-inf')
        # ko biết code như nào, dùng mask để chắn cái ma trận vuông score thôi đúng ko
        aftermask = score.masked_fill(mask == 0, float('-inf'))
        # 4. Apply softmax(dim=2) to masked scores
        scores = torch.softmax(aftermask,dim=-1)
        # 5. Return (scores @ V) rounded to 4 decimal places
        return torch.round(scores @ V*10000)/10000
