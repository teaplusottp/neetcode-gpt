import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def BOW(self, data):
        words = sorted(set(" ".join(data).split()))
        return {w: i+1 for i, w in enumerate(words)}  # start from 1


    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        # pass

        bow=self.BOW(positive+negative)
        encode_pos=[]
        encode_neg=[]
        print(bow)
        for i in positive:
            tmp=[]
            for word in i.split(" "):
                tmp.append(bow[word])
            encode_pos.append(torch.tensor(tmp))
        for j in negative:
            tmp=[]
            for word in j.split(" "):
                tmp.append(bow[word])
            encode_neg.append(torch.tensor(tmp))
        all_sentences = encode_pos+encode_neg
        out =  nn.utils.rnn.pad_sequence(all_sentences, batch_first=True, padding_value=0.0)

        return out
