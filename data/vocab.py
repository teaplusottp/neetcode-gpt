from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)
        a=sorted(set(list(text)))
        stoi={}
        itos={}
        for i in range(len(a)):
            itos[i]=a[i]
            stoi[a[i]]=i
        return stoi,itos

    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # Convert a string to a list of integers using stoi mapping
        res=[]
        for char in text:
            res.append(stoi[char])
        return res

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Convert a list of integers back to a string using itos mapping
        res=""
        for val in ids:
            res+=itos[val]
        return res

