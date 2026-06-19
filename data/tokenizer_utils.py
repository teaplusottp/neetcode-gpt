from typing import List, Dict

class Solution:

    def tokenize_numbers(self, numbers, vocab) :
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        def one_num(number_str,vocab,start=0):
            res=""
            s=start
            for i in range(start,len(number_str)):
                if(number_str[start:i+1] in vocab.keys()):
                    res=number_str[start:i+1]
                    s=i
            if(res==""):
                res=number_str[start:]
            return res,s
        
        res=[]
        for number in numbers:
            number=str(number)
            start=0
            tmp_lst=[]
            while(start<len(number)):
                tmp,start=one_num(number,vocab,start)
                start+=1
                tmp_lst.append(tmp)
            res.append(tmp_lst)
        return res

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        def one_char(txt,vocab,start=0):
            res=""
            s=start
            for i in range(start,len(txt)):
                if(txt[start:i+1] in vocab):
                    res=txt[start:i+1]
                    s=i
            if(res==""):
                res=txt[start:]
            return res,s
      
        start=0
        tmp_lst=[]
        count=0
        while(start<len(text)):
            tmp,start=one_char(text,vocab,start)
            start+=1
            count+=1
        return count


    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        tokens=self.count_tokens(text,vocab)
        print(tokens)
        words=len(text.split(" "))
        return round(tokens/words,4)
