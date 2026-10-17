"""Deterministic cyclic shuffled pool; preserve unconsumed indices across budgets."""
import torch

class CyclicSampler:
    def __init__(self,indices,seed):
        self.indices=torch.as_tensor(indices,dtype=torch.int64)
        if not len(self.indices):raise ValueError('Empty sampling pool')
        self.generator=torch.Generator().manual_seed(seed)
        self.pending=torch.empty(0,dtype=torch.int64)
    def draw(self,count):
        pieces=[]
        while count:
            if not len(self.pending):
                self.pending=self.indices[torch.randperm(len(self.indices),generator=self.generator)]
            take=min(count,len(self.pending));pieces.append(self.pending[:take])
            self.pending=self.pending[take:];count-=take
        return torch.cat(pieces)
