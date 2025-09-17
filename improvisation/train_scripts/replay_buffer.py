import pickle
import random

import torch


class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.buffer = []

    def push(self, input_seq, output_seq, responsibility):
        self.buffer.append((input_seq, output_seq, responsibility))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        input_seqs, output_seqs, responsibilities = zip(*batch)
        return (
            torch.stack(input_seqs),
            torch.stack(output_seqs),
            torch.stack(responsibilities)
        )
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.data, f)
            
    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            buffer = cls()
            buffer.data = pickle.load(f)
            return buffer

    def __len__(self):
        return len(self.buffer)
