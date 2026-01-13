import pdb

import torch
import copy
from torch.utils.data import Dataset

class DAggerDataset(Dataset):
    def __init__(self, batch_size, minibatch_size, is_discrete, is_rnn, device):
        self.is_rnn = is_rnn
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.device = device
        self.length = self.batch_size // self.minibatch_size
        self.is_discrete = is_discrete
        self.is_continuous = not is_discrete
        self.values_dict = {}
        self.special_names = ['rnn_states']

    def update_values_dict(self, values_dict):
        self.values_dict = values_dict

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        start = idx * self.minibatch_size
        end = (idx + 1) * self.minibatch_size
        self.last_range = (start, end)
        input_dict = {}
        for k, v in self.values_dict.items():
            if k not in self.special_names and v is not None:
                if type(v) is dict:
                    v_dict = { kd:vd[start:end] for kd, vd in v.items()}
                    input_dict[k] = v_dict
                else:
                    input_dict[k] = v[start:end]

        return input_dict