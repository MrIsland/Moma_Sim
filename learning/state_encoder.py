import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

class StateEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(StateEncoder, self).__init__()
        self.state_mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU()
        )

    def forward(self, feats):
        return self.state_mlp(feats)