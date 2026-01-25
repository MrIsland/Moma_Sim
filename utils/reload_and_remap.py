import os, sys
import numpy as np
import pdb
import random
import torch

from copy import deepcopy
from rl_games.algos_torch import torch_ext

ckpt_path = '/home/island/Desktop/mobile_manipulation/IsaacGymEnvs/isaacgymenvs/runs/MomaMovePickBottleStateFlyWGraspPosDR_13-13-28-36/nn'
ckpt_name = 'MomaMovePickBottleStateFlyWGraspPosDR.pth'

if __name__ == '__main__':
    checkpoint = torch.load(os.path.join(ckpt_path, ckpt_name), map_location='cuda:0')
    torch.save(checkpoint, os.path.join(ckpt_path, ckpt_name))
    print('reload_and_remap')

