import os, sys
import torch
from rl_games.algos_torch import model_builder, torch_ext

ckpt_path = '/home/island/Desktop/mobile_manipulation/IsaacGymEnvs/isaacgymenvs/runs/dagger/dagger_with_model/MomaMovePickBottleWGraspPosDR_12-23-00-41/nn'
ckpt_name = 'model_600.pth'

save_path = '/home/island/Desktop/mobile_manipulation/IsaacGymEnvs/isaacgymenvs/runs/dagger/dagger_with_model/MomaMovePickBottleWGraspPosDR_12-23-00-41/nn'
save_name = 'model_600_optim'

if __name__ == '__main__':
    ckpt_file = os.path.join(ckpt_path, ckpt_name)
    checkpoint = torch_ext.load_checkpoint(ckpt_file)
    state_dict = checkpoint['optimizer'].state_dict()
    checkpoint['optimizer'] = state_dict
    save_file = os.path.join(save_path, save_name)
    torch_ext.save_checkpoint(save_file, checkpoint)

    print('add optimizer to ckpt')