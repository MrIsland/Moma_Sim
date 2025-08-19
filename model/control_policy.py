import os, sys
import numpy as np
import torch

from .PointNavResNetNet import *
from rl_games.algos_torch import model_builder, torch_ext
from copy import copy
from utils.common import unsqueeze_obs, rescale_actions, to_torch
from rl_games.algos_torch import network_builder, players

class ControlPolicy:
    def __init__(self, cfg, device='cuda:0'):
        self.cfg = cfg
        self.device = device
        self.cfg_visual_encoder = self.cfg['network']['visual_encoder']
        self.ckpt_path = self.cfg['checkpoint'] if self.cfg['checkpoint'] != '' else None
        model_builder.register_network('pointnavresnetnet', lambda **kwargs: PointNavResNetBuilder())
        # model_builder.register_network('actor_critic', lambda **kwargs: network_builder.A2CBuilder())

        num_visual_observation = self.cfg_visual_encoder['in_channels'] * self.cfg_visual_encoder['img_h'] * self.cfg_visual_encoder['img_w']
        # obs_shape = 19 + 640 + num_visual_observation
        obs_shape = 15 + 640
        obs_shape = np.zeros(obs_shape).shape
        self.config = {
            'actions_num': 2 + 6 + 1,
            'input_shape': obs_shape,
            'num_seqs': 1,
            'value_size': 1,
            'normalize_value': True,
            'normalize_input': True,
        }
        self.normalize_input = self.config['normalize_input']
        self.normalize_value = self.config.get('normalize_value', False)
        builder = model_builder.ModelBuilder()
        self.model = builder.load(self.cfg).build(self.config)
        self.model.to(self.device)
        self.model.eval()

        self.clip_actions = True
        self.actions_low = np.ones(self.config['actions_num']) * -1
        self.actions_high = np.ones(self.config['actions_num'])
        self.actions_low = to_torch(self.actions_low)
        self.actions_high = to_torch(self.actions_high)
        self.states = None
        print('__init__')

    def restore(self, ckpt_path=None):
        if ckpt_path is None:
           ckpt_path = self.ckpt_path
        checkpoint = torch_ext.load_checkpoint(ckpt_path)
        self.model.load_state_dict(checkpoint['model'])
        if self.normalize_input and 'running_mean_std' in checkpoint:
            self.model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])


    def _preproc_obs(self, obs_batch):
        if type(obs_batch) is dict:
            obs_batch = copy.copy(obs_batch)
            for k, v in obs_batch.items():
                if v.dtype == torch.uint8:
                    obs_batch[k] = v.float() / 255.0
                else:
                    obs_batch[k] = v
        else:
            if obs_batch.dtype == torch.uint8:
                obs_batch = obs_batch.float() / 255.0
        return obs_batch

    def get_action(self, obs, is_deterministic = True):
        # obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs' : obs,
            'rnn_states' : self.states
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)
        mu = res_dict['mus']
        action = res_dict['actions']
        self.states = res_dict['rnn_states']
        if is_deterministic:
            current_action = mu
        else:
            current_action = action

        if self.clip_actions:
            return rescale_actions(self.actions_low, self.actions_high, torch.clamp(current_action, -1.0, 1.0))
        else:
            return current_action