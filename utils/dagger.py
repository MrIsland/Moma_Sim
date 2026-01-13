import os, sys
import pdb

import numpy as np
import random
from datetime import datetime

import enum
from isaacgym import gymutil
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from utils.common import to_torch
from rl_games.algos_torch import model_builder, torch_ext
from rl_games.common import vecenv
from rl_games.common.experience import ExperienceBuffer

from learning.PointNavResNetNet import PointNavResNetBuilder
from learning.ModelA2CContinuousLogStdVision import ModelA2CContinuousLogStdVision
from learning.lazy_pth_dataset import LazyPthDataset

from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from copy import copy
from utils.dagger_experience import ExperienceBufferDAgger
from utils.dagger_dataset import DAggerDataset
import time

def set_np_formatting():
    np.set_printoptions(edgeitems=30, infstr='inf',
                        linewidth=4000, nanstr='nan', precision=2,
                        suppress=False, threshold=10000, formatter=None)

def retrieve_logdir(args):
    logdir = os.path.join(args.logdir, 'dagger', '{}_{date:%d-%H-%M-%S}'.format(args.algo, date=datetime.now()))
    os.makedirs(logdir, exist_ok=True)
    return logdir

def swap_and_flatten01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    if arr is None:
        return arr
    s = arr.size()
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])

# def merge_args_into_cfg(cfg, args, skip_none=True):
#     args_dict = vars(args)
#     safe_args = {}
#
#     for k, v in args_dict.items():
#         # 跳过 None（可选）
#         if skip_none and v is None:
#             continue
#         # 枚举 -> name
#         if isinstance(v, enum.Enum):
#             safe_args[k] = v.name
#         # 原始类型 -> 保留
#         elif isinstance(v, (str, int, float, bool, list, dict, type(None))):
#             safe_args[k] = v
#         # 其他类型 -> 转字符串
#         else:
#             safe_args[k] = str(v)
#
#     args_cfg = OmegaConf.create(safe_args)
#     pdb.set_trace()
#     return OmegaConf.merge(cfg, args_cfg)

def get_args(cfg):
    custom_parameters = [
        {"name": "--task", "type": str, "default": "MomaMovePickBottleWGraspPosDR",
            "help": "Student model's task"},
        {"name": "--algo", "type": str, "default": "dagger_with_dataset",
            "help": "Choose an algorithm"},
        {"name": "--task_cfg", "type":str, "default": "./cfg/task",
            "help": "Task config"},
        {"name": "--rl_train_cfg", "type": str, "default": "./cfg/train",
         "help": "RL train config"},
        {"name": "--dagger_cfg", "type": str, "default": "./cfg/dagger",
         "help": "Dagger config"},
        {"name": "--logdir", "type": str, "default": "./runs",
         "help": "Logging file path"}
    ]
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)
    logdir = retrieve_logdir(args)
    with open(os.path.join(logdir, 'config_dagger.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    print('logdir:    {}'.format(logdir))
    print('args:      {}'.format(args))

    return args

def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action =  action * d + m
    return scaled_action

class DAggerWithDatasetAgent:
    def __init__(self,
                 cfg,
                 num_learning_epochs=10,
                 num_mini_batches=32,
                 learning_rate=1e-3,
                 print_log=True,
                 is_train=True,
                 logdir='runs',
                 device='cpu'):
        self.cfg = cfg
        self.num_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.lr = learning_rate
        self.print_log = print_log
        self.is_train = is_train
        self.log_dir = logdir
        self.device = device
        self.dataset_path = cfg.dataset.path

        self.cfg_visual_encoder = self.cfg.student.train.params.network.visual_encoder
        self.cfg_builder = self.cfg.student.train.params

        model_builder.register_network('pointnavresnetnet', lambda **kwargs: PointNavResNetBuilder())
        model_builder.register_model('continuous_a2c_logstd_vision',
                                        lambda network, **kwargs: ModelA2CContinuousLogStdVision(network))

        num_vision_obs = self.cfg_visual_encoder['img_h'] / 14 * self.cfg_visual_encoder['img_w'] / 14 * 64
        obs_shape = 15 + 640 + int(num_vision_obs)
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

        self.model = builder.load(self.cfg.student.train.params).build(self.config)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), self.lr)

        self.clip_actions = True
        self.actions_low = np.ones(self.config['actions_num']) * -1
        self.actions_high = np.ones(self.config['actions_num'])
        self.actions_low = to_torch(self.actions_low, device=self.device)
        self.actions_high = to_torch(self.actions_high, device=self.device)
        self.states = None

        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.epoch_num = 0
        self.current_learning_iteration = 0

        self.dataset = LazyPthDataset(self.dataset_path)
        self.loader = DataLoader(self.dataset, batch_size=1, shuffle=True, num_workers=1)

    def update_epoch(self):
        self.epoch_num += 1
        return self.epoch_num

    def restore(self, ckpt_path=None):
        if ckpt_path is None:
           ckpt_path = self.ckpt_path
        checkpoint = torch_ext.load_checkpoint(ckpt_path)
        self.model.load_state_dict(checkpoint['model'])
        if self.normalize_input and 'running_mean_std' in checkpoint:
            self.model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

    def log(self, loss, cnt):
        self.writer.add_scalar('Loss/mean_policy_loss', loss, cnt)

    def save_ckpt(self, epoch_idx):
        model_dir = os.path.join(self.log_dir, 'nn')
        os.makedirs(model_dir, exist_ok=True)
        # torch.save(self.model.state_dict(), os.path.join(model_dir, 'model_{}.pt'.format(epoch_idx)))
        state = self.get_full_state_weights()
        fn = os.path.join(model_dir, 'model_{}'.format(epoch_idx))
        torch_ext.save_checkpoint(fn, state)

    def get_full_state_weights(self):
        state = self.get_weights()
        state['epoch'] = 0
        state['frame'] = 0
        state['optimizer'] = {}

        return state

    def get_weights(self):
        state = self.get_stats_weights()
        state['model'] = self.model.state_dict()
        return state

    def get_stats_weights(self):
        state = {}
        if self.normalize_input:
            state['running_mean_std'] = self.model.running_mean_std.state_dict()
        if self.normalize_value:
            state['reward_mean_std'] = self.model.value_mean_std.state_dict()

        return state

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

    def get_actions(self, obses, is_deterministic=True):
        obses = self._preproc_obs(obses)
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': obses,
            'rnn_states': self.states
        }
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

    def get_obses(self, obs_dict):
        obs = ["eef_pos", "eef_6d", "robot_vel", "fusion_grasp_pos_vector", "ve_feature"]
        # obs = ["eef_pos", "eef_6d", "robot_vel", "fusion_grasp_pos_vector"]
        obs_buf = to_torch(torch.cat([obs_dict[ob] for ob in obs], dim=-1), device=self.device).squeeze(0)
        return obs_buf

    def update(self, batch):
        mean_policy_loss = 0
        save_dict_idx = batch['save_dict_idx']
        num_envs = batch['num_envs']
        # obs = batch['obs']
        action = to_torch(batch['action'], device=self.device).squeeze(0)
        obs = self.get_obses(batch['obs'])
        num_batch_epoch = num_envs // self.num_mini_batches
        # pdb.set_trace()
        for _ in range(4):
            for idx in range(num_batch_epoch):
                batch_obses = obs[idx * self.num_mini_batches: (idx + 1) * self.num_mini_batches]
                batch_actions = action[idx * self.num_mini_batches: (idx + 1) * self.num_mini_batches]
                stu_actions = self.get_actions(batch_obses)
                self.optimizer.zero_grad()
                loss = F.mse_loss(batch_actions, stu_actions.float())
                loss.backward()
                self.optimizer.step()

                mean_policy_loss += loss.item()

        num_updates = self.num_mini_batches * num_batch_epoch
        mean_policy_loss /= num_updates
        return mean_policy_loss

    def train_epoch(self, iter, log_interval):
        self.model.train()
        t_loader = tqdm(self.loader, disable=False, dynamic_ncols=True)
        epoch_loss = 0
        for i, batch in enumerate(t_loader):
            loss = self.update(batch)
            if self.print_log:
                self.log(loss, i)
            if i % log_interval == 0:
                with torch.no_grad():
                    self.save_ckpt(iter * len(self.dataset) + i)
            epoch_loss += loss

        epoch_loss /= len(t_loader)
        print('epoch_iter:   {};  epoch_loss:    {}'.format(iter, epoch_loss))


    def run(self, log_interval=1):
        if self.is_train:
            for iter in range(0, self.num_epochs):
                print('-------------- iter:  {} --------------'.format(iter))
                self.train_epoch(iter, log_interval)
                self.update_epoch()
        else:
            #### 只测试成功率
            pass

class DAggerWithModelAgent:
    def __init__(self,
                 cfg,
                 num_learning_epochs=10,
                 num_mini_batches=32,
                 learning_rate=1e-3,
                 print_log=True,
                 is_train=True,
                 logdir='runs',
                 device='cpu'):
        self.cfg = cfg
        self.cfg_student = cfg.student
        self.cfg_expert = cfg.expert
        self.num_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.lr = learning_rate
        self.print_log = print_log
        self.is_train = is_train
        self.log_dir = logdir
        self.device = device

        self.cfg_visual_encoder = self.cfg.student.train.params.network.visual_encoder
        self.cfg_builder = self.cfg.student.train.params

        model_builder.register_network('pointnavresnetnet', lambda **kwargs: PointNavResNetBuilder())
        model_builder.register_model('continuous_a2c_logstd_vision',
                                        lambda network, **kwargs: ModelA2CContinuousLogStdVision(network))

        num_vision_obs = self.cfg_visual_encoder['img_h'] / 14 * self.cfg_visual_encoder['img_w'] / 14 * 64
        state_obs_size = 15 + 640
        vision_obs_size = int(num_vision_obs)
        expert_obs_shape = np.zeros(state_obs_size).shape
        student_obs_shape = np.zeros(state_obs_size + vision_obs_size).shape

        self.config_student = {
            'actions_num': 2 + 6 + 1,
            'input_shape': student_obs_shape,
            'num_seqs': 1,
            'value_size': 1,
            'normalize_value': True,
            'normalize_input': True,
        }

        self.config_expert = {
            'actions_num': 2 + 6 + 1,
            'input_shape': expert_obs_shape,
            'num_seqs': 1,
            'value_size': 1,
            'normalize_value': True,
            'normalize_input': True,
        }

        self.normalize_input = self.config_student['normalize_input']
        self.normalize_value = self.config_student.get('normalize_value', False)
        builder = model_builder.ModelBuilder()

        self.model_student = builder.load(self.cfg_student.train.params).build(self.config_student)
        self.model_expert = builder.load(self.cfg_expert.train.params).build(self.config_expert)
        self.model_student.to(self.device)
        self.model_expert.to(self.device)
        self.model_expert.eval()
        self.restore(ckpt_path=self.cfg_expert.checkpoint, model_name='expert')
        self.restore(ckpt_path=self.cfg_student.checkpoint, model_name='student')
        self.student_obs = ["eef_pos", "eef_6d", "robot_vel", "grasp_pos_vector", "ve_feature"]
        self.expert_obs = ["eef_pos", "eef_6d", "robot_vel", "perfect_grasp_pos_vector"]

        self.optimizer = optim.Adam(self.model_student.parameters(), self.lr)

        self.clip_actions = True
        self.actions_low = np.ones(self.config_student['actions_num']) * -1
        self.actions_high = np.ones(self.config_student['actions_num'])
        self.actions_low = to_torch(self.actions_low, device=self.device)
        self.actions_high = to_torch(self.actions_high, device=self.device)
        self.states = None

        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.epoch_num = 0
        self.current_learning_iteration = 0

        self.vec_env = None
        self.create_env()

        self.cfg_stu_params = self.cfg_student.train.params
        config = self.cfg_stu_params['config']
        self.num_actors = config['num_actors']
        self.horizon_length = config['horizon_length']
        self.central_value_config = config.get('central_value_config', None)
        self.has_central_value = self.central_value_config is not None
        self.use_action_masks = config.get('use_action_masks', False)
        self.ppo_device = config.get('device', 'cuda:0')

        self.normalize_advantage = config['normalize_advantage']
        self.normalize_rms_advantage = config.get('normalize_rms_advantage', False)
        self.normalize_input = self.config_student['normalize_input']
        self.normalize_value = self.config_student.get('normalize_value', False)
        self.truncate_grads = self.config_student.get('truncate_grads', False)
        self.num_agents = self.env_info.get('agents', 1)
        self.value_size = self.env_info.get('value_size', 1)
        self.minibatch_size = config['minibatch_size']
        self.max_epochs = config['max_epochs']
        self.observation_space = self.env_info['observation_space']
        self.rnn_states = None
        self.is_tensor_obses = False

        self.batch_size = self.horizon_length * self.num_actors * self.num_agents
        self.batch_size_envs = self.horizon_length * self.num_actors
        self.frame = 0

        self.init_tensors()
        self.DAgger_dataset = DAggerDataset(self.batch_size, self.minibatch_size, is_discrete=False, is_rnn=False,
                                            device=self.ppo_device)

    def create_env(self):
        env_config = self.config_student.get('env_config', {})
        env_name = self.cfg_student.train.params.config.env_name
        self.vec_env = vecenv.create_vec_env(env_name, 1, **env_config)
        self.env_info = self.vec_env.get_env_info()

    def init_tensors(self):
        batch_size = self.num_agents * self.num_actors
        algo_info = {
            'num_actors' : self.num_actors,
            'horizon_length' : self.horizon_length,
            'has_central_value' : self.has_central_value,
            'use_action_masks' : self.use_action_masks
        }
        self.experience_buffer = ExperienceBufferDAgger(self.env_info, algo_info, self.ppo_device)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.ppo_device)
        self.dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.ppo_device)

        self.update_list = ['actions']
        if self.use_action_masks:
            self.update_list += ['action_masks']
        self.tensor_list = self.update_list + ['obses', 'dones']

    def preprocess_actions(self, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()
        return actions

    def _preproc_obs(self, obs_batch):
        if type(obs_batch) is dict:
            obs_batch = copy.copy(obs_batch)
            for k,v in obs_batch.items():
                if v.dtype == torch.uint8:
                    obs_batch[k] = v.float() / 255.0
                else:
                    obs_batch[k] = v
        else:
            if obs_batch.dtype == torch.uint8:
                obs_batch = obs_batch.float() / 255.0
        return obs_batch

    # def get_action_values(self, obs_dict):
    #     processed_obs_exp = self._preproc_obs(obs_dict['expert_obs'])
    #     processed_obs_stu = self._preproc_obs(obs_dict['student_obs'])
    #     self.model_student.eval()
    #     input_dict_stu = {
    #         'is_train': False,
    #         'prev_actions': None,
    #         'obs' : processed_obs_stu,
    #         'rnn_states' : self.rnn_states
    #     }
    #
    #     input_dict_exp = {
    #         'is_train': False,
    #         'prev_actions': None,
    #         'obs' : processed_obs_exp,
    #         'rnn_states' : self.rnn_states
    #     }
    #
    #     with torch.no_grad():
    #         res_dict_student = self.model_student(input_dict_stu)
    #         res_dict_expert = self.model_expert(input_dict_exp)
    #
    #     res_dict = {
    #         'student': res_dict_student,
    #         'expert': res_dict_expert,
    #     }
    #
    #     return res_dict

    def get_expert_actions(self, obses, is_deterministic=True):
        obses = self._preproc_obs(obses)
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': obses,
            'rnn_states': self.states
        }
        with torch.no_grad():
            res_dict = self.model_expert(input_dict)
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

    def get_student_actions(self, obses, is_deterministic=True):
        obses = self._preproc_obs(obses)
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': obses,
            'rnn_states': self.states
        }
        res_dict = self.model_student(input_dict)
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

    def update(self, obs_dict):
        student_obses = obs_dict['obs']
        expert_actions = obs_dict['actions']

        student_actions = self.get_student_actions(student_obses)
        self.optimizer.zero_grad()
        loss = F.mse_loss(expert_actions, student_actions.float())
        loss.backward()
        self.optimizer.step()

        return loss

    def play_steps(self):
        update_list = self.update_list

        step_time = 0.0
        for n in range(self.horizon_length):
            # if self.use_action_masks:
            #     masks = self.vec_env.get_action_masks()
            #     res_dict = self.get_masked_action_values(self.obs, masks)
            # else:
            #     res_dict = self.get_action_values(self.obs_dict)
            with torch.no_grad():
                expert_actions = self.get_expert_actions(self.obs_dict['expert_obs'])
                student_actions = self.get_student_actions(self.obs_dict['student_obs'])

            res_dict = {
                'expert': {'actions': expert_actions.float()},
                'student': {'actions': student_actions.float()},
            }

            #  student - obs / expert - action
            self.experience_buffer.update_data('obses', n, self.obs_dict['student_obs'])   # (num_envs, obs_shape)
            self.experience_buffer.update_data('dones', n, self.dones)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict['expert'][k])

            step_time_start = time.time()
            self.obs_dict, rewards, self.dones, infos = self.env_step(res_dict['student']['actions'])
            step_time_end = time.time()

            step_time += (step_time_end - step_time_start)
            not_dones = 1.0 - self.dones.float()

        batch_dict = self.experience_buffer.get_transformed_list(swap_and_flatten01, self.tensor_list)
        batch_dict['played_frames'] = self.batch_size
        batch_dict['step_time'] = step_time

        return batch_dict

    def set_eval(self):
        self.model_student.eval()
        if self.normalize_rms_advantage:
            self.advantage_mean_std.eval()

    def set_train(self):
        self.model_student.train()
        if self.normalize_rms_advantage:
            self.advantage_mean_std.train()

    def obs_to_tensors(self, obs):
        obs_is_dict = isinstance(obs, dict)
        if obs_is_dict:
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value)
        else:
            upd_obs = self.cast_obs(obs)
        # if not obs_is_dict or 'obs' not in obs:
        if not obs_is_dict:
            upd_obs = {'obs' : upd_obs}
        return upd_obs

    def cast_obs(self, obs):
        if isinstance(obs, torch.Tensor):
            self.is_tensor_obses = True
        elif isinstance(obs, np.ndarray):
            assert(obs.dtype != np.int8)
            if obs.dtype == np.uint8:
                obs = torch.ByteTensor(obs).to(self.ppo_device)
            else:
                obs = torch.FloatTensor(obs).to(self.ppo_device)
        return obs

    def _obs_to_tensors_internal(self, obs):
        if isinstance(obs, dict):
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value)
        else:
            upd_obs = self.cast_obs(obs)
        return upd_obs

    def env_reset(self):
        # obs = self.vec_env.reset()
        states = self.vec_env.env.states
        expert_obs_buf = torch.cat([states[ob] for ob in self.expert_obs], dim=-1)
        expert_obs_buf = torch.clamp(expert_obs_buf, -self.vec_env.env.clip_obs, self.vec_env.env.clip_obs).to(self.vec_env.env.rl_device)

        student_obs_buf = torch.cat([states[ob] for ob in self.student_obs], dim=-1)
        student_obs_buf = torch.clamp(student_obs_buf, -self.vec_env.env.clip_obs, self.vec_env.env.clip_obs).to(self.vec_env.env.rl_device)

        obs_dict = {
            'expert_obs': expert_obs_buf,
            'student_obs': student_obs_buf,
        }
        obs_dict = self.obs_to_tensors(obs_dict)

        return obs_dict

    def env_step(self, actions):
        actions = self.preprocess_actions(actions)
        # pdb.set_trace()
        obs, rewards, dones, infos = self.vec_env.step(actions)

        states = self.vec_env.env.states
        expert_obs_buf = torch.cat([states[ob] for ob in self.expert_obs], dim=-1)
        expert_obs_buf = torch.clamp(expert_obs_buf, -self.vec_env.env.clip_obs, self.vec_env.env.clip_obs).to(self.vec_env.env.rl_device)

        student_obs_buf = torch.cat([states[ob] for ob in self.student_obs], dim=-1)
        student_obs_buf = torch.clamp(student_obs_buf, -self.vec_env.env.clip_obs, self.vec_env.env.clip_obs).to(self.vec_env.env.rl_device)
        obs_dict = {
            'expert_obs': expert_obs_buf,
            'student_obs': student_obs_buf,
        }

        if self.is_tensor_obses:
            if self.value_size == 1:
                rewards = rewards.unsqueeze(1)
            return self.obs_to_tensors(obs_dict), rewards.to(self.ppo_device), dones.to(self.ppo_device), infos
        else:
            if self.value_size == 1:
                rewards = np.expand_dims(rewards, axis=1)
            return self.obs_to_tensors(obs_dict), torch.from_numpy(rewards).to(self.ppo_device).float(), torch.from_numpy(dones).to(self.ppo_device), infos

    def prepare_dataset(self, batch_dict):
        actions = batch_dict['actions']   # expert actions
        obses = batch_dict['obses']       # student obses
        dones = batch_dict['dones']

        dataset_dict = {}
        dataset_dict['obs'] = obses
        dataset_dict['actions'] = actions
        dataset_dict['dones'] = dones
        # dataset_dict['step_time'] = step_time

        self.DAgger_dataset.update_values_dict(dataset_dict)

    def train_epoch(self, iter, log_interval):
        ### 1. Data collection
        self.vec_env.set_train_info(self.frame, self)
        self.set_eval()
        with torch.no_grad():
            batch_dict = self.play_steps()
        self.set_train()

        self.curr_frames = batch_dict.pop('played_frames')

        self.prepare_dataset(batch_dict)
        ### 2. training
        mean_policy_loss = 0
        for mini_ep in range(0, self.num_mini_batches):
            for i in range(len(self.DAgger_dataset)):
                loss = self.update(self.DAgger_dataset[i])    # 2048
                mean_policy_loss += loss.item()

        num_updates = self.batch_size * self.num_mini_batches
        mean_policy_loss /= num_updates

        return mean_policy_loss

    def run(self, log_interval=1):
        if self.is_train:
            self.obs_dict = self.env_reset()
            self.curr_frames = self.batch_size_envs
            while True:
                epoch_num = self.update_epoch()
                epoch_loss = self.train_epoch(epoch_num, log_interval=log_interval)
                should_exit = False

                if epoch_num > self.max_epochs:
                    print('MAX EPOCHS NUM!')
                    should_exit = True
                if self.print_log:
                    self.log(epoch_loss, epoch_num)

                if epoch_num % log_interval == 0:
                    with torch.no_grad():
                        print('save_ckpt with epoch_num {}, with epoch_loss {}'.format(epoch_num, epoch_loss))
                        self.save_ckpt(epoch_num)
                print('!!! epoch_num:  {} - epoch_loss:  {};   {date:%d-%H-%M-%S}'.format( epoch_num, epoch_loss, date=datetime.now(),))

                if should_exit:
                    break
        else:
            #### 只测试成功率
            pass
        pass

    def update_epoch(self):
        self.epoch_num += 1
        return self.epoch_num

    def restore(self, ckpt_path=None, model_name='expert'):
        if ckpt_path is None:
            return
        checkpoint = torch_ext.load_checkpoint(ckpt_path)
        if model_name == 'expert':
            self.model_expert.load_state_dict(checkpoint['model'])
            if self.normalize_input and 'running_mean_std' in checkpoint:
                self.model_expert.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        elif model_name == 'student':
            self.model_student.load_state_dict(checkpoint['model'])
            if self.normalize_input and 'running_mean_std' in checkpoint:
                self.model_student.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

    def log(self, loss, cnt):
        self.writer.add_scalar('Loss/mean_policy_loss', loss, cnt)

    def save_ckpt(self, epoch_idx):
        model_dir = os.path.join(self.log_dir, 'nn')
        os.makedirs(model_dir, exist_ok=True)
        # torch.save(self.model.state_dict(), os.path.join(model_dir, 'model_{}.pt'.format(epoch_idx)))
        state = self.get_full_state_weights()
        fn = os.path.join(model_dir, 'model_{}'.format(epoch_idx))
        torch_ext.save_checkpoint(fn, state)

    def get_full_state_weights(self):
        state = self.get_weights()
        state['epoch'] = 0
        state['frame'] = 0
        state['optimizer'] = optim.Adam(self.model_student.parameters(), float(5e-4), eps=1e-08, weight_decay=0.0)

        return state

    def get_weights(self):
        state = self.get_stats_weights()
        state['model'] = self.model_student.state_dict()
        return state

    def get_stats_weights(self):
        state = {}
        if self.normalize_input:
            state['running_mean_std'] = self.model_student.running_mean_std.state_dict()
        if self.normalize_value:
            state['reward_mean_std'] = self.model_student.value_mean_std.state_dict()

        return state
