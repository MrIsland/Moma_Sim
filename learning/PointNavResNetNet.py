import copy
import pdb

import torch
from torch import nn
import torch.nn.functional as F
from rl_games.algos_torch.network_builder import NetworkBuilder
from learning.pointnet import PointNetEncoder
from learning.resnet import resnet18, ResNetEncoder
from learning.depth_anything_v2_encoder import DepthAnythingV2Encoder, ReduceAndPool

class PointNavResNetNet(NetworkBuilder.BaseNetwork):
    def __init__(self, params, **kwargs):
        actions_num = kwargs.pop('actions_num')
        input_shape = kwargs.pop('input_shape')
        self.value_size = kwargs.pop('value_size', 1)
        self.num_seqs = num_seqs = kwargs.pop('num_seqs', 1)

        NetworkBuilder.BaseNetwork.__init__(self)
        self.load(params)
        self.actor_cnn = nn.Sequential()
        self.critic_cnn = nn.Sequential()
        self.actor_mlp = nn.Sequential()
        self.critic_mlp = nn.Sequential()
        self.grasp_pos_encoder = PointNetEncoder(global_feat=True, channel=10) # pos: 3, rot: 6, score: 1
        # self.visual_encoder = ResNetEncoder(make_backbone=resnet18, normalize_visual_inputs=False)  # resnet - in_channels: 3
        # pdb.set_trace()
        # self.visual_encoder = DepthAnythingV2Encoder(encoder='vits', device='cuda', input_size=self.ve_img_h)
        self.visual_encoder = ReduceAndPool(in_ch=64, proj_ch=32, out_spatial=(1, 1))
        if self.has_cnn:
            if self.permute_input:
                input_shape = torch_ext.shape_whc_to_cwh(input_shape)
            cnn_args = {
                'ctype': self.cnn['type'],
                'input_shape': input_shape,
                'convs': self.cnn['convs'],
                'activation': self.cnn['activation'],
                'norm_func_name': self.normalization,
            }
            self.actor_cnn = self._build_conv(**cnn_args)

            if self.separate:
                self.critic_cnn = self._build_conv(**cnn_args)

        self.num_ve_input = int(self.ve_img_h / 14 * self.ve_img_w / 14 * 64)
        ## 计算mlp输入的维度
        if self.ve_net is not None:
            state_input_shape = input_shape[0] - 640 - self.num_ve_input
            # mlp_input_shape = self._calc_input_size((state_input_shape + 512 + 792064,), self.actor_cnn)
            # mlp_input_shape = self._calc_input_size((state_input_shape + 512 + 113152,), self.actor_cnn)
            mlp_input_shape = self._calc_input_size((state_input_shape + 512 + 256,), self.actor_cnn)
        else:
            state_input_shape = input_shape[0] - 640
            mlp_input_shape = self._calc_input_size((state_input_shape + 512,), self.actor_cnn)
        in_mlp_shape = mlp_input_shape

        if len(self.units) == 0:
            out_size = mlp_input_shape
        else:
            out_size = self.units[-1]

        if self.has_rnn:
            if not self.is_rnn_before_mlp:
                rnn_in_size = out_size
                out_size = self.rnn_units
                if self.rnn_concat_input:
                    rnn_in_size += in_mlp_shape
            else:
                rnn_in_size = in_mlp_shape
                in_mlp_shape = self.rnn_units

            if self.separate:
                self.a_rnn = self._build_rnn(self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers)
                self.c_rnn = self._build_rnn(self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers)
                if self.rnn_ln:
                    self.a_layer_norm = torch.nn.LayerNorm(self.rnn_units)
                    self.c_layer_norm = torch.nn.LayerNorm(self.rnn_units)
            else:
                self.rnn = self._build_rnn(self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers)
                if self.rnn_ln:
                    self.layer_norm = torch.nn.LayerNorm(self.rnn_units)
        mlp_args = {
            'input_size': in_mlp_shape,
            'units': self.units,
            'activation': self.activation,
            'norm_func_name': self.normalization,
            'dense_func': torch.nn.Linear,
            'd2rl': self.is_d2rl,
            'norm_only_first_layer': self.norm_only_first_layer
        }
        self.actor_mlp = self._build_mlp(**mlp_args)
        if self.separate:
            self.critic_mlp = self._build_mlp(**mlp_args)

        self.value = self._build_value_layer(out_size, self.value_size)
        self.value_act = self.activations_factory.create(self.value_activation)

        if self.is_discrete:
            self.logits = torch.nn.Linear(out_size, actions_num)
        '''
            for multidiscrete actions num is a tuple
        '''

        if self.is_multi_discrete:
            self.logits = torch.nn.ModuleList([torch.nn.Linear(out_size, num) for num in actions_num])
        if self.is_continuous:
            self.mu = torch.nn.Linear(out_size, actions_num)
            self.mu_act = self.activations_factory.create(self.space_config['mu_activation'])
            mu_init = self.init_factory.create(**self.space_config['mu_init'])
            self.sigma_act = self.activations_factory.create(self.space_config['sigma_activation'])
            sigma_init = self.init_factory.create(**self.space_config['sigma_init'])

            if self.fixed_sigma:
                self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)
            else:
                self.sigma = torch.nn.Linear(out_size, actions_num)
        mlp_init = self.init_factory.create(**self.initializer)
        if self.has_cnn:
            cnn_init = self.init_factory.create(**self.cnn['initializer'])
        else:
            cnn_init = copy.deepcopy(mlp_init)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                cnn_init(m.weight)
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                mlp_init(m.weight)
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)
        if self.is_continuous:
            mu_init(self.mu.weight)
            if self.fixed_sigma:
                sigma_init(self.sigma)
            else:
                sigma_init(self.sigma.weight)
        print('PointNavResNetNet')

    def forward(self, obs_dict):
        obs = obs_dict['obs']    # (num_envs, state + 640 + h*w*c)
        states = obs_dict.get('rnn_states', None)
        dones = obs_dict.get('dones', None)
        bptt_len = obs_dict.get('bptt_len', 0)
        if self.ve_net is not None:
            num_ve_input = int(self.ve_img_h / 14 * self.ve_img_w / 14 * 64)
            ve_input = obs[:, -num_ve_input:].reshape(obs.shape[0], int(self.ve_img_h / 14), int(self.ve_img_w / 14), 64)
            ve_feature = self.visual_encoder(ve_input.permute(0, 3, 1, 2))
            # ve_feature = self.reduce_and_pool(ve_feature.permute(0, 3, 1, 2))
            other_obs = obs[:, :-num_ve_input]
        else:
            other_obs = obs
        grasp_feature = self.grasp_pos_encoder(other_obs[:, -640:].reshape(obs.shape[0], -1, 10).permute(0, 2, 1))
        # pdb.set_trace()
        if self.ve_net is not None:
            obs = torch.cat((other_obs[:, :-640], grasp_feature, ve_feature), dim=1)  #  783
        else:
            obs = torch.cat((other_obs[:, :-640], grasp_feature), dim=1)

        if self.has_cnn:
            # for obs shape 4
            # input expected shape (B, W, H, C)
            # convert to (B, C, W, H)
            if self.permute_input and len(obs.shape) == 4:
                obs = obs.permute((0, 3, 1, 2))

        if self.separate:
            a_out = c_out = obs
            a_out = self.actor_cnn(a_out)
            a_out = a_out.contiguous().view(a_out.size(0), -1)

            c_out = self.critic_cnn(c_out)
            c_out = c_out.contiguous().view(c_out.size(0), -1)

            if self.has_rnn:
                seq_length = obs_dict.get('seq_length', 1)

                if not self.is_rnn_before_mlp:
                    a_out_in = a_out
                    c_out_in = c_out
                    a_out = self.actor_mlp(a_out_in)
                    c_out = self.critic_mlp(c_out_in)

                    if self.rnn_concat_input:
                        a_out = torch.cat([a_out, a_out_in], dim=1)
                        c_out = torch.cat([c_out, c_out_in], dim=1)

                batch_size = a_out.size()[0]
                num_seqs = batch_size // seq_length
                a_out = a_out.reshape(num_seqs, seq_length, -1)
                c_out = c_out.reshape(num_seqs, seq_length, -1)

                a_out = a_out.transpose(0, 1)
                c_out = c_out.transpose(0, 1)
                if dones is not None:
                    dones = dones.reshape(num_seqs, seq_length, -1)
                    dones = dones.transpose(0, 1)

                if len(states) == 2:
                    a_states = states[0]
                    c_states = states[1]
                else:
                    a_states = states[:2]
                    c_states = states[2:]
                a_out, a_states = self.a_rnn(a_out, a_states, dones, bptt_len)
                c_out, c_states = self.c_rnn(c_out, c_states, dones, bptt_len)

                a_out = a_out.transpose(0, 1)
                c_out = c_out.transpose(0, 1)
                a_out = a_out.contiguous().reshape(a_out.size()[0] * a_out.size()[1], -1)
                c_out = c_out.contiguous().reshape(c_out.size()[0] * c_out.size()[1], -1)

                if self.rnn_ln:
                    a_out = self.a_layer_norm(a_out)
                    c_out = self.c_layer_norm(c_out)

                if type(a_states) is not tuple:
                    a_states = (a_states,)
                    c_states = (c_states,)
                states = a_states + c_states

                if self.is_rnn_before_mlp:
                    a_out = self.actor_mlp(a_out)
                    c_out = self.critic_mlp(c_out)
            else:
                a_out = self.actor_mlp(a_out)
                c_out = self.critic_mlp(c_out)

            value = self.value_act(self.value(c_out))

            if self.is_discrete:
                logits = self.logits(a_out)
                return logits, value, states

            if self.is_multi_discrete:
                logits = [logit(a_out) for logit in self.logits]
                return logits, value, states

            if self.is_continuous:
                mu = self.mu_act(self.mu(a_out))
                if self.fixed_sigma:
                    sigma = mu * 0.0 + self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(a_out))

                return mu, sigma, value, states
        else:
            out = obs
            out = self.actor_cnn(out)
            out = out.flatten(1)

            if self.has_rnn:
                seq_length = obs_dict.get('seq_length', 1)

                out_in = out
                if not self.is_rnn_before_mlp:
                    out_in = out
                    out = self.actor_mlp(out)
                    if self.rnn_concat_input:
                        out = torch.cat([out, out_in], dim=1)

                batch_size = out.size()[0]
                num_seqs = batch_size // seq_length
                out = out.reshape(num_seqs, seq_length, -1)

                if len(states) == 1:
                    states = states[0]

                out = out.transpose(0, 1)
                if dones is not None:
                    dones = dones.reshape(num_seqs, seq_length, -1)
                    dones = dones.transpose(0, 1)
                out, states = self.rnn(out, states, dones, bptt_len)
                out = out.transpose(0, 1)
                out = out.contiguous().reshape(out.size()[0] * out.size()[1], -1)

                if self.rnn_ln:
                    out = self.layer_norm(out)
                if self.is_rnn_before_mlp:
                    out = self.actor_mlp(out)
                if type(states) is not tuple:
                    states = (states,)
            else:
                out = self.actor_mlp(out)
            value = self.value_act(self.value(out))

            if self.central_value:
                return value, states

            if self.is_discrete:
                logits = self.logits(out)
                return logits, value, states
            if self.is_multi_discrete:
                logits = [logit(out) for logit in self.logits]
                return logits, value, states
            if self.is_continuous:
                mu = self.mu_act(self.mu(out))
                if self.fixed_sigma:
                    sigma = self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(out))
                return mu, mu*0 + sigma, value, states

    def is_separate_critic(self):
        return self.separate

    def is_rnn(self):
        return self.has_rnn

    def load(self, params):
        self.separate = params.get('separate', False)
        self.units = params['mlp']['units']
        self.activation = params['mlp']['activation']
        self.initializer = params['mlp']['initializer']
        self.is_d2rl = params['mlp'].get('d2rl', False)
        self.norm_only_first_layer = params['mlp'].get('norm_only_first_layer', False)
        self.value_activation = params.get('value_activation', 'None')
        self.normalization = params.get('normalization', None)
        self.has_rnn = 'rnn' in params
        self.has_space = 'space' in params
        self.central_value = params.get('central_value', False)
        self.joint_obs_actions_config = params.get('joint_obs_actions', None)

        if self.has_space:
            self.is_multi_discrete = 'multi_discrete' in params['space']
            self.is_discrete = 'discrete' in params['space']
            self.is_continuous = 'continuous' in params['space']
            if self.is_continuous:
                self.space_config = params['space']['continuous']
                self.fixed_sigma = self.space_config['fixed_sigma']
            elif self.is_discrete:
                self.space_config = params['space']['discrete']
            elif self.is_multi_discrete:
                self.space_config = params['space']['multi_discrete']
        else:
            self.is_discrete = False
            self.is_continuous = False
            self.is_multi_discrete = False

        if self.has_rnn:
            self.rnn_units = params['rnn']['units']
            self.rnn_layers = params['rnn']['layers']
            self.rnn_name = params['rnn']['name']
            self.rnn_ln = params['rnn'].get('layer_norm', False)
            self.is_rnn_before_mlp = params['rnn'].get('before_mlp', False)
            self.rnn_concat_input = params['rnn'].get('concat_input', False)

        if 'cnn' in params:
            self.has_cnn = True
            self.cnn = params['cnn']
            self.permute_input = self.cnn.get('permute_input', True)
        else:
            self.has_cnn = False

        if 'visual_encoder' in params:
            self.ve_net = params['visual_encoder']['net']
            self.ve_in_channels = params['visual_encoder']['in_channels']
            self.ve_img_h = params['visual_encoder']['img_h']
            self.ve_img_w = params['visual_encoder']['img_w']
            self.ve_flag_crop = params['visual_encoder']['flag_crop']
        else:
            self.ve_net = None

# ============================================== #

class PointNavResNetBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        return PointNavResNetNet(self.params, **kwargs)

    def __call__(self, name, **kwargs):
        return self.build(name, **kwargs)