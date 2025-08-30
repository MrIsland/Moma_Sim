import os, sys
import numpy as np
import time
from model.mujoco_robot_env import MujocoRobotEnv
from tqdm import tqdm
from utils.common import *

if __name__ == '__main__':
    cfg_path = './cfg'
    cfg_yaml = 'config.yaml'
    config = load_config(os.path.join(cfg_path, cfg_yaml))
    mujoco_robot_env = MujocoRobotEnv(config)
    cmd_arm_limit = np.array([0.05, 0.05, 0.05, 0.1, 0.1, 0.1])
    cmd_base_limit = np.array([0.3, 0.6])
    num_obs = 583
    action_scale = config['asset']['arm_action_scale']
    obs_input_path = '/home/island/Desktop/mobile_manipulation/IsaacGymEnvs/isaacgymenvs/test_vis/pt'
    for i in range(num_obs):
        pt_file = os.path.join(obs_input_path, '{}.pt'.format(i + 1))
        loaded_obs = torch.load(pt_file)
        action = mujoco_robot_env.control_policy.get_action(loaded_obs)
        print('loaded_obs:    {}'.format(loaded_obs[:, :9]))
        action = action.squeeze(0).cpu().detach().numpy()
        action_base, action_arm, action_gripper = action[:2], action[2:-1], np.array([action[-1]])
        action_base = action_base * cmd_base_limit
        # action_base = np.array([0.0, 0.0])
        action_arm = action_arm * cmd_arm_limit / action_scale
        # action_arm = np.array([0.0, 0.0, 0.01, 0, 0, 0])
        action_gripper = np.array([1])
        # print(action)
        # with open('../data/output_action.txt', 'a') as file:
        #     for row in action:
        #         line = ' '.join([f'{x:.6f}' for x in row.tolist()])
        #         file.write(line + '\n')

        action = np.concatenate([action_base, action_arm, action_gripper], axis=0)
        obs = mujoco_robot_env.run_simulation(action)

