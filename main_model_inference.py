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
    # mujoco_robot_env.run_simulation()
    cmd_arm_limit = np.array([0.05, 0.05, 0.05, 0.1, 0.1, 0.1])
    cmd_base_limit = np.array([0.3, 0.6])
    cnt = 0
    obs = None
    action_scale = config['asset']['arm_action_scale']
    while True:
        if cnt == 0:
            # 1. 第一帧的action
            action_base = np.array([0, 0])
            action_arm = np.array([0, 0., 0., 0, 0, 0])
            action_gripper = np.array([1])
        else:
            grasp_poses_factory = mujoco_robot_env.grasp_poses_factory
            eef_pos = mujoco_robot_env.states['eef_pos']
            eef_quat = mujoco_robot_env.states['eef_quat']

            # 2. Model Inference
            action = mujoco_robot_env.control_policy.get_action(obs)
            action = action.squeeze(0).cpu().detach().numpy()
            action_base, action_arm, action_gripper = action[:2], action[2:-1], np.array([action[-1]])
            action_base = action_base * cmd_base_limit
            action_arm = action_arm * cmd_arm_limit / action_scale
            action_gripper = np.array([1])

        action = np.concatenate([action_base, action_arm, action_gripper], axis=0)
        # print('action:      {}'.format(action))
        obs = mujoco_robot_env.run_simulation(action)
        cnt += 1

    print("end main")