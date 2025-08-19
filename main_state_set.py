import os, sys
import numpy as np
import time
from model.mujoco_robot_env import MujocoRobotEnv
from tqdm import tqdm
from utils.common import *

# if __name__ == '__main__':
#     cfg_path = './cfg'
#     cfg_yaml = 'config.yaml'
#     config = load_config(os.path.join(cfg_path, cfg_yaml))
#     mujoco_robot_env = MujocoRobotEnv(config)
#     # mujoco_robot_env.run_simulation()
#
#     txt_path = '/home/island/Desktop/mobile_manipulation/realrobot/data/pose.txt'
#     ee_pos_list = read_data_list(txt_path)
#     ee_pos_list = ee_pos_list + ee_pos_list[-2::-1]
#     ee_pos_list_ln = len(ee_pos_list)
#     cnt = 0
#     # deg = np.array([163.71200561523438, 23.763999938964844, -15.92300033569336, 14.496000289916992, -101.65299987792969, 0.3149999976158142])
#     # 2.85731, 0.41476, -0.27791, 0.25300, -1.77418, 0.00550
#
#     while True:
#         curr_data_indice = cnt % ee_pos_list_ln
#         curr_data = ee_pos_list[curr_data_indice]
#         curr_data = [float(x) for x in curr_data.strip().split()]
#
#         action_base = np.array([0, 0])
#         action_arm = np.array([0, 0, 0, 0, 0, 0])
#         # action_arm = curr_data - mujoco_robot_env.arm_ee_pos
#         # print('mujoco_robot_env.arm_ee_pos:    {}'.format(mujoco_robot_env.arm_ee_pos))
#         # print('action_arm:    {}'.format(action_arm))
#         action_gripper = np.array([1])
#         action = np.concatenate([action_base, action_arm, action_gripper], axis=0)
#
#         mujoco_robot_env.run_simulation(action)
#         cnt += 1
#         time.sleep(0.02)
#     print("end main")
#


if __name__ == '__main__':
    cfg_path = './cfg'
    cfg_yaml = 'config.yaml'
    config = load_config(os.path.join(cfg_path, cfg_yaml))
    mujoco_robot_env = MujocoRobotEnv(config)
    # mujoco_robot_env.run_simulation()

    txt_path = '/home/island/Desktop/mobile_manipulation/realrobot/data/test_robot_pos.txt'
    ee_pos_list = read_data_list(txt_path)
    ee_pos_list_ln = len(ee_pos_list)
    cnt = 0
    # deg = np.array([163.71200561523438, 23.763999938964844, -15.92300033569336, 14.496000289916992, -101.65299987792969, 0.3149999976158142])
    # 2.85731, 0.41476, -0.27791, 0.25300, -1.77418, 0.00550
    for i in range(ee_pos_list_ln):
        curr_data = ee_pos_list[i]
        curr_data = [float(x) for x in curr_data.strip().split()]
        mujoco_robot_env.run_simulation_set_state(curr_data)
        cnt += 1
        time.sleep(0.02)
    print("end main")