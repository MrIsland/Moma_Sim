# @brief: 绘制曲线图,查看joint state/velocity的变化趋势.

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from plot_trends import plot_joint_trends
import pandas as pd


def read_data_list(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    lines = [line.strip() for line in lines]

    return lines

def plot_trends(data_file_path_joint, data_file_path_vel):
    data_joint = np.loadtxt(data_file_path_joint)
    data_vel = np.loadtxt(data_file_path_vel)

    t = np.arange(data_joint.shape[0])
    joints = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
    velocities = ['vx', 'wz']
    data_frame = {
        'joints': pd.DataFrame(data_joint[:, 7:13], columns=joints, index=t),
        'velocity': pd.DataFrame(data_vel[:, :2], columns=velocities, index=t),
    }
    summaries = plot_joint_trends(data_frame, kinds=['joints', 'velocity'], save_path=None, show=True)
    print('Returned summaries keys:  {}'.format(summaries.keys()))


if __name__ == '__main__':
    data_file_path_joint = '../test_vis/test_robot_pos.txt'
    data_file_path_vel = '../test_vis/test_action.txt'
    # data = read_data_list(data_file_path)
    plot_trends(data_file_path_joint, data_file_path_vel)
