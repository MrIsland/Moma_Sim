import sys
import numpy as np
import mujoco
from mujoco import viewer
import copy
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from utils.common import *

from utils.ik_arm_pybullet import *


class MomaMujoco:
    def __init__(self, xml_path, urdf_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.gravity[:] = [0.0, 0.0, 0.0]
        self.viewer = viewer.launch_passive(self.model, self.data)
        self.arm_freedom = 6

    def set_initial_joint_position(self, initial_qpos):
        self.data.qpos[:self.arm_freedom] = initial_qpos
        mujoco.mj_forward(self.model, self.data)

    def set_and_run_target_ee_pos(self, ee_pos):
        target_joint_state = self.eepos2jointstate(ee_pos)
        self.set_and_run_target_joint_state(target_joint_state)

    def set_and_run_target_joint_state(self, joint_state):
        # print('joint_state:     {}'.format(joint_state))
        self.run_simulation(joint_state)

    def eepos2jointstate(self, ee_pos):
        curr_joint_state = copy.deepcopy(self.data.qpos[:self.arm_freedom])
        # print('curr_joint_state:   {}'.format(curr_joint_state))
        # curr_joint_state = np.rad2deg(curr_joint_state)
        urdf_file = "./assets/island_moma_robot/urdf/rm_65_6f_description.urdf"
        ee_link_index = 5
        target_pos, target_rpy = ee_pos[:3], ee_pos[3:]
        target_ori = p.getQuaternionFromEuler(target_rpy)
        target_joint_state = solve_ik(urdf_file, ee_link_index, target_pos, target_ori)
        # print('curr_joint_state:   {}'.format(curr_joint_state))
        # print('target_joint_state:   {}'.format(target_joint_state))
        return target_joint_state

    # def run_simulation(self, target_js, step_per_target=1000):
    #     current_qpos = copy.deepcopy(self.data.qpos[:len(target_js)])
    #     delta_qpos = (target_js - current_qpos) / step_per_target
    #     print('max(delta_qpos):    {}'.format(delta_qpos.max()))
    #     for i in range(step_per_target+1):
    #         self.data.qpos[:len(current_qpos)] = delta_qpos * i + current_qpos
    #         # current_qpos = delta_qpos * i + current_qpos
    #         mujoco.mj_step(self.model, self.data)
    #         self.viewer.sync()

    def run_simulation(self, target_js, max_js_angle=2e-4):
        current_qpos = copy.deepcopy(self.data.qpos[:self.arm_freedom])
        # print('target_js:     {};   current_qpos:    {}'.format(np.rad2deg(target_js), np.rad2deg(current_qpos)))
        diff = target_js - current_qpos
        max_diff = np.abs(diff).max()

        # 根据最大差值和阈值1e-4，计算需要的步数
        step_per_target = int(np.ceil(max_diff / max_js_angle))
        # print(f"max diff: {max_diff:.6f}, step_per_target: {step_per_target}")

        # 避免step为0的情况
        step_per_target = max(step_per_target, 1)

        delta_qpos = diff / step_per_target
        # print('max(delta_qpos):    {}'.format(delta_qpos.max()))
        for i in range(step_per_target):
            self.data.qpos[:len(current_qpos)] = delta_qpos * i + current_qpos
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()

    def _close_moma_viewer(self):
        self.viewer.close()


if __name__ == '__main__':
    xml_path = './assets/island_moma_robot/xml/rm.xml'
    urdf_path = './assets/island_moma_robot/urdf/rm_65_6f_description.urdf'
    # initial_qpos = np.array([ 2.85731352, 0.41476004, -0.27790878, 0.25300293, -1.77417954, 0.00549779])
    # initial_qpos = np.array([172.253,-10.828,-28.204,-62.067,-113.751,-1.639])

    initial_qpos = np.array(
        [163.71200561523438, 23.763999938964844, -15.92300033569336, 14.496000289916992, -101.65299987792969,
         0.3149999976158142])
    moma_mujoco = MomaMujoco(xml_path, urdf_path)
    moma_mujoco.set_initial_joint_position(np.deg2rad(initial_qpos))
    traj_path = '/home/island/Desktop/mobile_manipulation/realrobot/data/pose.txt'
    # traj_path = '/home/island/Desktop/mobile_manipulation/realrobot/data/joint.txt'
    # traj_path = './data/pose.txt'
    js_list = read_data_list(traj_path)
    js_list = js_list + js_list[-2::-1]
    js_list_ln = len(js_list)
    # 模拟多个 target joint state 发送
    # for i, js in tqdm(enumerate(js_list)):
    #     tmp_js = [float(x) for x in js.strip().split()]
    #     moma_mujoco.set_and_run_target_joint_state(np.deg2rad(tmp_js))

    for i, js in tqdm(enumerate(js_list)):
        tmp_ee_pos = [float(x) for x in js.strip().split()]
        moma_mujoco.set_and_run_target_ee_pos(tmp_ee_pos)