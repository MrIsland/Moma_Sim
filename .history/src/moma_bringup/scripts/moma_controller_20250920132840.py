#!/home/island/.conda/envs/rlgpu/bin/python3
import os, sys
import rospy
import numpy as np
import time
import rospkg
from tqdm import tqdm
from utils.common import *
import mujoco
from mujoco import viewer
from gymnasium.envs.mujoco.mujoco_rendering import OffScreenViewer
from scipy.spatial.transform import Rotation as R
from utils.ik_robot_controller import IKArmController
from moma_state_publisher import SyncAndPublish
from moma_bringup.msg import moma_cmd
import pybullet as p
import copy

class MomaController:
    def __init__(self, cfg):
        rospy.loginfo("[Master] Starting")
        rospy.Subscriber("/moma_cmd", moma_cmd, self.task_callback, queue_size=10)
        self.cfg = cfg
        self.asset_root = self.cfg['asset']['assetRoot']
        self.asset_moma = self.cfg['asset']['assetFileNameMoma']
        self.asset_arm = self.cfg['asset']['assetFileNameArm']
        self.asset_arm_urdf = self.cfg['asset']['assetFileNameArmURDF']
        moma_xml = os.path.join(self.asset_root, self.asset_moma)
        self.robot_model = mujoco.MjModel.from_xml_path(moma_xml)
        self.robot_data = mujoco.MjData(self.robot_model)
        joint_name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        self.arm_joint_ids = [self.robot_model.joint(name).id for name in joint_name]
        self.arm_joint_ids = self.robot_model.jnt_qposadr[self.arm_joint_ids]
        self.ee_body_id = self.robot_model.body('arm_link6').id
        self.viewer = viewer.launch_passive(self.robot_model, self.robot_data)
        self.cam_viewer = OffScreenViewer(self.robot_model, self.robot_data)
        self.momaDefaultDofPos = np.array(self.cfg['asset']['momaDefaultDofPos'])
        self.momaDefaultStatePos = np.array(self.cfg['asset']['momaDefaultStatePos'])
        self.ik_arm_controller = IKArmController(
            urdf_path=os.path.join(self.asset_root, self.asset_arm_urdf),
            end_effector_link_index=5,
            use_gui = False
        )
        self.action = None
        self.cnt = 0
        self.task_cnt = 0
        self.reset()
        print('Before the SyncAndPublish:     {}'.format(id(self)))

        self.sync_pub = SyncAndPublish(self)

    def task_callback(self, msg):
        if msg.data:
            # rospy.loginfo("[Master] Received new task")
            self.action = msg.data
            self.task_cnt += 1
            rospy.loginfo('self.task_cnt:     {}'.format(self.task_cnt))
               
    def render(self):
        self.viewer.sync()

    def _close_viewer(self):
        self.viewer.close()

    def _set_gripper_state(self, gripper_state):
        finger_joint_id = self._get_part_id('finger_joint', 'joint')
        right_kckle_joint_id = self._get_part_id('right_kckle_joint', 'joint')
        finger_joint_id = self.robot_model.jnt_qposadr[finger_joint_id]
        right_kckle_joint_id = self.robot_model.jnt_qposadr[right_kckle_joint_id]
        self.robot_data.qpos[finger_joint_id:right_kckle_joint_id+1] = np.array([gripper_state,
                                                                                    -1 * gripper_state,
                                                                                    gripper_state,
                                                                                    gripper_state,
                                                                                    -1 * gripper_state,
                                                                                    gripper_state])

    def _set_arm_delta_ee_pos(self, delta_ee_pos):
        # curr_joint_state = self.robot_data.qpos[self.arm_joint_ids]
        curr_joint_state = self.ik_arm_controller.get_joint_state()
        curr_pos, curr_quat = self.ik_arm_controller.get_ee_pos_quat_from_pybullet()
        # curr_quat = p.getQuaternionFromEuler(curr_orn)
        target_pos, target_quat = self.ik_arm_controller.compute_target_pos_mul(delta_ee_pos)
        delta_pos = target_pos - curr_pos
        # delta_quat = target_quat - curr_quat
        max_ee_pos = 0.001
        step = int(np.ceil(np.max(np.abs(delta_ee_pos)) / max_ee_pos))
        # print("step:", step)
        delta_ee_pos_interval = delta_pos / step
        # delta_quat_interval = delta_quat / step
        for i in range(step):
            tmp_ee_pos = curr_pos + (i + 1) * delta_ee_pos_interval
            # tmp_quat = curr_quat + i * delta_quat_interval
            tmp_quat = quaternion_slerp(curr_quat, target_quat, (i + 1) / step)
            result_joint_state = self.ik_arm_controller.compute_ik_without_mul(tmp_ee_pos, tmp_quat)
            self.ik_arm_controller.set_joint_state(result_joint_state)
            self.process_goal_joint_state(result_joint_state)
            # time.sleep(0.003)
            # if i == step-1 :
            #     print("after interval" , result_joint_state)
            # print("result_js:" , result_joint_state)
        # print('calculate target:     {}    {}'.format(target_pos, target_quat))
        # print('implement target:     {}'.format(self.ik_arm_controller.get_ee_pos_quat_from_pybullet()))
    
    def process_goal_joint_state(self, joint_state, max_js_angle=1e-3):
        self._set_arm_joint_pos(joint_state)
        
    def _set_arm_joint_pos(self, joint_state):
        arm_link1_id = self._get_part_id('joint1', 'joint')
        arm_link6_id = self._get_part_id('joint6', 'joint')
        arm_link1_id = self.robot_model.jnt_qposadr[arm_link1_id]
        arm_link6_id = self.robot_model.jnt_qposadr[arm_link6_id]
        self.robot_data.qpos[arm_link1_id:arm_link6_id+1] = joint_state
        mujoco.mj_fwdPosition(self.robot_model, self.robot_data)

    def _set_base_pos(self, base_pos):
        self.robot_data.qpos[:7] = base_pos

    def _set_base_qvel(self, base_qvel):
        self.robot_data.qvel[:] = 0
        self.robot_data.qacc[:] = 0
        quat = self.robot_data.qpos[3:7]
        rot_mat = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        v_robot = np.array([base_qvel[0], 0, 0])
        v_world = rot_mat @ v_robot
        vx_world, vy_world = v_world[0], v_world[1]
        self.robot_data.qvel[0] = vx_world
        self.robot_data.qvel[1] = vy_world
        self.robot_data.qvel[5] = base_qvel[1]
        
    def _move_arm(self, delta_ee_pos):
        # pos, orn = self.ik_arm_controller.get_ee_pos()
        # self._get_state()
        pos, orn = self.sync_pub.ee_pos[:3], self.sync_pub.ee_pos[3:]
        self.arm_joint_state = self.sync_pub.joint_state_data
        self.arm_ee_pos = np.concatenate([pos, orn])
        target_joint_state = self.ik_arm_controller.compute_ik(delta_ee_pos, self.arm_joint_state, self.arm_ee_pos)
        # print('target_joint_state:     {}'.format(target_joint_state))
        # self.ik_arm_controller.set_joint_state(target_joint_state)
        rospy.loginfo('set mujoco joint state:       {}'.format(target_joint_state))
        self.robot_data.qpos[self.arm_joint_ids] = target_joint_state
        rospy.loginfo('self.robot_data_qpos[self.arm_joint_ids]:       {}'.format(self.robot_data.qpos[self.arm_joint_ids]))

    def _move_arm_with_interpolation(self, delta_ee_pos):
        self._set_arm_delta_ee_pos(delta_ee_pos)

    def _get_part_id(self, part_name, type='GEOM'):
        part_id = None
        if type == 'GEOM' or type == 'geom':
            part_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_GEOM, part_name)
        elif type == 'BODY' or type == 'body':
            part_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, part_name)
        elif type == "JOINT" or type == 'joint':
            part_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_JOINT, part_name)
        elif type == 'CAM' or type == 'cam':
            part_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_CAMERA, part_name)
        elif type == 'SITE' or type == 'site':
            part_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_SITE, part_name)
        return part_id

    def depth_2_meters(self, depth):
        extend = self.robot_model.stat.extent
        near = self.robot_model.vis.map.znear * extend
        far = self.robot_model.vis.map.zfar * extend
        return near / (1 - depth * (1 - near / far))
            
    def reset(self):
        mujoco.mj_resetData(self.robot_model, self.robot_data)
        self._set_arm_joint_pos(self.momaDefaultDofPos[:6])
        self._set_gripper_state(1)
        self._set_base_pos(self.momaDefaultStatePos)
        mujoco.mj_forward(self.robot_model, self.robot_data)
        self.ik_arm_controller.set_joint_state(self.momaDefaultDofPos[:6])
        self.cnt = 0
        self.viewer.sync()

    def step(self, action):
        cmd_arm_limit = np.array([0.05, 0.05, 0.05, 0.1, 0.1, 0.1])
        cmd_base_limit = np.array([0.3, 0.6])
        base_qvel = action[:2]
        delta_ee_pos = action[2:-1]
        base_qvel = base_qvel * cmd_base_limit
        delta_ee_pos = delta_ee_pos * cmd_arm_limit / 25


        # base_qvel = base_qvel
        # delta_ee_pos = delta_ee_pos

        gripper_state = action[-1]
        # self._set_base_pos(self.momaDefaultStatePos)

        #### tmp write action to file ####
        # with open('./actions_recieve.txt', 'a') as file:
        #     action_tmp = copy.deepcopy(action)
        #     chassis_vel = action_tmp[:2]
        #     arm_joint_state = action_tmp[2:-1]
        #     gripper_state = action_tmp[-1]
        #     # chassis_vel = np.array(chassis_vel) / cmd_base_limit
        #     # arm_joint_state = np.array(arm_joint_state) / cmd_arm_limit * 25
        #     # print('chassis_vel:     {}'.format(chassis_vel))
        #     # print('arm_joint_state:  {}'.format(arm_joint_state))
        #     # print('gripper_state:     {}'.format(np.array([gripper_state])))
        #     action_final = np.concatenate([chassis_vel, arm_joint_state, np.array([gripper_state])])

        #     line = ' '.join([f'{x:.6f}' for x in action_final.tolist()])
        #     file.write(line + '\n')

        # with open('./joint_state.txt', 'a') as file:
        #     arm_js = self.robot_data.qpos[self.arm_joint_ids]
        #     line = ' '.join([f'{x:.6f}' for x in arm_js.tolist()])
        #     file.write(line + '\n')
        # print('1111111111   base_qvel:  {}'.format(base_qvel))
        self._set_base_qvel(base_qvel)
        # print('2222222222   delta_ee_pos:  {}'.format(delta_ee_pos))
        self._move_arm(delta_ee_pos)
        # self._move_arm_with_interpolation(delta_ee_pos)
        self._set_gripper_state(1)
        mujoco.mj_forward(self.robot_model, self.robot_data)
        mujoco.mj_step(self.robot_model, self.robot_data)
        rospy.loginfo('mujoco.mj_step     {}'.format(self.robot_data.qpos[self.arm_joint_ids]))
        # self._get_state()
        # ####### TODO:  _get_observation() 
        # obs = self._get_observation()
        # return obs
    
    def run_simulation(self):
        self.step(self.action)
        self.render()
        self.cnt += 1
        rospy.loginfo('self.cnt:      {}'.format(self.cnt))
        # return obs
    
    def _get_observation(self):
        return None

    def _get_image(self):
        # 确保 MuJoCo 状态更新
        self.cam_viewer.make_context_current()
        rgb = self.cam_viewer.render(
            render_mode='rgbd_tuple', camera_id=0, segmentation=False
        )
        depth = self.cam_viewer.render(
            render_mode='depth_array', camera_id=0,
        )
        seg = self.cam_viewer.render(
            render_mode='rgbd_tuple', camera_id=0, segmentation=True
        )
        depth = self.depth_2_meters(depth)
        mask = (depth <= 5.0)
        depth = np.where(mask, depth, 0)
        return rgb, depth, seg
        
    def _get_state(self):
        joint_state = self.robot_data.qpos[self.arm_joint_ids]
        # joint_state = self.ik_arm_controller.get_joint_state()
        curr_pos, curr_orn = self.ik_arm_controller.get_ee_pos_rpy_from_pybullet(curr_joint_state=joint_state)
        ee_pos = np.concatenate([curr_pos, curr_orn])  # 修正concatenate语法
        rgb, depth , _ = self._get_image()
        quat = self.robot_data.qpos[3:7]
        # print('self.robot_data.qpos:     {}'.format(self.robot_data.qpos[:7]))
        rot_mat = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()

        rot_inverse = np.linalg.inv(rot_mat)
        vx_world = self.robot_data.qvel[0] 
        vy_world = self.robot_data.qvel[1]
        v_world = [vx_world, vy_world, 0.0]  # 添加z分量，使其成为3D向量
        base_qvel = [0.0, self.robot_data.qvel[5]]  # 直接创建包含值的列表
        v_robot = rot_inverse @ np.array(v_world)  # 现在维度匹配了
        base_qvel[0] = v_robot[0]  # 只使用x分量
        
        chassis_vel = base_qvel  # 使用修正后的base_qvel
        odometry = self.robot_data.qpos[:7]
        rospy.loginfo('get_state:       {}'.format(joint_state))
        return joint_state, rgb, depth, chassis_vel, odometry, ee_pos
    
    def run_simulation_set_state(self, state):
        # base_state = state[:7]
        base_state = state[:7]
        self.robot_data.qvel[:] = 0
        arm_dof = state[-6:]
        # base_state = np.array([-0.5, -0.5, 0, 0, 0, 0, 1])
        # arm_dof = np.array([0, 0, 0, 0, 0, 0])
        gripper_state = 1
        base_pos = np.array(base_state[:3])
        base_quat = np.roll(base_state[3:], 1)
        # self._set_base_pos(self.momaDefaultStatePos)
        self._set_base_pos(np.hstack([base_pos, base_quat]))
        self._set_arm_joint_pos(arm_dof)
        self._set_gripper_state(gripper_state)

        mujoco.mj_forward(self.robot_model, self.robot_data)
        mujoco.mj_step(self.robot_model, self.robot_data)
        # mujoco.mj_rnePostConstraint(self.robot_model, self.robot_data)
        self.render()
        self.cnt += 1


######  Set action
if __name__ == '__main__':
    cfg_path = rospkg.RosPack().get_path('moma_bringup')
    cfg_path = os.path.join(cfg_path, '../../')
    cfg_yaml = 'assets/config/config.yaml'
    config = load_config(os.path.join(cfg_path, cfg_yaml))
    rospy.init_node('master_controller')
    rospy.loginfo("[Master] Ready.")
    moma_controller = MomaController(config)
    rate = rospy.Rate(60)
    while not rospy.is_shutdown():
        if moma_controller.action is not None:
            moma_controller.run_simulation()
            moma_controller.action = None  # 处理完后清空
        rate.sleep()


######  Set state
# if __name__ == '__main__':
#     cfg_path = rospkg.RosPack().get_path('moma_bringup')
#     cfg_path = os.path.join(cfg_path, '../../')
#     cfg_yaml = 'assets/config/config.yaml'
#     config = load_config(os.path.join(cfg_path, cfg_yaml))
#     rospy.init_node('master_controller')
#     rospy.loginfo("[Master] Ready.")
#     moma_controller = MomaController(config)
#     # rate = rospy.Rate(60)
#     txt_path = '/home/island/Desktop/mobile_manipulation/realrobot/data/test_robot_pos.txt'
#     ee_pos_list = read_data_list(txt_path)
#     ee_pos_list_ln = len(ee_pos_list)
#     # while not rospy.is_shutdown():
#     for i in range(ee_pos_list_ln):
#         curr_data = ee_pos_list[i]
#         curr_data = [float(x) for x in curr_data.strip().split()]
#         moma_controller.run_simulation_set_state(curr_data)
        # cnt += 1

