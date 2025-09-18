#!/home/demo/anaconda3/envs/moma/bin/python3
import pybullet as p
import pybullet_data
import numpy as np
from scipy.spatial.transform import Rotation as R
import pdb
class IKArmController:
    def __init__(self, urdf_path, end_effector_link_index,use_gui=False):
        """
        初始化逆运动学控制器

        参数:
        - urdf_path: URDF 文件路径（相对路径或绝对路径）
        - end_effector_link_index: 末端执行器对应的 link index（int）
        """
        self.urdf_path = urdf_path
        self.end_effector_link_index = end_effector_link_index
        
        # 添加：保存上一次计算的目标位姿
        self.last_target_ee_pos = None
        
        self.physics_client = p.connect(p.GUI if use_gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # 用于找 plane.urdf 等内置资源
        p.setGravity(0, 0, 0)  # 设置重力
        # 加载 URDF 模型（初始位置为原点）
        self.plane = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0], useFixedBase=True)

        # 获取关节数量（忽略固定关节）
        self.num_joints = p.getNumJoints(self.robot_id)
        self.movable_joints = [i for i in range(self.num_joints) if p.getJointInfo(self.robot_id, i)[2] != p.JOINT_FIXED]
        
        self.joint_indices = []
        self.joint_limits_lower = []
        self.joint_limits_upper = []
        self.joint_ranges = []
        
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_type = joint_info[2]
            if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                self.joint_indices.append(i)
                self.joint_limits_lower.append(joint_info[8])
                self.joint_limits_upper.append(joint_info[9])
                self.joint_ranges.append(joint_info[9] - joint_info[8])
    
    def get_ee_pos_from_pybullet(self, curr_joint_state):
        for i,j_idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, j_idx, curr_joint_state[i])
        link_state = p.getLinkState(self.robot_id, self.end_effector_link_index)
        current_pos = np.array(link_state[4])
        current_quat = np.array(link_state[5])  # 四元数
        current_orn = p.getEulerFromQuaternion(current_quat)
        return current_pos, current_orn
    
    def get_ee_pos(self):
        """ 获取末端执行器位置和四元数姿态 """
        link_state = p.getLinkState(self.robot_id, self.end_effector_link_index)
        pos = np.array(link_state[4])
        orn = np.array(link_state[5])  # 四元数
        orn_rpy = p.getEulerFromQuaternion(orn)
        return pos, orn_rpy

    def compute_target_pos_mul(self, delta_pos, curr_joint_state):
        # print(delta_pos)
        current_pos,current_orn = self.get_ee_pos_from_pybullet(curr_joint_state)
        # print(f"current_pos:{current_pos},current_orn:{current_orn}")
        R_curr = R.from_euler('xyz', current_orn)
        R_delta = R.from_euler('xyz', delta_pos[3:])
        # import pdb; pdb.set_trace()
        target_quat = (R_delta * R_curr).as_quat()
        target_pos = current_pos + delta_pos[:3]
        # print(f"target_pos:{target_pos},target_orn:{R.from_quat(target_quat).as_euler('xyz')}")
        return target_pos, target_quat
    
    def compute_ik(self, delta_pos, curr_joint_state):
        """读取delta eepos"""
        
        # 使用上一次计算的target位姿替代传入的curr_ee_pos（如果存在）
        # if self.last_target_ee_pos is not None:
        #     actual_curr_ee_pos = self.last_target_ee_pos
        # else:
        #     actual_curr_ee_pos = curr_ee_pos
        # target_pos, target_quat = self.compute_target_pos_mul(delta_pos, actual_curr_ee_pos)
        
        target_pos, target_quat = self.compute_target_pos_mul(delta_pos, curr_joint_state)
        # for i,j_idx in enumerate(self.joint_indices):
            # p.resetJointState(self.robot_id, j_idx, np.deg2rad(curr_joint_state[i]))
        # IK 求解
        # rest_pose = np.deg2rad(np.array(curr_joint_state))
        rest_pose = np.array(curr_joint_state)
        joint_angles = p.calculateInverseKinematics(self.robot_id, 
                                                    self.end_effector_link_index, 
                                                    target_pos, 
                                                    targetOrientation=target_quat,
                                                    lowerLimits=self.joint_limits_lower,
                                                    upperLimits=self.joint_limits_upper,
                                                    jointRanges=self.joint_ranges,
                                                    restPoses=rest_pose.tolist(),
                                                    maxNumIterations=1000, 
                                                    residualThreshold=1e-3, solver = p.IK_DLS)
        # print('##########################################################')
        # print('curr_joint_state:          {}'.format(curr_joint_state))
        # print('target_pos:    {};   target_quat:    {}'.format(target_pos, target_quat))
        # print('target_joint_state:        {}'.format(joint_angles[:len(self.movable_joints)]))
        print('joint_angles:     {};    len()'.format(joint_angle))
        return joint_angles[:len(self.movable_joints)]
    
    def compute_ik_without_mul(self, target_pos, target_quat, curr_joint_state):
        """读取delta eepos"""
        for i,j_idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, j_idx, np.deg2rad(curr_joint_state[i]))
        # IK 求解
        rest_pose = np.deg2rad(np.array(curr_joint_state))
        joint_angles = p.calculateInverseKinematics(self.robot_id, 
                                                    self.end_effector_link_index, 
                                                    target_pos, 
                                                    targetOrientation=target_quat,
                                                    lowerLimits=self.joint_limits_lower,
                                                    upperLimits=self.joint_limits_upper,
                                                    jointRanges=self.joint_ranges,
                                                    restPoses=rest_pose.tolist(),
                                                    maxNumIterations=1000, 
                                                    residualThreshold=1e-3, solver = p.IK_DLS)
        return joint_angles[:len(self.movable_joints)]


if __name__ == '__main__':
    ik_arm_controller = IKArmController(
        urdf_path="/home/demo/Desktop/island_moma_robot/code/src/Moma/moma_bringup/asset/urdf/rm_65_6f_description.urdf",
        end_effector_link_index=5
    )
    delta_pos = np.array([-1.00000005e-03,  7.51810003e-05,  2.53879989e-04,  1.66579604e-03,
        2.00000009e-03,  2.00000009e-03])
    curr_joint_state = np.array([-38.380001068115234, 0.0, 45.82899856567383, -0.0010000000474974513, 77.89900207519531, 180.00100708007812])
    curr_ee_pos = np.array([-0.223167, 0.176763, 0.553317, 3.141, -0.982, 2.471])
    joint_angle = ik_arm_controller.compute_ik(delta_pos, curr_joint_state, curr_ee_pos)
    # print('joint_angle;    {}'.format(joint_angle))
