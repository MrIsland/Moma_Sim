import pybullet as p
import pybullet_data
import numpy as np
import time
from scipy.spatial.transform import Rotation as R


class IKARMController:
    def __init__(self, urdf_path, ee_link_index, fixed_joints=None, use_gui=True):
        """
        urdf_path: 机器人 URDF 文件路径
        ee_link_index: 末端执行器 link 的索引
        fixed_joints: 需要固定的关节索引列表（可选）
        use_gui: 是否使用 GUI
        """
        self.physics_client = p.connect(p.GUI if use_gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)

        # 加载平面 & 机器人
        self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF(urdf_path, basePosition=[0, 0, 0], useFixedBase=True)

        self.ee_link_index = ee_link_index
        self.fixed_joints = fixed_joints or []

        self.joint_indices = []
        self.joint_limits_lower = []
        self.joint_limits_upper = []
        self.joint_ranges = []
        self.rest_poses = []

        # 获取可动关节信息
        for i in range(p.getNumJoints(self.robot)):
            joint_info = p.getJointInfo(self.robot, i)
            joint_type = joint_info[2]
            if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                self.joint_indices.append(i)
                self.joint_limits_lower.append(joint_info[8])
                self.joint_limits_upper.append(joint_info[9])
                self.joint_ranges.append(joint_info[9] - joint_info[8])
                self.rest_poses.append((joint_info[8] + joint_info[9]) / 2)

    def set_joint_state(self, init_joint_angles=None):
        """ 设置机器人初始状态 """
        if init_joint_angles is None:
            init_joint_angles = np.zeros(len(self.joint_indices))

        for i, j_idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot, j_idx, init_joint_angles[i])

    def get_joint_state(self):
        """ 获取当前关节角度 """
        joint_states = p.getJointStates(self.robot, self.joint_indices)
        return np.array([s[0] for s in joint_states])

    def get_ee_pose(self):
        """ 获取末端执行器位置和四元数姿态 """
        link_state = p.getLinkState(self.robot, self.ee_link_index)
        pos = np.array(link_state[4])
        orn = np.array(link_state[5])  # 四元数
        orn_rpy = p.getEulerFromQuaternion(orn)
        return pos, orn_rpy

    def get_ee_pose_quat(self):
        link_state = p.getLinkState(self.robot, self.ee_link_index)
        pos = np.array(link_state[4])
        quat = np.array(link_state[5])  # 四元数
        return pos, quat


    # def compute_target_pose_mul(self, delta_pos):
    #     """
    #     curr_pos: (3,) numpy数组，当前EE位置
    #     curr_orn: (3,) numpy数组，当前EE欧拉角 (xyz顺序，单位弧度)
    #     delta_pos: (6,) numpy数组，增量，前3为平移，后3为欧拉角 (xyz顺序，单位弧度)
    #
    #     返回：
    #         pos_new: (3,) 新位置
    #         orn_new: (3,) 新欧拉角 (xyz顺序)
    #     """
    #
    #     curr_pos, curr_quat = self.get_ee_pose_quat()
    #     R_curr = R.from_quat(curr_quat).as_matrix()  # 3x3
    #     R_delta = R.from_euler('xyz', delta_pos[3:]).as_matrix()  # 3x3
    #     T_curr = np.eye(4)
    #     T_curr[:3, :3] = R_curr
    #     T_curr[:3, 3] = curr_pos
    #     T_delta = np.eye(4)
    #     T_delta[:3, :3] = R_delta
    #     T_delta[:3, 3] = delta_pos[:3]
    #     T_new = T_delta @ T_curr
    #     pos_new = T_new[:3, 3]
    #     R_new = T_new[:3, :3]
    #     orn_new = R.from_matrix(R_new).as_quat()
    #     return pos_new, orn_new

    def compute_target_pose_mul(self, delta_pos, curr_ee_pos = None):
        """
        curr_pos: (3,) numpy数组，当前EE位置
        curr_orn: (3,) numpy数组，当前EE欧拉角 (xyz顺序，单位弧度)
        delta_pos: (6,) numpy数组，增量，前3为平移，后3为欧拉角 (xyz顺序，单位弧度)

        返回：
            pos_new: (3,) 新位置
            orn_new: (3,) 新欧拉角 (xyz顺序)
        """
        if curr_ee_pos is None:
            curr_pos, curr_quat = self.get_ee_pose_quat()
        else:
            curr_pos = curr_ee_pos[:3]
            curr_quat = R.from_euler('xyz', curr_ee_pos[3:]).as_quat()
        R_curr = R.from_quat(curr_quat)
        R_delta = R.from_euler('zxy', delta_pos[3:])
        target_quat = (R_delta * R_curr).as_quat()
        target_pos = curr_pos + delta_pos[:3]
        return target_pos, target_quat

    def compute_target_pose(self, delta_pos):
        """
        根据 delta ee pos 计算目标末端位置
        delta_pos: [dx, dy, dz]
        """
        curr_pos, curr_orn = self.get_ee_pose()
        target_pos = curr_pos + delta_pos[:3]
        target_orn = curr_orn + delta_pos[3:]

        return target_pos, target_orn

    def compute_ik(self, delta_pos):
        """
        计算逆运动学
        输入: delta ee pos
        输出: 目标关节角
        """
        target_pos, target_orn = self.compute_target_pose_mul(delta_pos)
        joint_angles = p.calculateInverseKinematics(
            self.robot,
            self.ee_link_index,
            target_pos,
            targetOrientation=target_orn,
            lowerLimits=self.joint_limits_lower,
            upperLimits=self.joint_limits_upper,
            jointRanges=self.joint_ranges,
            restPoses=self.get_joint_state().tolist(),
            # restPoses=rest_pos,
            maxNumIterations=2000,
            residualThreshold=1e-7,
            solver = p.IK_DLS
            # currentPositions=self.get_joint_state()
        )
        return np.array(joint_angles)

    def compute_ik_input(self, delta_pos, curr_joint_state, curr_ee_pos):
        """
        计算逆运动学
        输入: delta ee pos
        输出: 目标关节角
        """
        target_pos, target_orn = self.compute_target_pose_mul(delta_pos, curr_ee_pos)
        curr_joint_state = np.deg2rad(curr_joint_state)
        joint_angles = p.calculateInverseKinematics(
            self.robot,
            self.ee_link_index,
            target_pos,
            targetOrientation=target_orn,
            lowerLimits=self.joint_limits_lower,
            upperLimits=self.joint_limits_upper,
            jointRanges=self.joint_ranges,
            restPoses=curr_joint_state.tolist(),
            # restPoses=rest_pos,
            maxNumIterations=2000,
            residualThreshold=1e-7,
            solver = p.IK_DLS
            # currentPositions=self.get_joint_state()
        )

        return np.array(joint_angles)

    def move_arm(self, joint_angles):
        """ 发送关节角目标 """
        for i, j_idx in enumerate(self.joint_indices):
            p.setJointMotorControl2(
                self.robot,
                j_idx,
                p.POSITION_CONTROL,
                targetPosition=joint_angles[i],
                force=200
            )

    def step_simulation(self, steps=100):
        for _ in range(steps):
            p.stepSimulation()
            time.sleep(0.01)

    def close(self):
        p.disconnect()


# if __name__ == "__main__":
#     # ✅ 使用 KUKA 机械臂测试
#     robot_urdf = '/home/island/Desktop/mobile_manipulation/realrobot/moma_mujoco/assets/island_moma_robot/urdf/rm_65_6f_description.urdf'
#     robot = IKARMController(robot_urdf, ee_link_index=5)
#     init_js = np.array([-0.67, 0.0, 0.8, 0.0, 1.36, 1.57])
#     robot.set_joint_state(init_js)
#     for _ in range(1000):
#         delta = np.array([0.0, 0.0, 0.01, 0.0, 0.0, 0.0])  # 末端向上移动
#         joint_angles = robot.compute_ik(delta)
#         robot.move_arm(joint_angles)
#         robot.step_simulation()
#
#     robot.close()

if __name__ == '__main__':
    robot_urdf = '/home/island/Desktop/mobile_manipulation/realrobot/moma_mujoco/assets/island_moma_robot/urdf/rm_65_6f_description.urdf'
    robot = IKARMController(robot_urdf, ee_link_index=5, use_gui=False)
    # zhenji
    # delta_pos = np.array([0.001, -0.001,  0.001,  0.002,  0.002,  0.002])
    # curr_joint_state = np.array([-38.380001068115234, 0.0, 45.82899856567383, -0.0010000000474974513, 77.89900207519531, 180.00100708007812])
    # curr_ee_pos = np.array([-0.223167, 0.176763, 0.553317, 3.141, -0.982, 2.471])

    # igrape mujoco
    # delta_pos = np.array([0.001, -0.001,  0.001,  0.002,  0.002,  0.002])
    # curr_joint_state = np.rad2deg(np.array([-0.67, 0.0, 0.8, 0.0, 1.36, 3.14]))
    # print('curr_joint_state:      {}'.format(curr_joint_state))
    # curr_ee_pos = np.array([-0.23049106, 0.18260613, 0.54694819, 3.1392061956769806, -0.9815835886948071, 2.474462428073723])

    # island mujoco
    delta_pos = np.array([0.001, -0.001,  0.001,  0.002,  0.002,  0.002])
    curr_joint_state = np.rad2deg(np.array([-0.67, 0.0, 0.8, 0.0, 1.36, 3.14]))
    curr_ee_pos = np.array([-0.23049106, 0.18260613, 0.54694819, 3.1392061956769806, -0.9815835886948071, 2.474462428073723])
    # curr_ee_pos = np.array([-0.23049, 0.18261, 0.54695]
    robot.set_joint_state(init_joint_angles=np.array([-0.67, 0.0, 0.8, 0.0, 1.36, 3.14]))
    joint_angle = robot.compute_ik_input(delta_pos, curr_joint_state, curr_ee_pos)
    print('joint_angle;    {}'.format(joint_angle))
