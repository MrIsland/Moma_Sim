import pybullet as p
import pybullet_data
import numpy as np


def solve_ik(urdf_path, end_effector_link_index, target_position, target_orientation=None):
    """
    使用 pybullet 的 calculateInverseKinematics 进行逆运动学求解（无可视化）

    参数:
    - urdf_path: URDF 文件路径（相对路径或绝对路径）
    - end_effector_link_index: 末端执行器对应的 link index（int）
    - target_position: 目标末端执行器的位置 [x, y, z]
    - target_orientation: 可选，目标末端姿态四元数 [x, y, z, w]；如不提供只考虑位置

    返回:
    - joint_angles: 求解得到的每个关节的角度列表
    """
    # 启动 pybullet 的 DIRECT 模式
    physics_client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # 用于找 plane.urdf 等内置资源

    # 加载 URDF 模型（初始位置为原点）
    robot_id = p.loadURDF(urdf_path, useFixedBase=True)

    # 获取关节数量（忽略固定关节）
    num_joints = p.getNumJoints(robot_id)
    movable_joints = [i for i in range(num_joints) if p.getJointInfo(robot_id, i)[2] != p.JOINT_FIXED]

    # IK 求解
    if target_orientation is not None:
        joint_angles = p.calculateInverseKinematics(robot_id, end_effector_link_index, target_position, target_orientation,
                                                    restPoses=np.deg2rad(np.array([172.253,-10.828,-28.204,-62.067,-113.751,-1.639])),
                                                    maxNumIterations=100, residualThreshold=1e-4, solver = p.IK_DLS)
    else:
        joint_angles = p.calculateInverseKinematics(robot_id, end_effector_link_index, target_position)

    # 断开连接
    p.disconnect()

    # 返回对应的非固定关节角度
    return joint_angles[:len(movable_joints)]


# ==== 示例使用 ====

if __name__ == "__main__":
    # 使用 KUKA 机械臂 URDF 示例（你也可以换成自己的 URDF 路径）
    urdf_file = "/home/island/Desktop/mobile_manipulation/realrobot/moma_mujoco/assets/island_moma_robot/urdf/rm_65_6f_description.urdf"  # pybullet_data 内置的

    # 设置末端执行器 link index（对于 kuka 是 6）
    ee_link_index = 5

    # 设置目标位置
    target_pos = [-0.034167, -0.031189, 0.671426]

    # 设置目标姿态（四元数），也可以设置为 None
    target_ori = p.getQuaternionFromEuler([2.756, -1.494, -2.791])

    # 求解
    joint_angles = solve_ik(urdf_file, ee_link_index, target_pos, target_ori)

    print("求解得到的关节角为（弧度）:")
    print(np.round(joint_angles, 4))
    print("求解得到的关节角为（角度）:")
    print(np.round(np.rad2deg(joint_angles), 4))
