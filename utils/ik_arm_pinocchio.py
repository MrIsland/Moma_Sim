import numpy as np
import pinocchio
from numpy.linalg import norm, solve
from scipy.spatial.transform import Rotation as R
DEBUG = False

class IKArm:
    def __init__(self, urdf_path: str, joint_id: int = 6):
        self.urdf_path = urdf_path
        self.model = pinocchio.buildModelFromUrdf(self.urdf_path)
        self.data = self.model.createData()
        self.joint_id = joint_id  # 通常为末端执行器的关节 ID

    def inverse_kinematics(self, current_q, target_dir, target_pos,
                           eps=1e-4, it_max=20000, dt=1e-2, damp=1e-8):
        """
        使用迭代法进行逆运动学求解。

        参数:
            current_q (np.ndarray): 当前关节角度 (n,)
            target_dir (np.ndarray): 目标方向的旋转矩阵 (3x3)
            target_pos (np.ndarray): 目标位置向量 (3,)
        返回:
            (List[float]): 计算出的关节角度解
        """
        oMdes = pinocchio.SE3(target_dir, np.array(target_pos))
        q = current_q.copy()
        i = 0
        flag = False
        while True:
            pinocchio.forwardKinematics(self.model, self.data, q)
            iMd = self.data.oMi[self.joint_id].actInv(oMdes)
            err = pinocchio.log(iMd).vector

            if norm(err) < eps:
                success = True
                break
            if i >= it_max:
                success = False
                break

            J = pinocchio.computeJointJacobian(self.model, self.data, q, self.joint_id)
            J = -np.dot(pinocchio.Jlog6(iMd.inverse()), J)
            v = -J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
            q = pinocchio.integrate(self.model, q, v * dt)

            i += 1

        if DEBUG:
            if not i % 10:
                print(f"{i}: error = {err.T}")
            if success:
                print("Convergence achieved!")
            else:
                print("Warning: the iterative algorithm did not converge.")

            print(f"\nresult: {q.flatten().tolist()}")
            print(f"\nfinal error: {err.T}")
        return q.flatten().tolist(), success


def rpy2matrix(rpy):

    """
    将 RPY 欧拉角（以弧度为单位）转换为 3x3 旋转矩阵。

    参数：
        roll: 绕 x 轴的旋转角（弧度）
        pitch: 绕 y 轴的旋转角（弧度）
        yaw: 绕 z 轴的旋转角（弧度）

    返回：
        3x3 numpy 数组，表示旋转矩阵
    """
    return R.from_euler('xyz', rpy).as_matrix()

def generate_random_ee_pose():
    poses = []
    # 随机位置
    x = np.random.uniform(0.2, 0.6)
    y = np.random.uniform(-0.4, 0.4)
    z = np.random.uniform(0.2, 0.6)
    pos = [x, y, z]

    # 随机旋转（小幅变化，基本朝下或前）
    rpy = np.random.uniform(low=[-0.5, -0.5, -np.pi], high=[0.5, 0.5, np.pi])
    rot_mat = R.from_euler('xyz', rpy).as_matrix()

    return pos, rpy

def matrix4x4_to_xyzrpy(matrix):
    r = matrix[:3, :3]
    t = matrix[:3, 3]
    r = R.from_matrix(r)
    rpy = r.as_euler('xyz')
    return t, rpy

def get_forward_kinecmatics(forward_q, robot):
    pinocchio.forwardKinematics(robot.model, robot.data, np.deg2rad(forward_q))
    ee_pos = robot.data.oMi[6]
    t, rpy = matrix4x4_to_xyzrpy(ee_pos.np)
    return t, rpy

if __name__ == "__main__":
    robot = IKArm('/home/island/Desktop/mobile_manipulation/realrobot/moma_mujoco/assets/island_moma_robot/urdf/rm_65_6f_description.urdf')

    for idx in range(len(robot.model.names)):
        print(f"ID: {idx}, Name: {robot.model.names[idx]}")

    forward_q = np.array([163.71200561523438, 23.763999938964844, -15.92300033569336, 14.496000289916992, -101.65299987792969, 0.3149999976158142])
    # forward_q = np.array([0, 0, 0, 0, 0, 0])
    # 前向运动学测试
    pinocchio.forwardKinematics(robot.model, robot.data, np.deg2rad(forward_q))
    print('???????????????   {}'.format(np.deg2rad(forward_q)))
    ee_pos = robot.data.oMi[6]
    t, rpy = matrix4x4_to_xyzrpy(ee_pos.np)
    print('translation:     {};  rpy:     {}'.format(t, rpy))

    print('forwardKinematics  !!!!')

    # 示例输入
    # current_q = np.array([50, 100, 20, 30, 40, 1])  # 初始关节角度为0
    current_q = np.zeros(robot.model.nq)  # 初始关节角度为0
    ori_pos, ori_rpy = generate_random_ee_pose()
    # print('current_q:   {}'.format(current_q))
    # target_dir = np.array([1.5, -2.0, 0.4])  # 目标方向为单位旋转矩阵
    # target_dir = rpy2matrix(target_dir)
    # target_dir = R.from_euler('xyz', [0, 0, 0]).as_matrix
    # theta = 0
    # target_dir = np.array([
    #     [1, 0, 0],
    #     [0, np.cos(theta), -np.sin(theta)],
    #     [0, np.sin(theta), np.cos(theta)]
    # ])
    # target_pos = [0.3, 0.2, 0.7]  # 目标位置
    target_pos = ori_pos.copy()
    target_dir = R.from_euler('xyz', ori_rpy).as_matrix()
    result_q, success = robot.inverse_kinematics(current_q, target_dir, target_pos)
    print('result_q:     {}'.format(result_q))
    pinocchio.forwardKinematics(robot.model, robot.data, np.array(result_q))
    ee_pose = robot.data.oMi[6]
    # 3. 拆分为位置和平移
    ee_pos = ee_pose.translation  # numpy 3D 向量
    ee_rot = ee_pose.rotation  # 3x3 旋转矩阵
    ee_rpy = R.from_matrix(ee_rot).as_euler('xyz', degrees=False)

    # print("EE Position:", ee_pos)
    # print("EE Rotation Matrix:\n", ee_rot)
    if success:
        print('final ee pose:    {}'.format(ee_pose))
        print('ori -  pos:    {}, rpy:    {}'.format(ori_pos, ori_rpy))
        print('fnl -  pos:    {}, rpy:    {}'.format(ee_pos, ee_rpy))
    else:
        print('solve_ik failing !!!')