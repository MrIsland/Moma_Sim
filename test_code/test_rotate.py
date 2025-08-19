import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R

def visualize_rotation_difference(quat1, quat2):
    """
    可视化两个四元数之间的旋转差异（使用Open3D）
    :param quat1: 四元数1，格式 [x, y, z, w]
    :param quat2: 四元数2，格式 [x, y, z, w]
    """
    # 构造两个旋转矩阵
    rot1 = R.from_quat(quat1).as_matrix()
    rot2 = R.from_quat(quat2).as_matrix()

    # 创建两个坐标系原点
    frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    frame1.rotate(rot1, center=(0, 0, 0))
    frame1.translate((0, 0, 0))  # 原点

    frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    frame2.rotate(rot2, center=(0, 0, 0))
    frame2.translate((0.3, 0, 0))  # 向右平移

    # 显示
    o3d.visualization.draw_geometries([frame1, frame2])


quat1 = [-1.09e-04, -4.04e-05,  7.08e-01, -7.06e-01]
quat2 = [0, 0, 0, 1]
# quat1 = [ -1.0934e-04, -4.0359e-05, 7.0804e-01, -7.0617e-01]
# quat2 = [ 4.0230e-05, -1.0934e-04, 7.0673e-01,  7.0748e-01]
visualize_rotation_difference(quat1, quat2)