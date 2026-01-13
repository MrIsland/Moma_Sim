import os, sys
import pdb
import random

import numpy as np
from collections.abc import Sequence, Iterable
from numbers import Number
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
from isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, calc_heading_quat_inv, quat_to_tan_norm, \
    my_quat_rotate
from type_common import str_to_dtype, is_arr, is_dict, is_seq_of, is_type, scalar_type, is_str
from scipy.spatial.transform import Rotation as R
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import open3d as o3d
import matplotlib.pyplot as plt

def astype(x, dtype):
    if dtype is None:
        return x
    assert is_arr(x) and is_str(dtype), (type(x), type(dtype))
    if is_arr(x, 'np'):
        return x.astype(str_to_dtype(dtype, 'np'))
    elif is_arr(x, 'torch'):
        return x.to(str_to_dtype(dtype, 'torch'))
    elif is_type(dtype):
        return dtype(x)
    else:
        raise NotImplementedError(f"As type {type(x)}")

def to_torch(x, dtype=None, device=None, non_blocking=False):
    import torch
    if x is None:
        return None
    elif is_dict(x):
        return {k: to_torch(x[k], dtype, device, non_blocking) for k in x}
    elif is_seq_of(x):
        return type(x)([to_torch(y, dtype, device, non_blocking) for y in x])

    if isinstance(x, torch.Tensor):
        ret = x.detach()
    elif isinstance(x, (Sequence, Number)):
        ret = torch.from_numpy(np.array(x))
    elif isinstance(x, np.ndarray):
        ret = torch.from_numpy(x)
    else:
        raise NotImplementedError(f"{x} {dtype}")
    if device is not None:
        ret = ret.to(device, non_blocking=non_blocking)
    if dtype is not None:
        ret = astype(ret, dtype)
    return ret

def mov(tensor, device):
    return torch.from_numpy(tensor.cpu().numpy()).to(device)

def rotation_matrix_to_axis_angle(matrix):
    epsilon = 1e-6
    angle = np.arccos(np.clip(np.trace(matrix) - 1, -1, 1) / 2.0)
    return angle


def rotation_matrix_to_rad(matrix):
    epsilon = 1e-6
    angle = np.arccos(np.clip(np.trace(matrix) - 1, -1, 1) / 2.0)
    return np.deg2rad(angle)

def rotation_matrix_to_axis_angle_batch(R):
    """R: (N, 3, 3) —> 返回 shape (N,) 角度"""
    trace = np.trace(R, axis1=1, axis2=2)
    cos_theta = np.clip((trace - 1) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_theta)
    return angle

def normalize(quat):
    norm = np.linalg.norm(quat)
    return quat / norm

def slerp(q1, q2, t, ensure_shortest_path=True):
    # 首先确保两个四元数都是归一化的
    q1 = normalize(q1)
    q2 = normalize(q2)

    # 计算两个四元数之间的点积
    cos_omega = np.dot(q1, q2)

    # 如果确保选择最短路径且点积为负，则反转其中一个四元数
    if ensure_shortest_path and cos_omega < 0.0:
        q2 = -q2
        cos_omega = -cos_omega

    # 如果两个四元数几乎平行，直接使用线性插值
    if cos_omega > 0.95:
        return normalize((1.0 - t) * np.array(q1) + t * np.array(q2))

    omega = np.arccos(cos_omega)
    so = np.sin(omega)

    weighted_quat = (np.sin((1.0 - t) * omega) / so) * np.array(q1) + (np.sin(t * omega) / so) * np.array(q2)

    return normalize(weighted_quat)

def random_z_rotation_quaternion(batch_size=1, device='cpu'):
    """
    生成绕 Z 轴的随机旋转四元数（batch 支持）。

    返回: Tensor shape [batch_size, 4]，格式为 [x, y, z, w]
    """
    theta = torch.rand(batch_size, device=device) * 2 * math.pi  # [0, 2π)

    # 绕 Z 轴的四元数: q = [x, y, z, w] = [0, 0, sin(θ/2), cos(θ/2)]
    half_theta = theta / 2
    sin_half = torch.sin(half_theta)
    cos_half = torch.cos(half_theta)

    quaternions = torch.stack([
        torch.zeros_like(theta),  # x
        torch.zeros_like(theta),  # y
        sin_half,                 # z
        cos_half                  # w
    ], dim=1)

    return quaternions

def average_pos(pos, weights):
    assert len(pos) == len(weights)
    weights = np.array(weights)
    _weights = weights / np.sum(weights)
    result = np.sum(pos * _weights[:, np.newaxis], axis=0)
    return result

def average_quaternions(matrixs, weights):
    """
    输入:
        quaternions: N x 4 numpy array, 每一行是一个单位四元数 [x, y, z, w]
        weights: N 元素的列表或数组，对应每个四元数的权重
    输出:
        一个单位四元数 [x, y, z, w]，表示加权平均结果
    """
    assert len(matrixs) == len(weights)
    quats = []
    for mat in matrixs:
        quat = R.from_matrix(mat).as_quat()  # 转为四元数 (x, y, z, w)
        quat = quat / np.linalg.norm(quat)  # 归一化（可选，确保单位）
        quats.append(quat)
    # 确保所有四元数归一化
    weights = np.array(weights)
    _weights = weights / np.sum(weights)

    # 选择第一个四元数为参考
    q_ref = R.from_quat(quats[0])
    q_ref_inv = q_ref.inv()

    # Log map
    log_vecs = []
    for q in quats:
        q_i = R.from_quat(q)
        delta = q_ref_inv * q_i
        log_vec = delta.as_rotvec()
        log_vecs.append(log_vec)

    # 加权平均
    log_vecs = np.array(log_vecs)
    mean_log = np.average(log_vecs, axis=0, weights=_weights)

    # Exp map
    delta_rot = R.from_rotvec(mean_log)
    mean_q = (q_ref * delta_rot).as_quat()

    return mean_q / np.linalg.norm(mean_q)  # 返回单位四元数

def voxel_downsample(pcd, voxel_size):
    # pcd: (N, 7) tensor -> [x, y, z, r, g, b, label]
    coords = pcd[:, :3]
    colors_labels = pcd[:, 3:]  # (N, 4)

    voxel_coords = torch.floor(coords / voxel_size).int()  # (N, 3)

    keys, inverse_indices = torch.unique(voxel_coords, return_inverse=True, dim=0)

    # 3. 对每个 voxel 做平均池化（你也可以改成 random 选择一个点）
    down_coords = torch.zeros((keys.shape[0], 3), device=pcd.device)
    down_colors_labels = torch.zeros((keys.shape[0], 4), device=pcd.device)

    for i in range(keys.shape[0]):
        mask = (inverse_indices == i)
        down_coords[i] = coords[mask].mean(dim=0)
        down_colors_labels[i] = colors_labels[mask].mean(dim=0)  # label 会被平均（你可以改成众数）

    return torch.cat([down_coords, down_colors_labels], dim=1)  # (M, 7)


def draw_camera(extrinsic, size=0.2, color=[1, 0, 0]):
    """
    画一个相机框（视锥体），给定相机的外参矩阵。
    extrinsic: 4x4 numpy array，camera-to-world 变换矩阵
    """
    # 相机坐标系下的5个点：光心 + 成像平面4个角点
    points = np.array([
        [0, 0, 0],                      # 相机原点
        [-1, -0.75, 1],                 # 左下
        [1, -0.75, 1],                  # 右下
        [1, 0.75, 1],                   # 右上
        [-1, 0.75, 1]                   # 左上
    ]) * size

    # 线连接：光心到4角 + 成像平面边框
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1]
    ]

    # 将点从相机坐标变换到世界坐标
    points_hom = np.hstack((points, np.ones((5, 1))))
    points_world = (extrinsic @ points_hom.T).T[:, :3]

    # 构造 LineSet
    cam_lines = o3d.geometry.LineSet()
    cam_lines.points = o3d.utility.Vector3dVector(points_world)
    cam_lines.lines = o3d.utility.Vector2iVector(lines)
    cam_lines.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return cam_lines

def load_urdf_with_txt(data_path, obj_txt, scale_txt, height_txt):
    # 读取txt文件中列出的物体名称
    with open(obj_txt, 'r', encoding='utf-8') as f:
        urdf_list = ['{}.urdf'.format(line.strip()) for line in f.readlines() if line.strip()]

    with open(scale_txt, 'r', encoding='utf-8') as f:
        scale_list = [float(line.strip()) for line in f.readlines() if line.strip()]

    with open(height_txt, 'r', encoding='utf-8') as f:
        height_list = [float(line.strip()) for line in f.readlines() if line.strip()]
    # for root, _, files in os.walk(data_path):
    #     for file in files:
    #         if file.endswith('.urdf') and file[:-5] in obj_names:
    #             urdf_list.append(file)
    return urdf_list, scale_list, height_list

def load_urdf(data_path):
    urdf_list = []
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith('.urdf'):
                urdf_list.append(file)
    return urdf_list

def sample_from_regions_np(regions, n_samples=10):
    # 转成 NumPy 数组，每行是 [x_min, x_max, y_min, y_max]
    arr = np.array([[r["x"][0], r["x"][1], r["y"][0], r["y"][1]] for r in regions])

    # 随机选择区域索引
    idx = np.random.choice(len(arr), size=n_samples)

    # 取出每个样本对应的区域
    chosen = arr[idx]  # shape (n_samples, 4)

    # 生成随机数并映射到对应区间
    xs = np.random.rand(n_samples) * (chosen[:, 1] - chosen[:, 0]) + chosen[:, 0]
    ys = np.random.rand(n_samples) * (chosen[:, 3] - chosen[:, 2]) + chosen[:, 2]

    return xs, ys

def pose_to_matrix(pos, quat):
    rot = R.from_quat(quat.cpu().numpy())  # xyzw
    mat = torch.eye(4)
    mat[:3, :3] = torch.tensor(rot.as_matrix())
    mat[:3, 3] = pos
    return mat

def quat_to_rotmat_batch(quat):
    """批量四元数 -> 旋转矩阵 (num_envs, 3, 3)"""
    # 归一化
    from torch.nn.functional import normalize
    quat = normalize(quat, dim=-1)
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    rot = torch.stack([
        1 - 2 * (yy + zz),  2 * (xy - wz),      2 * (xz + wy),
        2 * (xy + wz),      1 - 2 * (xx + zz),  2 * (yz - wx),
        2 * (xz - wy),      2 * (yz + wx),      1 - 2 * (xx + yy)
    ], dim=-1).reshape(-1, 3, 3)
    return rot


def pose_to_matrix_batch(pos, quat):
    """批量 pos+quat -> 4x4 变换矩阵"""
    num_envs = pos.shape[0]
    rot = quat_to_rotmat_batch(quat)  # (num_envs, 3, 3)
    mat = torch.eye(4, device=pos.device).unsqueeze(0).repeat(num_envs, 1, 1)
    mat[:, :3, :3] = rot
    mat[:, :3, 3] = pos
    return mat


def rotmat_to_euler_xyz_batch(rot):
    sy = torch.sqrt(rot[:, 0, 0]**2 + rot[:, 1, 0]**2)
    singular = sy < 1e-6
    x = torch.atan2(rot[:, 2, 1], rot[:, 2, 2])
    y = torch.atan2(-rot[:, 2, 0], sy)
    z = torch.atan2(rot[:, 1, 0], rot[:, 0, 0])
    x[singular] = torch.atan2(-rot[singular, 1, 2], rot[singular, 1, 1])
    y[singular] = torch.atan2(-rot[singular, 2, 0], sy[singular])
    z[singular] = 0
    return torch.stack([x, y, z], dim=-1)


#@brief: opengl和opencv相机坐标下的点云转换，注意是行向量的点云
def opengl_trans_opencv(points):
    T_gl2cv = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ], dtype=np.float32)

    return points @ to_torch(T_gl2cv, device=points.device)

#@brief: opengl和opencv相机坐标下的点云转换，注意是行向量的点云
def opengl_trans_opencv_np(points):
    T_gl2cv = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0,  -1]
    ], dtype=np.float32)

    return points @ T_gl2cv

def get_quat_180(input_quat):
    rot_x_180 = torch.tensor([[1.0, 0.0, 0.0, 0.0]],
                             device=input_quat.device, dtype=input_quat.dtype).repeat(input_quat.shape[0], 1)
    output_quat = quat_mul(input_quat, rot_x_180)

    # #### vis
    # R1 = R.from_quat(input_quat.cpu().numpy()).as_matrix()
    # R2 = R.from_quat(output_quat.cpu().numpy()).as_matrix()
    #
    # frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    # frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    # frame1.rotate(R1.squeeze(), center=(0, 0, 0))
    # frame2.rotate(R2.squeeze(), center=(0, 0, 0))
    #
    # frame2.translate([0.5, 0.0 ,0.0])
    # o3d.visualization.draw_geometries([frame1, frame2])
    # pdb.set_trace()

    return output_quat

def pre_process_rgb_img(img_batch, img_h, img_w):
    img_batch = img_batch.permute(0, 3, 1, 2)
    _, _, h, w = img_batch.shape
    left = (w - h) // 2
    img_batch = img_batch[:, :, :, left:left+h]
    img_batch = F.interpolate(img_batch, size=(img_h, img_w), mode='bilinear', align_corners=False)

    img_batch = img_batch.permute(0, 2, 3, 1)
    return img_batch  # (num_env, 224, 224, 3)

def low_pass_filter(cur_data, filter_data, process_buf, alpha):
    mask = process_buf.unsqueeze(-1)
    filtered_velocity = alpha * filter_data + (1 - alpha) * cur_data
    final_velocity = torch.where(mask == 0, cur_data, filtered_velocity)
    return final_velocity

def accumulate_reward(name, reward_tensor, progress_buf, reward_dict):
    reward_dict[name].index_add_(0, progress_buf, reward_tensor.float())
    reward_dict['{}_count'.format(name)].index_add_(0, progress_buf, torch.ones_like(progress_buf).float())


def debug_vis_draw_scalar(writer, reward_dict, frame_idx):
    """
    将 reward_dict 中的统计数据转换为折线图并写入 TensorBoard。

    Args:
        writer: TensorBoard 的 SummaryWriter 实例
        reward_dict: 包含累积 reward 和计数 count 的字典
        frame_idx: 当前的 global_step
    """
    # 1. 过滤出需要画图的数据 key（排除以 _count 结尾的 key）
    data_keys = [k for k in reward_dict.keys() if not k.endswith('_count')]

    # 2. 创建 Matplotlib Figure
    # figsize 可以根据需要调整，(12, 8) 比较清晰
    fig, ax = plt.subplots(figsize=(12, 8))
    subplots = []
    has_valid_data = False

    # 3. 遍历每个 reward 项进行处理和绘制
    for name in data_keys:
        count_name = name + '_count'

        # 确保对应的 count 存在
        if count_name not in reward_dict:
            continue

        # 获取 Tensor 数据并转为 CPU numpy 数组
        # 假设数据是 shape 为 (max_episode_length,) 的一维向量
        sum_vals = reward_dict[name].detach().cpu()
        counts = reward_dict[count_name].detach().cpu()

        # 避免除以 0：创建一个掩码，只有 count > 0 的地方才计算平均值
        valid_mask = counts > 0

        # 如果该 reward 没有任何记录，则跳过
        if not valid_mask.any():
            continue

        has_valid_data = True

        # 计算平均值： mean = sum / count
        # 初始化为 0 或 NaN
        means = torch.zeros_like(sum_vals)
        means[valid_mask] = sum_vals[valid_mask] / counts[valid_mask]

        # 转换为 numpy 以便绘图
        x_axis = np.arange(len(means))
        y_axis = means.numpy()

        # 绘制折线
        # 只绘制 count > 0 的部分，或者绘制整条线（未经历的 step 默认为 0）
        # 这里选择绘制整条线，并在 label 中显示该 reward 的名称
        ax.plot(x_axis, y_axis, label=name, linewidth=2, alpha=0.8)

        #### subplot
        sub_fig, sub_ax = plt.subplots(figsize=(8, 5))

        sub_ax.plot(x_axis, y_axis, label=f'{name}', linewidth=2)

        # 标出一些关键信息
        sub_ax.set_title(f'Curve: {name} (Step {frame_idx})')
        sub_ax.set_xlabel('Episode Step')
        sub_ax.set_ylabel('Mean Reward')
        sub_ax.grid(True, linestyle='--', alpha=0.5)
        sub_ax.legend()

        # --- 写入 TensorBoard ---
        # 使用 f-string 将其分类到 Analysis/Separate 目录下
        writer.add_figure(f'Analysis/Separate_Curves/{name}', sub_fig, global_step=frame_idx)

        # 重要：关闭当前图表以释放内存
        plt.close(sub_fig)


    # 4. 设置图表格式
    if has_valid_data:
        ax.set_title(f'Average Reward Curves over Episode Steps (Step {frame_idx})')
        ax.set_xlabel('Episode Step (Time)')
        ax.set_ylabel('Mean Reward Value')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='best', fontsize='small')  # 自动放置图例

        # 5. 写入 TensorBoard
        # 使用 add_figure 而不是 add_scalar
        writer.add_figure('Analysis/Episode_Reward_Curves', fig, global_step=frame_idx)

    # 关闭图形以释放内存，防止在训练循环中内存泄漏
    plt.close(fig)
