import os, sys
import numpy as np
import yaml
import torch
import open3d as o3d

def read_data_list(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    lines = [line.strip() for line in lines]

    return lines


def load_config(path, default_path=None):
    """
    Loads config file.
    Args:
        path (str): path to config file.
        default_path (str, optional): whether to use default path. Defaults to None.
    Returns:
        cfg (dict): config dict.
    """
    # load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.full_load(f)

    # check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')
    package_name = cfg_special.get('package_name')
    # if yes, load this config first as default
    # if no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.full_load(f)
    else:
        cfg = dict()

    # include main configuration
    update_recursive(cfg, cfg_special)

    # if package_name is not None:
    #     rospack = rospkg.RosPack()
    #     pkg_path = rospack.get_path(package_name)
    #     resolve_relative_paths(cfg, pkg_path)

    return cfg


def update_recursive(dict1, dict2):
    """
    Update two config dictionaries recursively.
    Args:
        dict1 (dict): first dictionary to be updated.
        dict2 (dict): second dictionary which entries should be used.
    """
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


def resolve_relative_paths(cfg, base_path):
    """
    Recursively resolve relative paths in cfg using base_path.

    Any value that is a string and starts with ./ or ../ or is just a relative path
    will be replaced by an absolute path based on base_path.
    """
    for k, v in cfg.items():
        if isinstance(v, str):
            if not os.path.isabs(v):
                # Skip ROS topic names like '/camera/image_raw'
                if v.startswith('./') or v.startswith('../'):
                    cfg[k] = os.path.normpath(os.path.join(base_path, v))
        elif isinstance(v, dict):
            resolve_relative_paths(v, base_path)
        elif isinstance(v, list):
            for i in range(len(v)):
                if isinstance(v[i], str) and not os.path.isabs(v[i]) and not v[i].startswith('/'):
                    v[i] = os.path.normpath(os.path.join(base_path, v[i]))
                elif isinstance(v[i], dict):
                    resolve_relative_paths(v[i], base_path)

def unsqueeze_obs(obs):
    if type(obs) is dict:
        for k, v in obs.items():
            obs[k] = unsqueeze_obs(v)
    else:
        if len(obs.size()) > 1 or obs.size()[0] > 1:
            obs = obs.unsqueeze(0)
    return obs


def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action =  action * d + m
    return scaled_action

def to_torch(x, dtype=torch.float, device='cuda:0', requires_grad=False):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)

def rotation_matrix_to_axis_angle(matrix):
    epsilon = 1e-6
    angle = np.arccos(  np.clip(np.trace(matrix) - 1, -1, 1) / 2.0)
    return angle

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

def mov(tensor, device):
    return torch.from_numpy(tensor.cpu().numpy()).to(device)


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