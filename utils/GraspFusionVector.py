"""
这个文件中的grasp fusion是对于同一个voxel中的grasp pos进行fusion。
"""
import os, sys
import pdb

import numpy as np
from queue import Queue
import time
# from .interval_tree import RedBlackTree, Node, BLACK, RED, NIL
# from .interval_tree import RedBlackTree, Node, BLACK, RED, NIL
import random
import math
import torch
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'third_party', 'graspnet-baseline', 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'third_party', 'graspnet-baseline', 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'third_party', 'graspnet-baseline', 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from graspnet import GraspNet, pred_decode
from graspnetAPI import GraspGroup
from collision_detector import ModelFreeCollisionDetector
import open3d as o3d
import copy
from common import rotation_matrix_to_axis_angle, slerp, voxel_downsample, draw_camera, rotation_matrix_to_axis_angle_batch
from scipy.spatial.transform import Rotation as R

class Grasp6d:
    def __init__(self, grasp_pos, grasp_ori, score, grasp_group_array):
        self.grasp_pos = grasp_pos
        self.grasp_ori = grasp_ori

        self.score = score
        self.fused_count = 0

        self.branch_array = [None, None, None, None, None, None, None, None]
        self.branch_distance = np.full((8), 15)
        self.frame_id = 0
        self.grasp_group_array = grasp_group_array
        # self.grasp_group_array = self.update(grasp_pos, grasp_ori, score, grasp_group_array)


    def update(self):
        self.grasp_group_array[0] = self.score
        self.grasp_group_array[13:16] = self.grasp_pos
        self.grasp_group_array[4:4+9] = self.grasp_ori.reshape(-1)


class UnionFind:
    def __init__(self):
        self.parent = dict()
        self.size = dict()

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        # 按秩合并（rank 低的合到高的）
        if self.rank[px] < self.rank[py]:
            self.parent[px] = py
        elif self.rank[px] > self.rank[py]:
            self.parent[py] = px
        else:
            self.parent[py] = px
            self.rank[px] += 1

    def get_clusters(self):
        clusters = defaultdict(set)
        for node in self.parent:
            root = self.find(node)
            clusters[root].add(node)
        return clusters

class GraspPosDetector:
    def __init__(self, args):
        self.args = args
        self.ckpt_path = self.args['ckpt_path']
        self.num_point = self.args['num_point']
        self.num_view = self.args['num_view']
        self.collision_thresh = self.args['collision_threshold']
        self.voxel_size = self.args['voxel_size']

        self.net = self._get_net()

    def _get_net(self):
        # Init the model
        net = GraspNet(input_feature_dim=0, num_view=self.num_view, num_angle=12, num_depth=4,
                       cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01, 0.02, 0.03, 0.04], is_training=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        # Load checkpoint
        checkpoint = torch.load(self.ckpt_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print("-> loaded checkpoint %s (epoch: %d)" % (self.ckpt_path, start_epoch))
        # set model to eval mode
        net.eval()
        return net

    def _get_and_process_data(self, frame_pcd):
        # convert data
        color_masked = frame_pcd[:, 3:6].cpu().numpy()
        cloud_masked = frame_pcd[:, 0:3].cpu().numpy()
        if len(cloud_masked) >= self.num_point:
            idxs = np.random.choice(len(cloud_masked), self.num_point, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_point - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]
        # convert data
        cloud = o3d.geometry.PointCloud()
        cloud.colors = o3d.utility.Vector3dVector(color_sampled.astype(np.float32))
        cloud.points = o3d.utility.Vector3dVector(cloud_sampled.astype(np.float32))
        # o3d.visualization.draw_geometries([cloud])
        end_points = dict()
        cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cloud_sampled = cloud_sampled.to(device)
        end_points['point_clouds'] = cloud_sampled
        end_points['cloud_colors'] = color_sampled
        return end_points, cloud

    def vis_grasp_pos(self, gg, cloud, with_camera=True):
        if gg is None:
            return
        gg.nms()
        gg.sort_by_score()
        # gg = gg[:64]
        gg = gg
        # pdb.set_trace()
        grippers = gg.to_open3d_geometry_list()
        rotation_matrices = gg.rotation_matrices
        translations = gg.translations
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrices[0]
        transform[:3, 3] = translations[0]
        axis.transform(transform)

        if with_camera:

            o3d.visualization.draw_geometries([cloud, *grippers, axis])
        else:
            o3d.visualization.draw_geometries([cloud, *grippers, axis])

    def _get_grasps(self, net, end_points):
        with torch.no_grad():
            end_points = net(end_points)
            grasp_preds = pred_decode(end_points)
        gg_array = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(gg_array)
        return gg

    def collision_detection(self, gg, cloud):
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=self.voxel_size)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.collision_thresh)
        gg = gg[~collision_mask]
        return gg

    def _get_grasp_prediction(self, frame_pcd):
        # frame_pcd: tensor(num_pcd, 7)  [x, y, z, r, g, b, label]
        end_points, cloud = self._get_and_process_data(frame_pcd)
        gg = self._get_grasps(self.net, end_points)
        if self.collision_thresh > 0:
            gg = self.collision_detection(gg, np.array(cloud.points))
        gg = gg.sort_by_score()
        return gg, cloud

class GraspFusionVector:
    def __init__(self, args):
        self.args = args
        # self.x_rb_tree = RedBlackTree(self.args["interval_size"])
        # self.y_rb_tree = RedBlackTree(self.args["interval_size"])
        # self.z_rb_tree = RedBlackTree(self.args["interval_size"])

        self.scene_node = set()   # 当前存有的所有grasp pos
        self.fusion_step = 0
        self.connected_num = 0
        self.score_total = 0
        self.fused_total = 0
        self.grasp_pos_id = 0

        self.grasp_pos_detector = GraspPosDetector(self.args['graspnet'])
        self.union_set = UnionFind()
        self.pcd_list = []

    def reset_gltree(self):
        # del self.x_rb_tree, self.y_rb_tree, self.z_rb_tree, self.scene_node
        #
        # self.x_rb_tree = RedBlackTree(self.args["interval_size"])
        # self.y_rb_tree = RedBlackTree(self.args["interval_size"])
        # self.z_rb_tree = RedBlackTree(self.args["interval_size"])

        self.scene_node = set()
        self.fusion_step = 0
        self.connected_num = 0
        self.score_total = 0
        self.fused_total = 0

    def add_new_frame(self, frame_pcd, frame_valid, frame_idx, target_label):
        self.fusion_step += 1

        ### 1. 获取目标点云
        target_pcd = self._get_target_pcd(frame_pcd, frame_valid, target_label)
        if target_pcd is None:
            # print('Current frame do not have the target point cloud')
            return None

        ### 2. 从target_pcd中预测grasp pos
        frame_grasp_group, frame_cloud = self.grasp_pos_detector._get_grasp_prediction(target_pcd)

        ### 3. 将本帧预测到的grasp pos进行fusion，包括了对grasp pos的筛选
        per_image_node_set = self.fusion(frame_grasp_group, frame_cloud, self.fusion_step)

        ### 4. 可视化
        tmp_frame_grasp_group = self._get_gg_from_cur_scene_node(self.scene_node)

        self.grasp_pos_detector.vis_grasp_pos(tmp_frame_grasp_group, frame_cloud)
        return frame_cloud

    def fusion(self, frame_grasp_group, frame_cloud, frame_idx):
        ### 1. 对已经存在的grasp pos和要加入的grasp pos进行向量化
        curr_scene_node_gg = self._get_gg_from_cur_scene_node(self.scene_node)
        curr_scene_node_gg_array = curr_scene_node_gg.grasp_group_array if curr_scene_node_gg is not None else None
        # todo: 这里加一个shuffle
        new_gg_array = frame_grasp_group.grasp_group_array

        ### 2: add new grasps vector version
        # per_image_node_list = self.valid_grasping_identification(curr_scene_node_gg_array, new_gg_array, frame_idx)
        per_image_node_list = self.valid_grasping_identification(new_gg_array, new_gg_array, frame_idx)

        ### 3: 返回出来的应该是要加入的新的grasp pos vector, 将其加入到self.scene_node之中
        per_image_node_set = self.add_new_grasps(per_image_node_list)
        return per_image_node_set

    def valid_grasping_identification(self, curr_scene_node_gg_array, new_gg_array, frame_idx, min_cluster_size=5):
        per_image_node_list = []

        new_pos = new_gg_array[:, 13:16]
        new_ori = new_gg_array[:, 4:4+9].reshape(-1, 3, 3)
        new_scores = new_gg_array[:, 0]
        ### todo 1. 检测哪些新加的grasp pos能够被融合。 [直接融合进新的scene_node]
        # if curr_scene_node_gg_array is not None:
        #     old_pos = curr_scene_node_gg_array[:, 13:16]
        #     old_ori = curr_scene_node_gg_array[:, 4:4+9].reshape(-1, 3, 3)
        #     old_score = curr_scene_node_gg_array[:, 0]
        #
        #     N_new, N_old = new_pos.shape[0], old_pos.shape[0]
        #
        #     rel_rot = np.einsum('nik,mkj->nmij', new_ori, old_ori.transpose(0, 2, 1))  # (N_new, N_old, 3)
        #     angles = rotation_matrix_to_axis_angle_batch(rel_rot.reshape(-1, 3, 3)).reshape(N_new, N_old)
        #
        #     angle_mask = angles < self.args['min_ori_threshold']
        #     near_mask = np.linalg.norm(new_pos[:, None, :] - old_pos[None, :, :], axis=2) < self.args['min_octree_threshold']
        #     fuse_mask = angle_mask & near_mask
        #     fused = fuse_mask.any(axis=1)
        #     remain_idx = np.where(~fused)[0]
        #     fusion_idx = np.where(fused)[0]
        #     # fusion
        #     for i in fusion_idx:
        #         score, pos, quat = new_scores[i], new_pos[i], R.from_matrix(new_ori[i]).as_quat()
        #         for j in np.where(fuse_mask[i])[0]:
        #             fusion_old_score, fusion_old_pos, fusion_old_quat = old_score[j], old_pos[j], R.from_matrix(old_ori[j]).as_quat()
        #             weight = score / (fusion_old_score + score)
        #             fused_pos = (1 - weight) * fusion_old_pos + weight * pos
        #             fused_quat = slerp(quat, fusion_old_quat, 1 - weight)
        #             fused_score = score + fusion_old_score
        #             curr_scene_node_gg_array[j][0], curr_scene_node_gg_array[j][13:16], curr_scene_node_gg_array[j][4:4+9] \
        #                 = fused_score, fused_pos, R.from_quat(fused_quat).as_matrix().reshape(-1)
        #             break
        # else:
        #     remain_idx = np.arange(new_ori.shape[0])
        #
        # if len(remain_idx) == 0:
        #     return []
        remain_idx = np.arange(new_ori.shape[0])

        ### todo 2. 检测哪些新加的grasp pos能够与之前grasp pos有连接。 [增加到待加入list中去]
        if curr_scene_node_gg_array is not None:
            old_pos = curr_scene_node_gg_array[:, 13:16]
            old_ori = curr_scene_node_gg_array[:, 4:4+9].reshape(-1, 3, 3)
            old_score = curr_scene_node_gg_array[:, 0]

            remain_scene_node_gg_array = new_gg_array[remain_idx]
            remain_pos = remain_scene_node_gg_array[:, 13:16]
            remain_ori = remain_scene_node_gg_array[:, 4:4+9].reshape(-1, 3, 3)
            remain_scores = remain_scene_node_gg_array[:, 0]

            N_new, N_old = remain_pos.shape[0], old_pos.shape[0]

            rel_rot = np.einsum('nik,mkj->nmij', remain_ori, old_ori.transpose(0, 2, 1))  # (N_new, N_old, 3)
            angles = rotation_matrix_to_axis_angle_batch(rel_rot.reshape(-1, 3, 3)).reshape(N_new, N_old)
            angle_mask = angles < self.args['min_ori_threshold']
            D_old_new = np.linalg.norm(new_pos[:, None, :] - old_pos[None, :, :], axis=2)
            D_new_new = np.linalg.norm(new_pos[:, None, :] - new_pos[None, :, :], axis=2)


            print('in the todo 2')


        ### todo 3. 内部融合或连接
        ### todo 3+. 剩下的grasp pos中，相互之间连接小于5。 [判断剩下的可以保留的以及删除的]

        return per_image_node_list  # 返回最终能够保留下来的node list，以grasp_group_array的形式

    def add_new_grasps(self, per_image_node_list):
        per_image_node_set = set()
        ### todo 1. 通过list生成node

        return per_image_node_set




    def _get_target_pcd(self, frame_pcd, frame_valid, target_label):
        valid_mask = frame_valid.squeeze() > 0  # shape: (N,)

        if target_label is not None:
            label_mask = frame_pcd[:, 6] == target_label
            combined_mask = valid_mask & label_mask
        else:
            combined_mask = valid_mask

        pcd_valid = frame_pcd[combined_mask]  # shape: (M, 7)

        if pcd_valid.shape[0] == 0:
            return None
        return pcd_valid

    def _get_gg_from_cur_scene_node(self, scene_node):
        list_scene_node = list(scene_node)
        all_grasp_group_array = []
        curr_gg = None
        for node in list_scene_node:
            tmp_grasp_group_array = node.grasp_group_array
            all_grasp_group_array.append(tmp_grasp_group_array)

        if all_grasp_group_array:
            curr_gg = np.array(all_grasp_group_array)
        if curr_gg is not None:
            curr_gg = GraspGroup(curr_gg)

        return curr_gg

    def _generate_o3d_pcd(self, target_pcd):
        pcd = target_pcd[:, :3]
        color = target_pcd[:, 3:6]
        final_pcd = o3d.geometry.PointCloud()
        final_pcd.points = o3d.utility.Vector3dVector(pcd.cpu().numpy())
        final_pcd.colors = o3d.utility.Vector3dVector(color.cpu().numpy())
        return final_pcd

    def vis_pcd(self, frame_pcd, frame_valid, target_label):

        # Step 1: 过滤有效点
        valid_mask = frame_valid.squeeze() > 0  # shape: (N,)

        # Step 2: 若指定了标签，则进一步过滤
        if target_label is not None:
            label_mask = frame_pcd[:, 6] == target_label
            combined_mask = valid_mask & label_mask
        else:
            combined_mask = valid_mask

        pcd_valid = frame_pcd[combined_mask]  # shape: (M, 7)

        if pcd_valid.shape[0] == 0:
            print("无有效点可视化")
            return

        # Step 3: 提取坐标和颜色
        xyz = pcd_valid[:, :3].cpu().numpy()  # (M, 3)
        rgb = pcd_valid[:, 3:6].cpu().numpy()  # (M, 3)
        rgb = np.clip(rgb, 0.0, 1.0)

        # Step 4: 构造 Open3D 点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)

        # Step 5: 可视化
        window_title = f"Point Cloud - Label {target_label}" if target_label is not None else "Point Cloud"
        o3d.visualization.draw_geometries([pcd], window_name=window_title, width=800, height=600)


##############################################################################
#### test code ####

from common import to_torch, mov

def mov(tensor, device):
    return torch.from_numpy(tensor.cpu().numpy()).to(device)

def _depth_image_to_point_cloud_GPU_batch(
        camera_depth_tensor_batch, camera_rgb_tensor_batch,
        camera_seg_tensor_batch, camera_view_matrix_inv_batch,
        camera_proj_matrix_batch, u, v, width: float, height: float,
        depth_bar: float, device: torch.device,
        # z_p_bar: float = 3.0,
        # z_n_bar: float = 0.3,
):

    batch_num = camera_depth_tensor_batch.shape[0]

    depth_buffer_batch = mov(camera_depth_tensor_batch, device)
    rgb_buffer_batch = mov(camera_rgb_tensor_batch, device) / 255.0
    seg_buffer_batch = mov(camera_seg_tensor_batch, device)

    # Get the camera view matrix and invert it to transform points from camera to world space
    vinv_batch = camera_view_matrix_inv_batch

    # Get the camera projection matrix and get the necessary scaling
    # coefficients for deprojection

    proj_batch = camera_proj_matrix_batch
    fu_batch = 2 / proj_batch[:, 0, 0]
    fv_batch = 2 / proj_batch[:, 1, 1]

    centerU = width / 2
    centerV = height / 2

    Z_batch = depth_buffer_batch

    Z_batch = torch.nan_to_num(Z_batch, posinf=1e10, neginf=-1e10)
    X_batch = -(u.view(1, u.shape[-2], u.shape[-1]) - centerU) / width * Z_batch * fu_batch.view(-1, 1, 1)
    Y_batch = (v.view(1, v.shape[-2], v.shape[-1]) - centerV) / height * Z_batch * fv_batch.view(-1, 1, 1)

    R_batch = rgb_buffer_batch[..., 0].view(batch_num, 1, -1)
    G_batch = rgb_buffer_batch[..., 1].view(batch_num, 1, -1)
    B_batch = rgb_buffer_batch[..., 2].view(batch_num, 1, -1)
    S_batch = seg_buffer_batch.view(batch_num, 1, -1)

    valid_depth_batch = Z_batch.view(batch_num, -1) > -depth_bar

    Z_batch = Z_batch.view(batch_num, 1, -1)
    X_batch = X_batch.view(batch_num, 1, -1)
    Y_batch = Y_batch.view(batch_num, 1, -1)
    O_batch = torch.ones((X_batch.shape), device=device)

    position_batch = torch.cat((X_batch, Y_batch, Z_batch, O_batch, R_batch, G_batch, B_batch, S_batch), dim=1)
    # (b, N, 8)
    position_batch = position_batch.permute(0, 2, 1)
    position_batch[..., 0:4] = position_batch[..., 0:4] @ vinv_batch

    points_batch = position_batch[..., [0, 1, 2, 4, 5, 6, 7]]
    valid_batch = valid_depth_batch  # * valid_z_p_batch * valid_z_n_batch

    return points_batch, valid_batch

def _get_target_pcd(frame_pcd, frame_valid, target_label):
    valid_mask = frame_valid.squeeze() > 0  # shape: (N,)

    if target_label is not None:
        label_mask = frame_pcd[:, 6] == target_label
        combined_mask = valid_mask & label_mask
    else:
        combined_mask = valid_mask

    pcd_valid = frame_pcd[combined_mask]  # shape: (M, 7)

    if pcd_valid.shape[0] == 0:
        return None
    return pcd_valid

if __name__ == '__main__':
    import cv2, yaml
    path = '/home/island/Desktop/mobile_manipulation/IsaacGymEnvs/isaacgymenvs/examples/output'
    rgb_path = os.path.join(path, 'rgb_001.png')
    depth_path = os.path.join(path, 'depth_001.png')
    seg_path = os.path.join(path, 'seg_001.png')
    # 读取 RGB 图像（HWC, BGR）
    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0  # (3, H, W), [0, 1]

    # 读取 Depth 图像（单通道，通常是16位或32位）
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Depth image not found at {depth_path}")
    depth_tensor = torch.from_numpy(depth).unsqueeze(0).float()  # (1, H, W)

    # 读取 Segmentation 图像（单通道或三通道）
    seg = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
    if len(seg.shape) == 3:  # 如果是彩色分割图，转换成灰度
        seg = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
    seg_tensor = torch.from_numpy(seg).unsqueeze(0).long()  # (1, H, W), 可作为label

    # 放到 CUDA（cuda:0）
    device = torch.device("cuda:0")
    rgb_tensor = rgb_tensor.permute(1, 2, 0).to(device) * 255.
    depth_tensor = depth_tensor.permute(1, 2, 0).to(device) / 255.
    seg_tensor = seg_tensor.permute(1, 2, 0).to(device)
    vinv_matrix = torch.tensor([[0, -1, 0, 0],
                                [0, 0, 1, 0],
                                [-1, 0, 0, 0],
                                [0.5, 0, 0.5, 1]]).to(device)
    proj_matrix = torch.tensor([[1.7321, 0, 0, 0],
                                [0, 2.3094, 0, 0],
                                [0, 0, 0, -1],
                                [0, 0, 1e-3, 0]]).to(device)
    width, height = 640, 480
    camera_u = torch.arange(0, width)
    camera_v = torch.arange(0, height)
    camera_v2, camera_u2 = torch.meshgrid(camera_v, camera_u, indexing='ij')
    camera_u2 = to_torch(camera_u2, device=device)
    camera_v2 = to_torch(camera_v2, device=device)
    frame_pcd, frame_valid = _depth_image_to_point_cloud_GPU_batch(depth_tensor.squeeze(-1).unsqueeze(0),
                                                                   rgb_tensor.unsqueeze(0),
                                                                   seg_tensor.squeeze(-1).unsqueeze(0),
                                                                   vinv_matrix.unsqueeze(0),
                                                                   proj_matrix.unsqueeze(0),
                                                                   camera_u2,
                                                                   camera_v2,
                                                                   width,
                                                                   height,
                                                                   1.2,
                                                                   device)
    with open('../cfg/example/grasp_fusion.yaml', 'r') as file:
        config = yaml.safe_load(file)
    # args = {
    #     'ckpt_path': '/home/island/Desktop/mobile_manipulation/IsaacGymEnvs/isaacgymenvs/ckpts/graspnet/checkpoint-rs.tar',
    #     'num_point': 20000,
    #     'num_view': 300,
    #     'collision_threshold': 0.01,
    #     'voxel_size': 0.01,
    # }
    # grasp_pos_detector = GraspPosDetector(args)
    # target_pcd = _get_target_pcd(frame_pcd.squeeze(0), frame_valid.squeeze(0), 160)
    # gg, cloud = grasp_pos_detector._get_grasp_prediction(target_pcd)

    grasp_config = {"interval_size": 0.15,
                    "max_octree_threshold": 0.015,
                    "min_octree_threshold": 0.01,
                    "min_ori_threshold": np.pi / 2}
    grasp_fusioner = GraspFusionVector(config['graspfusioner'])
    grasp_fusioner.add_new_frame(frame_pcd.squeeze(0), frame_valid.squeeze(0), 1, 160)
    print('graspFusion')




