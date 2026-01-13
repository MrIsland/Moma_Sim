"""
本文件中尝试将输入的grasp pos，对其周边voxel中的grasp pos进行fusion。
在每一次加入一个新的predict grasp pos的时候，会统计这个grasp pos所在的voxel附近的所有grasp pos，然后根据距离和位置做一个统一的fusion
"""

import os, sys
import pdb

import numpy as np
from queue import Queue
import time
from .interval_tree import RedBlackTree, Node, BLACK, RED, NIL
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
from common import rotation_matrix_to_axis_angle, slerp, average_quaternions, average_pos
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
        cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
        cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
        end_points = dict()
        cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cloud_sampled = cloud_sampled.to(device)
        end_points['point_clouds'] = cloud_sampled
        end_points['cloud_colors'] = color_sampled
        return end_points, cloud

    def vis_grasp_pos(self, gg, cloud):
        gg.nms()
        gg.sort_by_score()
        gg = gg[:64]
        grippers = gg.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud, *grippers])

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
        # self.vis_grasp_pos(gg, cloud)
        gg = gg.sort_by_score()[:100]
        return gg, cloud

class GraspFusion:
    def __init__(self, args):
        self.args = args
        self.x_rb_tree = RedBlackTree(self.args["interval_size"])
        self.y_rb_tree = RedBlackTree(self.args["interval_size"])
        self.z_rb_tree = RedBlackTree(self.args["interval_size"])

        self.scene_node = set()
        self.fusion_step = 0
        self.connected_num = 0
        self.score_total = 0
        self.fused_total = 0

        self.grasp_pos_detector = GraspPosDetector(self.args['graspnet'])
        self.pcd_list = []

    def reset_gltree(self):
        del self.x_rb_tree, self.y_rb_tree, self.z_rb_tree, self.scene_node

        self.x_rb_tree = RedBlackTree(self.args["interval_size"])
        self.y_rb_tree = RedBlackTree(self.args["interval_size"])
        self.z_rb_tree = RedBlackTree(self.args["interval_size"])
        self.scene_node = set()

        self.connected_num = 0
        self.score_total = 0
        self.fused_total = 0

    def init_grasp_node(self, grasp_coor):
        self.x_tree_node_list = []
        self.y_tree_node_list = []
        self.z_tree_node_list = []
        for p in range(grasp_coor.shape[0]):
            x, y, z = grasp_coor[p, 0], grasp_coor[p, 1], grasp_coor[p, 2]
            x_temp_node = self.x_rb_tree.add(x)
            y_temp_node = self.y_rb_tree.add(y)
            z_temp_node = self.z_rb_tree.add(z)

            self.x_tree_node_list.append(x_temp_node)
            self.y_tree_node_list.append(y_temp_node)
            self.z_tree_node_list.append(z_temp_node)

    def query_voxel_neighbors(self, pos):
        neighbor_grasps = set()
        for dx in range(-2, 3):
            x_node = self.x_rb_tree.find(pos[0] + dx)
            if x_node is None:
                continue
            for dy in range(-2, 3):
                y_node = self.y_rb_tree.find(pos[1] + dy)
                if y_node is None:
                    continue
                for dz in range(-2, 3):
                    z_node = self.z_rb_tree.find(pos[2] + dz)
                    if z_node is None:
                        continue

                    x_set = x_node.set_list
                    y_set = y_node.set_list
                    z_set = z_node.set_list
                    intersection = x_set & y_set & z_set
                    neighbor_grasps |= intersection
        return neighbor_grasps


    def _generate_o3d_pcd(self, target_pcd):
        # pdb.set_trace()
        pcd = target_pcd[:, :3]
        color = target_pcd[:, 3:6]
        final_pcd = o3d.geometry.PointCloud()
        final_pcd.points = o3d.utility.Vector3dVector(pcd.cpu().numpy())
        final_pcd.colors = o3d.utility.Vector3dVector(color.cpu().numpy())
        return final_pcd

    def add_new_frame(self, frame_pcd, frame_valid, target_label=2):
        # frame_pcd: tensor(num_pcd, 7)  [x, y, z, r, g, b, label]
        self.fusion_step += 1
        # self.vis_pcd(frame_pcd, frame_valid, 2)

        ### 1. 从pcd和mask中获得target_pcd
        target_pcd = self._get_target_pcd(frame_pcd, frame_valid, target_label)
        # o3d_pcd = self._generate_o3d_pcd(target_pcd)
        # self.pcd_list.append(o3d_pcd)
        if target_pcd is None:
            # print('Current frame do not have the target point cloud')
            return

        ### 2. 从target_pcd中预测grasp pose
        frame_grasp_group, frame_cloud = self.grasp_pos_detector._get_grasp_prediction(target_pcd)
        # self.grasp_pos_detector.vis_grasp_pos(frame_grasp_group, frame_cloud)

        ### 3. 将本帧预测到的grasp pos进行fusion
        # self.fusion(frame_grasp_group, frame_cloud, self.fusion_step)
        self.fusion_with_voxel_size(frame_grasp_group, frame_cloud, self.fusion_step)
        tmp_frame_grasp_group = self._get_gg_from_cur_scene_node()

        ### 4. 可视化
        print('self.fused_total:    {}; grasp pos number:   {}'.format(self.fused_total, len(tmp_frame_grasp_group)))
        self.grasp_pos_detector.vis_grasp_pos(tmp_frame_grasp_group, frame_cloud)

        return frame_cloud

    def add_new_grasps(self, frame_grasp_group, frame_index):
        per_image_node_set = set()
        grasp_group_array = frame_grasp_group.grasp_group_array
        ln = grasp_group_array.shape[0]
        for p in range(ln):
            x_set_union = self.x_tree_node_list[p].set_list
            y_set_union = self.y_tree_node_list[p].set_list
            z_set_union = self.z_tree_node_list[p].set_list
            set_intersection = x_set_union[0] & y_set_union[0] & z_set_union[0]

            tmp_branch = [None] * 8
            tmp_branch_distance = np.full((8), self.args['max_octree_threshold'])
            is_find_nearest = False
            branch_record = set()
            list_intersection = list(set_intersection)
            random.shuffle(list_intersection)

            pos = grasp_group_array[p, 13:16]
            ori = grasp_group_array[p, 4:4+9].reshape(3, 3)
            quat_ori = R.from_matrix(ori).as_quat()

            score = grasp_group_array[p, 0]
            for grasp_iter in list_intersection:
                if rotation_matrix_to_axis_angle(grasp_iter.grasp_ori @ ori.T) > self.args["min_ori_threshold"]:
                    continue

                distance = np.linalg.norm(grasp_iter.grasp_pos - pos)
                if distance < self.args['min_octree_threshold']:
                    is_find_nearest = True
                    if frame_index != grasp_iter.frame_id:
                        grasp_iter.frame_id = frame_index
                        grasp_iter.fused_count += 1

                        weight = score / (grasp_iter.score + score)
                        grasp_iter.grasp_pos = (grasp_iter.grasp_pos * (1 - weight) + pos * weight)
                        quat_ori = slerp(quat_ori, R.from_matrix(grasp_iter.grasp_ori).as_quat(), 1 - weight)
                        grasp_iter.grasp_ori = R.from_quat(quat_ori).as_matrix()
                        grasp_iter.score = grasp_iter.score + score
                        grasp_iter.update()

                        self.fused_total += 1

                    per_image_node_set.add(grasp_iter)
                    break

                x = int(grasp_iter.grasp_pos[0] >= pos[0])
                y = int(grasp_iter.grasp_pos[1] >= pos[1])
                z = int(grasp_iter.grasp_pos[2] >= pos[2])
                branch_num = x * 4 + y * 2 + z

                if distance < grasp_iter.branch_distance[7 - branch_num]:
                    branch_record.add((grasp_iter, 7 - branch_num, distance))

                    if distance < tmp_branch_distance[branch_num]:
                        tmp_branch[branch_num] = grasp_iter
                        tmp_branch_distance[branch_num] = distance

            if not is_find_nearest:
                new_grasp = Grasp6d(pos, ori, score, grasp_group_array[p])
                new_grasp.frame_id = frame_index

                for grasp_branch in branch_record:
                    grasp_branch[0].branch_array[grasp_branch[1]] = new_grasp
                    grasp_branch[0].branch_distance[grasp_branch[1]] = grasp_branch[2]
                    self.connected_num += 1
                    self.fused_total += 1

                new_grasp.branch_array = tmp_branch
                new_grasp.branch_distance = tmp_branch_distance
                per_image_node_set.add(new_grasp)
                self.score_total += new_grasp.score

                for x_set in x_set_union:
                    x_set.add(new_grasp)
                for y_set in y_set_union:
                    y_set.add(new_grasp)
                for z_set in z_set_union:
                    z_set.add(new_grasp)

        # self.observation_window = self.observation_window.union(per_image_node_set)
        self.scene_node = self.scene_node.union(per_image_node_set)
        # pdb.set_trace()
        return per_image_node_set

    def fusion(self, frame_grasp_group, frame_cloud, frame_idx):
        group_coor = copy.deepcopy(frame_grasp_group.translations)
        self.init_grasp_node(group_coor)
        _ = self.add_new_grasps(frame_grasp_group, frame_idx)

    def fusion_with_voxel_size(self, frame_grasp_group, frame_cloud, frame_idx, voxel_size=2):
        group_coor = copy.deepcopy(frame_grasp_group.translations)
        per_image_node_set = set()
        grasp_group_array = frame_grasp_group.grasp_group_array
        ln = grasp_group_array.shape[0]
        for p in range(ln):
            # x_tree_node_list, y_tree_node_list, z_tree_node_list = [], [], []
            coor_x, coor_y, coor_z = group_coor[p][0], group_coor[p][1], group_coor[p][2]
            set_intersection = set()
            for d in range(-voxel_size, voxel_size + 1):
                tmp_coor_x = coor_x + d * self.args['interval_size']
                tmp_coor_y = coor_y + d * self.args['interval_size']
                tmp_coor_z = coor_z + d * self.args['interval_size']
                tmp_x_node = self.x_rb_tree.find_node(tmp_coor_x)
                tmp_y_node = self.y_rb_tree.find_node(tmp_coor_y)
                tmp_z_node = self.z_rb_tree.find_node(tmp_coor_z)
                x_set_union = tmp_x_node.set_list[0] if tmp_x_node is not None else set()
                y_set_union = tmp_y_node.set_list[0] if tmp_y_node is not None else set()
                z_set_union = tmp_z_node.set_list[0] if tmp_z_node is not None else set()

                set_intersection = set_intersection | (x_set_union & y_set_union & z_set_union)
                # x_tree_node_list.append(tmp_x_node)
                # y_tree_node_list.append(tmp_y_node)
                # z_tree_node_list.append(tmp_z_node)
            list_intersection = list(set_intersection)
            random.shuffle(list_intersection)
            pos = grasp_group_array[p, 13:16]
            ori = grasp_group_array[p, 4:4+9].reshape(3, 3)
            quat_ori = R.from_matrix(ori).as_quat()
            score = grasp_group_array[p, 0]

            correct_grasp_pos_idg_list = []
            correct_grasp_pos_idg_weights = []
            correct_grasp_pos_idg_quat = []
            correct_grasp_pos_idg_pos = []
            tmp_score = 0
            for idg, grasp_iter in enumerate(list_intersection):
                if rotation_matrix_to_axis_angle(grasp_iter.grasp_ori @ ori.T) > self.args["min_ori_threshold"]:
                    continue
                distance = np.linalg.norm(grasp_iter.grasp_pos - pos)
                if distance < self.args['min_octree_threshold']:
                    correct_grasp_pos_idg_list.append(idg)
                    tmp_score += grasp_iter.score

                    correct_grasp_pos_idg_weights.append(grasp_iter.score)
                    correct_grasp_pos_idg_quat.append(grasp_iter.grasp_ori)
                    correct_grasp_pos_idg_pos.append(grasp_iter.grasp_pos)

                    self.fused_total += 1
                    tmp_x_node = self.x_rb_tree.find_node(grasp_iter.grasp_pos[0])
                    tmp_y_node = self.x_rb_tree.find_node(grasp_iter.grasp_pos[1])
                    tmp_z_node = self.x_rb_tree.find_node(grasp_iter.grasp_pos[2])
                    tmp_x_node.set_list[0].discard(grasp_iter)
                    tmp_y_node.set_list[0].discard(grasp_iter)
                    tmp_z_node.set_list[0].discard(grasp_iter)
                    self.scene_node.discard(grasp_iter)
                    self.fused_total += 1

            correct_grasp_pos_idg_weights.append(score)
            correct_grasp_pos_idg_quat.append(ori)
            correct_grasp_pos_idg_pos.append(pos)
            final_pos = average_pos(correct_grasp_pos_idg_pos, correct_grasp_pos_idg_weights)
            final_ori = average_quaternions(correct_grasp_pos_idg_quat, correct_grasp_pos_idg_weights)
            new_grasp = Grasp6d(final_pos, R.from_quat(final_ori).as_matrix(), np.sum(correct_grasp_pos_idg_weights), grasp_group_array[p])
            new_grasp.frame_id = frame_idx
            new_grasp.update()
            per_image_node_set.add(new_grasp)

        self.scene_node = self.scene_node.union(per_image_node_set)
        list_per_image_node = list(per_image_node_set)
        for grasp_iter in list_per_image_node:
            x_temp_node = self.x_rb_tree.add(grasp_iter.grasp_pos[0])
            y_temp_node = self.y_rb_tree.add(grasp_iter.grasp_pos[1])
            z_temp_node = self.z_rb_tree.add(grasp_iter.grasp_pos[2])
            x_set_union, y_set_union, z_set_union = x_temp_node.set_list, y_temp_node.set_list, z_temp_node.set_list
            for x_set in x_set_union:
                x_set.add(grasp_iter)
            for y_set in y_set_union:
                y_set.add(grasp_iter)
            for z_set in z_set_union:
                z_set.add(grasp_iter)

        return per_image_node_set

    def _get_gg_from_cur_scene_node(self):
        list_scene_node = list(self.scene_node)
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
                    "min_octree_threshold": 0.03,
                    "min_ori_threshold": np.pi / 2}
    grasp_fusioner = GraspFusion(config['graspfusioner'])
    for i in range(0, 10):
        grasp_fusioner.add_new_frame(frame_pcd.squeeze(0), frame_valid.squeeze(0), 160)
    print('graspFusion')




