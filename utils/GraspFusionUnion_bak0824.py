"""
这个文件中的grasp fusion是对于同一个voxel中的grasp pos进行fusion。
"""
import os, sys
import pdb

import numpy as np
from queue import Queue
import time
# from interval_tree import RedBlackTree, Node, BLACK, RED, NIL
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
from common import rotation_matrix_to_axis_angle, slerp, voxel_downsample, draw_camera
from scipy.spatial.transform import Rotation as R
from collections import defaultdict

class Grasp6d:
    def __init__(self, grasp_pos, grasp_ori, score, grasp_group_array, grasp_idx):
        self.grasp_pos = grasp_pos
        self.grasp_ori = grasp_ori
        self.grasp_idx = grasp_idx

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

    def vis_grasp_pos(self, gg, cloud, with_camera=False):
        if gg is None:
            return
        gg.nms()
        gg.sort_by_score()
        # gg = gg[:64]
        gg = gg
        grippers = gg.to_open3d_geometry_list()
        rotation_matrices = gg.rotation_matrices
        translations = gg.translations
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        axis_ori = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrices[0]
        transform[:3, 3] = translations[0]
        axis.transform(transform)

        if with_camera:
            o3d.visualization.draw_geometries([cloud, *grippers, axis])
        else:
            o3d.visualization.draw_geometries([cloud, *grippers, axis, axis_ori])

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
        # print('before collision_t')
        if self.collision_thresh > 0:
            gg = self.collision_detection(gg, np.array(cloud.points))
        gg = gg.sort_by_score()
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
        self.grasp_pos_id = 0

        self.grasp_pos_detector = GraspPosDetector(self.args['graspnet'])
        self.pcd_list = []

        self.parent = {}
        self.component_size = defaultdict(int)
        self.grasp_idx = 0

    def reset_gltree(self):
        del self.x_rb_tree, self.y_rb_tree, self.z_rb_tree, self.scene_node

        self.x_rb_tree = RedBlackTree(self.args["interval_size"])
        self.y_rb_tree = RedBlackTree(self.args["interval_size"])
        self.z_rb_tree = RedBlackTree(self.args["interval_size"])

        self.scene_node = set()
        self.fusion_step = 0
        self.connected_num = 0
        self.score_total = 0
        self.fused_total = 0

    def init_grasp_node(self, grasp_coor):
        self.x_tree_node_list = []
        self.y_tree_node_list = []
        self.z_tree_node_list = []

        x_tree_node_list = []
        y_tree_node_list = []
        z_tree_node_list = []

        for p in range(grasp_coor.shape[0]):
            x_temp_node = self.x_rb_tree.add(grasp_coor[p, 0])
            y_temp_node = self.y_rb_tree.add(grasp_coor[p, 1])
            z_temp_node = self.z_rb_tree.add(grasp_coor[p, 2])

            x_tree_node_list.append(x_temp_node)
            y_tree_node_list.append(y_temp_node)
            z_tree_node_list.append(z_temp_node)

        self.x_tree_node_list = copy.deepcopy(x_tree_node_list)
        self.y_tree_node_list = copy.deepcopy(y_tree_node_list)
        self.z_tree_node_list = copy.deepcopy(z_tree_node_list)

    def _generate_o3d_pcd(self, target_pcd):
        pcd = target_pcd[:, :3]
        color = target_pcd[:, 3:6]
        final_pcd = o3d.geometry.PointCloud()
        final_pcd.points = o3d.utility.Vector3dVector(pcd.cpu().numpy())
        final_pcd.colors = o3d.utility.Vector3dVector(color.cpu().numpy())
        return final_pcd

    def add_new_frame(self, frame_pcd, frame_valid, frame_idx, target_label=2):
        # frame_pcd: tensor(num_pcd, 7)  [x, y, z, r, g, b, label]
        # print('frame_idx:     {}'.format(frame_idx))
        self.fusion_step += 1
        # self.vis_pcd(frame_pcd, frame_valid, target_label)
        ### 1. 获取目标点云
        target_pcd = self._get_target_pcd(frame_pcd, frame_valid, target_label)
        if target_pcd is None:
            # print('Current frame do not have the target point cloud')
            return None

        ### 2. 从target_pcd中预测grasp pos
        frame_grasp_group, frame_cloud = self.grasp_pos_detector._get_grasp_prediction(target_pcd)
        ### 3. 将本帧预测到的grasp pos进行fusion，包括了对grasp pos的筛选
        self.grasp_pos_detector.vis_grasp_pos(frame_grasp_group, frame_cloud)
        # print('len(frame_grasp_group):   {}'.format(len(frame_grasp_group)))
        # pdb.set_trace()

        per_image_node_set = self.fusion(frame_grasp_group, frame_cloud, self.fusion_step)

        ### 4. 可视化
        tmp_frame_grasp_group = self._get_gg_from_cur_scene_node(self.scene_node)
        # if frame_idx == 0:
        #     print('self.fused_total:    {}; grasp pos number:   {}'.format(self.fused_total, len(tmp_frame_grasp_group)))
        self.grasp_pos_detector.vis_grasp_pos(tmp_frame_grasp_group, frame_cloud)
        return frame_cloud

    def add_new_grasps(self, frame_grasp_group, frame_index):
        per_image_node_set = set()
        grasp_group_array = frame_grasp_group.grasp_group_array
        ln = grasp_group_array.shape[0]
        for p in range(ln):
            self.grasp_idx += 1
            self.component_size[self.grasp_idx] = 1
            x_set_union = self.x_tree_node_list[p].set_list   # 当前这个节点x所在的区间，以往的结点
            y_set_union = self.y_tree_node_list[p].set_list
            z_set_union = self.z_tree_node_list[p].set_list
            set_intersection = x_set_union[0] & y_set_union[0] & z_set_union[0]

            tmp_branch = [None] * 8
            tmp_branch_distance = np.full((8), self.args['max_octree_threshold'])
            is_find_nearest = False
            branch_record = set()
            list_intersection = list(set_intersection)
            random.shuffle(list_intersection)
            # print('list_intersection length:    {}'.format(len(list_intersection)))
            pos = grasp_group_array[p, 13:16]
            ori = grasp_group_array[p, 4:4+9].reshape(3, 3)
            quat_ori = R.from_matrix(ori).as_quat()

            score = grasp_group_array[p, 0]
            for grasp_iter in list_intersection:
                # 如果旋转差很多，就丢掉
                if rotation_matrix_to_axis_angle(grasp_iter.grasp_ori @ ori.T) > self.args["min_ori_threshold"]:
                    continue

                distance = np.linalg.norm(grasp_iter.grasp_pos - pos)
                # 距离很小，直接做融合
                if distance < self.args['min_octree_threshold']:
                    is_find_nearest = True
                    # if frame_index != grasp_iter.frame_id:
                    grasp_iter.frame_id = frame_index
                    grasp_iter.fused_count += 1

                    weight = score / (grasp_iter.score + score)
                    grasp_iter.grasp_pos = (grasp_iter.grasp_pos * (1 - weight) + pos * weight)
                    quat_ori = slerp(quat_ori, R.from_matrix(grasp_iter.grasp_ori).as_quat(), 1 - weight)
                    grasp_iter.grasp_ori = R.from_quat(quat_ori).as_matrix()
                    grasp_iter.score = grasp_iter.score + score
                    grasp_iter.update()

                    self.fused_total += 1
                    self.union(grasp_iter.grasp_idx, self.grasp_idx)
                    # per_image_node_set.add(grasp_iter)
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
                new_grasp = Grasp6d(pos, ori, score, grasp_group_array[p], self.grasp_idx)
                new_grasp.frame_id = frame_index

                for grasp_branch in branch_record:
                    grasp_branch[0].branch_array[grasp_branch[1]] = new_grasp
                    grasp_branch[0].branch_distance[grasp_branch[1]] = grasp_branch[2]
                    self.connected_num += 1
                    self.fused_total += 1
                    self.union(grasp_branch[0].grasp_idx, self.grasp_idx)

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

        valid_per_image_scene_node = self.valid_grasping_identification(per_image_node_set)

        self.scene_node = self.scene_node.union(valid_per_image_scene_node)
        return valid_per_image_scene_node

    #@brief: 假设筛选机制很严格，每一帧留下来的grasp pos都是很优秀的grasp pos
    #        那么对于新加入的grasp pos而言，在最后做一次筛选，如果能和之前grasp pos连上关系的就ok
    #        如果没有连上关系，只要连接的grasp pos数目大于tau_count(5)？ 就保存下来，不然就直接删除
    #@param per_image_node_set: 表示的是当前这个时刻新加的一些grasp pos，不包含那些加入融合的grasp pos
    def valid_grasping_identification(self, per_image_node_set):
        filter_set = set()
        for g in per_image_node_set:
            g_grasp_idx = g.grasp_idx
            root = self.find(g_grasp_idx)
            # print('g.score:   {}:     g.grasp_idx: {}'.format(g.score, g.grasp_idx))
            # print('root:   {};  self.component_size:   {}'.format(root, self.component_size[root]))
            if self.component_size[root] >= self.args.get('min_component_size', 3) and g.score > 1.0:
                if self.is_z_aligned(g.grasp_ori):
                    filter_set.add(g)

        for g in filter_set:
            grasp_pos = g.grasp_pos
            x_temp_node = self.x_rb_tree.add(grasp_pos[0])
            y_temp_node = self.y_rb_tree.add(grasp_pos[1])
            z_temp_node = self.z_rb_tree.add(grasp_pos[2])

            x_set_union = x_temp_node.set_list
            y_set_union = y_temp_node.set_list
            z_set_union = z_temp_node.set_list

            for x_set in x_set_union:
                x_set.add(g)
            for y_set in y_set_union:
                y_set.add(g)
            for z_set in z_set_union:
                z_set.add(g)

        return filter_set

    def is_z_aligned(self, grasp_ori, angle_threshold_deg=30):
        world_z = np.array([0, 0, 1])
        grasp_z = grasp_ori[:, 2]
        grasp_z = grasp_z / np.linalg.norm(grasp_z)  # 单位化

        cos_theta = np.dot(grasp_z, world_z)
        return cos_theta >= np.cos(np.radians(angle_threshold_deg))

    def fusion(self, frame_grasp_group, frame_cloud, frame_idx):
        group_coor = copy.deepcopy(frame_grasp_group.translations)
        self.init_grasp_node(group_coor)
        per_image_node_set = self.add_new_grasps(frame_grasp_group, frame_idx)
        self.scene_node = set(sorted(self.scene_node, key=lambda g: g.score, reverse=True)[:self.args['max_grasp_pos']])
        # print('len(self.scene_node):    {}'.format(len(self.scene_node)))
        return per_image_node_set

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
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)

        # Step 5: 可视化
        window_title = f"Point Cloud - Label {target_label}" if target_label is not None else "Point Cloud"
        o3d.visualization.draw_geometries([pcd, axis], window_name=window_title, width=800, height=600)

    def find(self, grasp):
        # 路径压缩
        if self.parent.get(grasp, grasp) != grasp:
            self.parent[grasp] = self.find(self.parent[grasp])
        return self.parent.get(grasp, grasp)

    def union(self, grasp_a, grasp_b):
        root_a = self.find(grasp_a)
        root_b = self.find(grasp_b)
        if root_a == root_b:
            return
        # 小集合合并到大集合
        if self.component_size[root_a] < self.component_size[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        self.component_size[root_a] += self.component_size[root_b]
        self.component_size[root_b] = 0

##############################################################################
#### test code ####

from common import to_torch, mov

def mov(tensor, device):
    return torch.from_numpy(tensor.cpu().numpy()).to(device)

#@Notice 生成的点云是在世界坐标系下的
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
    rgb_tensor = torch.ones_like(rgb_tensor) * 155
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
                    "min_ori_threshold": np.pi / 2,
                    'min_component_size': 10,
                    'max_grasp_pos': 200}
    grasp_fusioner = GraspFusion(config['graspfusioner'])
    grasp_fusioner.add_new_frame(frame_pcd.squeeze(0), frame_valid.squeeze(0), 1, 160)
    print('graspFusion')




