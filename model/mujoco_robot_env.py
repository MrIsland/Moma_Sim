import os, sys
import time

import torch
import numpy as np
import mujoco
from mujoco import viewer
import copy
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from utils.GraspFusionUnion import GraspFusion

from termcolor import cprint
from .ik_robot_controller import IKARMController
from .control_policy import ControlPolicy
from utils.common import *
from .pid import PID
import open3d as o3d
import torch.nn.functional as F
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_rotation_6d


from gymnasium.envs.mujoco.mujoco_rendering import OffScreenViewer, MujocoRenderer

class MujocoRobotEnv:
    def __init__(self, cfg):
        self.cfg = cfg
        ### 1. 模型urdf导入仿真
        self.asset_root = self.cfg['asset']['assetRoot']
        self.asset_moma = self.cfg['asset']['assetFileNameMoma']
        self.asset_arm = self.cfg['asset']['assetFileNameArm']
        moma_xml = os.path.join(self.asset_root, self.asset_moma)
        arm_xml = os.path.join(self.asset_root, self.asset_arm)
        self.robot_model = mujoco.MjModel.from_xml_path(moma_xml)
        self.robot_data = mujoco.MjData(self.robot_model)

        # self.arm_model = mujoco.MjModel.from_xml_path(arm_xml)
        # self.arm_data = mujoco.MjData(self.arm_model)

        ### 2. Algorithm创建
        self.control_policy = ControlPolicy(cfg['params'])
        self.control_policy.restore()
        # self.net = self.control_policy.model

        ### 3. 仿真交互
        self.viewer = viewer.launch_passive(self.robot_model, self.robot_data)
        self.cam_viewer = OffScreenViewer(self.robot_model, self.robot_data)

        ### Some Setting
        self.momaDefaultDofPos = np.array(self.cfg['asset']['momaDefaultDofPos'])
        self.momaDefaultStatePos = np.array(self.cfg['asset']['momaDefaultStatePos'])

        joint_name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        self.arm_joint_ids = [self.robot_model.joint(name).id for name in joint_name]
        self.arm_joint_ids = self.robot_model.jnt_qposadr[self.arm_joint_ids]
        self.ee_body_id = self.robot_model.body('arm_link6').id
        # arm_link1_id = self._get_part_id('joint1', 'joint')
        # arm_link6_id = self._get_part_id('joint6', 'joint')
        # self.arm_joints_ids = [np.arange(arm_link1_id, arm_link6_id + 1)]

        ### IKARMController
        self.asset_arm_urdf = self.cfg['asset']['assetFileNameArmURDF']
        arm_urdf = os.path.join(self.asset_root, self.asset_arm_urdf)
        self.ik_arm_controller = IKARMController(os.path.join(arm_urdf), ee_link_index=5, use_gui=False)
        self.arm_ee_pos = None

        Kp_arm, Ki_arm, Kd_arm = np.array(self.cfg['PID']['kp_arm']), np.array(self.cfg['PID']['ki_arm']), np.array(self.cfg['PID']['kd_arm'])
        llim_arm, ulim_arm = np.array(self.cfg['PID']['llim_arm']), np.array(self.cfg['PID']['ulim_arm'])
        self.arm_pid = PID("arm", Kp_arm, Ki_arm, Kd_arm, dim=6, llim=llim_arm, ulim=ulim_arm, debug=False)
        self.cnt = 0
        self.states = {}
        self.reset()

        self.last_ori_grasp_pos_tensor = None
        self.grasp_pose_factory = GraspFusion(self.cfg['graspfusioner'])
        # for i in range(self.robot_model.njnt):
        #     joint_name = self.robot_model.joint(i).name
        #     qpos_addr = self.robot_model.jnt_qposadr[i]
        #     print(f"{joint_name}: qpos[{qpos_addr}:{qpos_addr}]")
        # self._get_image()

        #### visual encoder
        self.img_h, self.img_w = self.cfg['params']['network']['visual_encoder']['img_h'], self.cfg['params']['network']['visual_encoder']['img_w']
        self.in_channels = self.cfg['params']['network']['visual_encoder']['in_channels']

        self.robot_state_list, self.robot_qvel_list = [], []
        print('MujocoRobotEnv')

    def render(self):
        self.viewer.sync()

    def _close_viewer(self):
        self.viewer.close()

    def _set_gripper_state(self, gripper_state):
        finger_joint_id = self._get_part_id('finger_joint', 'joint')
        right_kckle_joint_id = self._get_part_id('right_kckle_joint', 'joint')
        finger_joint_id = self.robot_model.jnt_qposadr[finger_joint_id]
        right_kckle_joint_id = self.robot_model.jnt_qposadr[right_kckle_joint_id]
        self.robot_data.qpos[finger_joint_id:right_kckle_joint_id+1] = np.array([gripper_state,
                                                                                 -1 * gripper_state,
                                                                                 gripper_state,
                                                                                 gripper_state,
                                                                                 -1 * gripper_state,
                                                                                 gripper_state])

    def _set_arm_joint_pos(self, joint_state):
        arm_link1_id = self._get_part_id('joint1', 'joint')
        arm_link6_id = self._get_part_id('joint6', 'joint')
        arm_link1_id = self.robot_model.jnt_qposadr[arm_link1_id]
        arm_link6_id = self.robot_model.jnt_qposadr[arm_link6_id]
        self.robot_data.qpos[arm_link1_id:arm_link6_id+1] = joint_state
        mujoco.mj_fwdPosition(self.robot_model, self.robot_data)
        # self.data_arm.qpos[0:6] = joint_state

    def _set_base_pos(self, base_pos):
        self.robot_data.qpos[:7] = base_pos

    def _set_base_qvel(self, base_qvel):
        self.robot_data.qvel[:] = 0
        self.robot_data.qacc[:] = 0
        quat = self.robot_data.qpos[3:7]
        rot_mat = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        v_robot = np.array([base_qvel[0], 0, 0])
        v_world = rot_mat @ v_robot
        vx_world, vy_world = v_world[0], v_world[1]
        # print('{} {} {}'.format(vx_world, vy_world, base_qvel[1]))
        self.robot_data.qvel[0] = vx_world
        self.robot_data.qvel[1] = vy_world
        self.robot_data.qvel[5] = base_qvel[1]

    #@brief: 获取某些部分的id
    def _get_part_id(self, part_name, type='GEOM'):
        part_id = None
        if type == 'GEOM' or type == 'geom':
            part_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_GEOM, part_name)
        elif type == 'BODY' or type == 'body':
            part_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, part_name)
        elif type == "JOINT" or type == 'joint':
            part_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_JOINT, part_name)
        elif type == 'CAM' or type == 'cam':
            part_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_CAMERA, part_name)
        elif type == 'SITE' or type == 'site':
            part_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_SITE, part_name)

        return part_id

    def _get_observation(self):
        # obs = ["eef_pos", "eef_quat", "robot_vel"]   3 + 4 + 6 + 6
        # obs += ["q_gripper"]
        # obs += ['grasp_pos_vector']
        # obs += ['curr_depth_img_tensor_batch']

        # obs = ["eef_pos", "eef_quat", "robot_vel", "q_gripper", "grasp_pos_vector", "curr_depth_img_tensor_batch"]
        # obs = ["eef_pos", "eef_quat", "robot_vel", "q_gripper", "grasp_pos_vector"]
        # obs = ["eef_pos", "eef_6d", "robot_vel", "q_gripper_arm", "q_gripper", "grasp_pos_vector"]
        # obs = ["eef_pos", "eef_6d", "robot_vel", "q_gripper", "grasp_pos_vector"]
        obs = ["eef_pos", "eef_6d", "robot_vel", "grasp_pos_vector"]
        self._refresh()
        obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)

        # if self.cnt % 20 == 0:
        #     o3d.visualization.draw_geometries([pcd])
        # print('_get_observation')
        print('loaded_obs:     {}'.format(obs_buf[:, :9]))
        return obs_buf

    def cam2world(self, pcd):
        cam_pos = self.robot_data.cam_xpos[0]
        cam_mat = self.robot_data.cam_xmat[0]
        T_world_cam = np.eye(4)
        T_world_cam[:3, :3] = cam_mat.reshape(3, 3)
        T_world_cam[:3, 3] = cam_pos
        correct_matrix = np.diag([1, -1, -1, 1])
        pcd.transform(T_world_cam @ correct_matrix)
        return pcd

    def generate_grasp_pos_batch(self, pos, x_axis, y_axis, z_axis, radius, num_grasps):
        """
        pos: (N, 3)
        x/y/z_axis: (N, 3)
        Returns: list of length N, each element is (num_grasps, 7)
        """
        N = pos.shape[0]
        thetas = np.linspace(0, 2 * np.pi, num_grasps, endpoint=False)  # (num_grasps,)
        cos_theta = np.cos(thetas)  # (num_grasps,)
        sin_theta = np.sin(thetas)  # (num_grasps,)

        # Compute offsets: (N, num_grasps, 3)
        x_component = cos_theta[None, :, None] * x_axis[:, None, :]
        y_component = sin_theta[None, :, None] * y_axis[:, None, :]
        offsets = radius * (x_component + y_component)  # (N, num_grasps, 3)

        # Grasp positions: (N, num_grasps, 3)
        grasp_positions = pos[:, None, :] + offsets

        # z_g: pointing to center: (N, num_grasps, 3)
        x_g = -offsets / np.linalg.norm(offsets, axis=2, keepdims=True)
        z_g = np.repeat(z_axis[:, None, :], num_grasps, axis=1)  # (N, num_grasps, 3)
        y_g = np.cross(z_g, x_g)

        # Re-orthonormalize
        y_g /= np.linalg.norm(x_g, axis=2, keepdims=True)
        z_g = np.cross(x_g, y_g)

        # Stack into rotation matrices: (N, num_grasps, 3, 3)
        R_grasp = np.stack([x_g, y_g, z_g], axis=-1)

        # Flatten to (N * num_grasps, 3, 3) for scipy Rotation
        R_grasp_flat = R_grasp.reshape(-1, 3, 3)
        grasp_quats_flat = R.from_matrix(R_grasp_flat).as_quat()  # (N * num_grasps, 4)

        # Combine positions and quaternions
        grasp_positions_flat = grasp_positions.reshape(-1, 3)  # (N * num_grasps, 3)
        grasp_poses_flat = np.concatenate([grasp_positions_flat, grasp_quats_flat], axis=1)

        # Reshape back to (N, num_grasps, 7)
        grasp_poses_batch = grasp_poses_flat.reshape(N, num_grasps, 7)
        return grasp_poses_batch

    def compute_good_grasp_pos(self, bottle_pos, bottle_mat, radius=0.1, num_grasps=20):
        # bottle_pos = self._bottleA_state[:, :3]  # (N, 3)
        # bottle_quat = self._bottleA_state[:, 3:7]  # (N, 4)

        z_axis = R.from_matrix(bottle_mat.reshape(1, 3, 3)).apply(np.array([0, 0, 1]))

        # Choose orthogonal vector to z_axis for cross product
        dot_z = np.abs(np.sum(z_axis * np.array([0, 0, 1]), axis=1))  # (N,)
        ortho_vec = np.where(dot_z[:, None] < 0.99,
                             np.array([0, 0, 1])[None, :],
                             np.array([0, 1, 0])[None, :])  # (N, 3)

        x_axis = np.cross(z_axis, ortho_vec)
        x_axis /= np.linalg.norm(x_axis, axis=1, keepdims=True)
        y_axis = np.cross(z_axis, x_axis)

        # Generate grasp poses for all objects (N * num_grasps, 7)
        grasp_poses_batch = self.generate_grasp_pos_batch(
            bottle_pos, x_axis, y_axis, z_axis, radius=radius, num_grasps=num_grasps
        )

        return grasp_poses_batch  # list of N elements, each is (num_grasps, 7)


    def convert_grasp_pose_to_features(self, grasp_poses: torch.Tensor, num_samples: int = 64) -> torch.Tensor:
        """
        Convert grasp poses (x, y, z, qx, qy, qz, qw) to (x, y, z, rot[:, :2], score),
        and sample with replacement to get fixed number of grasps per env.

        Args:
            grasp_poses: (num_envs, num_grasps, 7)
            num_samples: int, number of grasps to sample per env (default: 64)

        Returns:
            Tensor of shape (num_envs, num_samples, 10)
        """
        num_envs, num_grasps, _ = grasp_poses.shape
        device = grasp_poses.device

        # 1. Random sampling indices with replacement
        sample_indices = torch.randint(0, num_grasps, (num_envs, num_samples), device=device)  # (num_envs, 64)

        # 2. Gather the sampled grasp poses
        sampled_grasps = torch.gather(
            grasp_poses, dim=1,
            index=sample_indices.unsqueeze(-1).expand(-1, -1, 7)
        )  # (num_envs, 64, 7)

        # 3. Split position and quaternion
        pos = sampled_grasps[..., :3]  # (num_envs, 64, 3)
        quat = sampled_grasps[..., 3:]  # (num_envs, 64, 4)

        # 4. Convert quaternion to rotation matrix
        # Result: (num_envs, 64, 3, 3)
        def quat_to_rotmat(q):
            """Convert normalized quaternion to rotation matrix."""
            q = F.normalize(q, dim=-1)
            qw, qx, qy, qz = q[..., 3], q[..., 0], q[..., 1], q[..., 2]

            # Rotation matrix components
            R = torch.zeros(q.shape[:-1] + (3, 3), device=q.device)
            R[..., 0, 0] = 1 - 2 * (qy ** 2 + qz ** 2)
            R[..., 0, 1] = 2 * (qx * qy - qz * qw)
            R[..., 0, 2] = 2 * (qx * qz + qy * qw)
            R[..., 1, 0] = 2 * (qx * qy + qz * qw)
            R[..., 1, 1] = 1 - 2 * (qx ** 2 + qz ** 2)
            R[..., 1, 2] = 2 * (qy * qz - qx * qw)
            R[..., 2, 0] = 2 * (qx * qz - qy * qw)
            R[..., 2, 1] = 2 * (qy * qz + qx * qw)
            R[..., 2, 2] = 1 - 2 * (qx ** 2 + qy ** 2)
            return R

        rot = quat_to_rotmat(quat)  # (num_envs, 64, 3, 3)

        # 5. Extract first two columns (each is 3D vector)
        rot_cols = rot[..., :, :2]  # (num_envs, 64, 3, 2)
        rot_cols = rot_cols.reshape(num_envs, num_samples, 6)  # (num_envs, 64, 6)

        # 6. Append score
        score = torch.ones((num_envs, num_samples, 1), device=device)  # (num_envs, 64, 1)

        # 7. Concatenate all
        features = torch.cat([pos, rot_cols, score], dim=-1)  # (num_envs, 64, 10)

        return features

    def _refresh(self):
        site_id = self._get_part_id('gripper_site', 'SITE')

        ##### state info:
        # eef_pos
        rgb, depth, seg = self._get_image()
        site_xmat = self.robot_data.site_xmat[site_id]
        site_quat = R.from_matrix(site_xmat.reshape(3, 3)).as_quat()
        site_quat = np.roll(site_quat, 1)
        pcd = self.depth_to_pointcloud(rgb, depth, 'twist')

        pcd = self.cam2world(pcd)
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        # o3d.visualization.draw_geometries([pcd, axis])
        # bottle_pcd = self.seg_to_pointcloud(rgb, depth, seg, 'bottle', 'twist')

        curr_depth_tensor = to_torch(depth).unsqueeze(0)
        curr_depth_tensor = torch.nan_to_num(curr_depth_tensor, posinf=1e10, neginf=0)
        curr_depth_tensor = curr_depth_tensor.unsqueeze(-1).permute(0, 3, 1, 2)
        curr_depth_tensor = F.interpolate(curr_depth_tensor,
                                        size=(self.img_h, self.img_w),
                                        mode='bilinear',
                                        align_corners=False)

        bottle_id = self._get_part_id('bottle', 'geom')
        bottle_pos = self.robot_data.geom_xpos[bottle_id]
        bottle_mat = self.robot_data.geom_xmat[bottle_id]
        # print('bottle_pos:       {}'.format(bottle_pos))
        # self.grasp_pos_tensor, _ = self.generate_random_transform_batch(bottle_pos)
        self.grasp_poses_factory = self.compute_good_grasp_pos(bottle_pos[None, :], bottle_mat)
        self.grasp_pos_tensor = self.convert_grasp_pose_to_features(to_torch(self.grasp_poses_factory))
        # if self.cnt % 20 == 0:
        #     self.grasp_pos_tensor, self.ori_grasp_pos_tensor = self.grasp_pos_extraction(pcd, seg, obj_name='bottle', camera_name='twist')

        self.states.update({
            'eef_pos': to_torch(self.robot_data.site_xpos[site_id]).unsqueeze(0),
            'eef_quat': to_torch(site_quat).unsqueeze(0),
            'eef_6d': matrix_to_rotation_6d(quaternion_to_matrix(to_torch(site_quat).unsqueeze(0))),
            'q_gripper_arm': to_torch(self.robot_data.qpos[self.arm_joint_ids]).unsqueeze(0),
            # 'eef_6d': matrix_to_rotation_6d(to_torch(site_xmat).reshape(3, 3).unsqueeze(0)),
            'robot_vel': to_torch(self.robot_data.qvel[:6]).unsqueeze(0),
            'q_gripper': to_torch(self.robot_data.qpos[-6:]).unsqueeze(0),
            'curr_depth_img_tensor_batch': curr_depth_tensor.reshape(1, -1),
            'grasp_pos_vector': self.grasp_pos_tensor.reshape(1, -1)
        })
        # print('eef_pos:     {}'.format(self.robot_data.xpos[17]))
        # print('gripper_site pos:     {}'.format(self.states['eef_pos']))
        # print('gripper_site quat:     {}'.format(self.states['eef_quat']))
        pos, orn = self.ik_arm_controller.get_ee_pose()
        arm_ee_pos = np.concatenate([pos, orn])
        # print('arm_ee_pos:    {}'.format(arm_ee_pos))
        # print('--------------------------------------------------------------')
        # print('eef_pos:      {};   eef_quat:        {}'.format(self.states['eef_pos'], self.states['eef_quat']))
        # print('------------------------------------------------')
        # print('self.robot_data.qpos:     {}'.format(self.robot_data.qpos[:7]))
        # print('self.robot_data.vel:      {}'.format(self.robot_data.qvel[:6]))
        # self.robot_state_list.append(copy.copy(self.robot_data.qpos[:7]))
        # self.robot_qvel_list.append(copy.copy(self.robot_data.qvel[:6]))
        # print('------------------------------------------------')
        # print('self.robot_data.q_gripper:      {}'.format(self.robot_data.qpos[-6:]))
        # print('self.gripper_site_ee_pos:     {}'.format(np.concatenate([self.robot_data.site_xpos[site_id], -site_quat])))
        # print('self.gripper_site_ee_pos:     {}'.format(np.concatenate([self.robot_data.site_xpos[site_id], R.from_matrix(site_xmat.reshape(3, 3)).as_euler('xyz')])))


    def generate_random_transform_batch(self, pos, num_samples=64):
        rot_matrices = R.random(num_samples).as_matrix()
        rot_matrices = torch.from_numpy(rot_matrices).float()

        rot_cols = rot_matrices[:, :, :2].reshape(num_samples, 6).to(pos)
        pos_tensor = pos.repeat(num_samples, 1)
        transforms = torch.cat([pos_tensor, rot_cols, torch.ones(num_samples, 1).to(pos)], dim=1)
        quats = R.from_matrix(rot_matrices.cpu().numpy()).as_quat()
        ori_transforms = torch.cat([pos_tensor, torch.from_numpy(quats).to(rot_cols.device), torch.ones(num_samples, 1).to(pos)], dim=1)
        # ori_transforms = torch.cat([pos_tensor, ])
        return transforms, ori_transforms

    #@brief: 根据点云提取grasp pos，并根据物体是否可视对grasp pos进行调整
    def grasp_pos_extraction(self, pcd, seg, obj_name='bottle', camera_name='twist', valid_grasp_pos_num=64, device='cuda:0'):
        bottle_id = self._get_part_id('bottle', 'geom')
        bottle_pos = to_torch(self.robot_data.geom_xpos[bottle_id])

        grasp_pos_list = []  # tensor(pos, F(quat), score) -> 10
        ori_grasp_pos_list = [] # tensor(pos, quat, score) -> 8
        obj_id = self._get_part_id(obj_name, 'GEOM')
        frame_cloud = self.grasp_pose_factory.add_new_frame(pcd, seg, self.cnt, target_label=obj_id)
        if frame_cloud is None:
            grasp_pos_vector, ori_grasp_pos_vector = self.generate_random_transform_batch(bottle_pos)
        else:
            all_grasp_group_array = []
            list_scene_node = list(self.grasp_pose_factory.scene_node)
            if len(list_scene_node) == 0:
                grasp_pos_vector, ori_grasp_pos_vector = self.generate_random_transform_batch(bottle_pos)
            else:
                for node in list_scene_node:
                    tmp_grasp_group_array = node.grasp_group_array
                    if node.score > 0.3:
                        all_grasp_group_array.append(tmp_grasp_group_array)
                if len(all_grasp_group_array) == 0:
                    grasp_pos_vector, ori_grasp_pos_vector = self.generate_random_transform_batch(bottle_pos)
                    grasp_pos_list.append(grasp_pos_vector)
                    ori_grasp_pos_list.append(ori_grasp_pos_vector)

                    grasp_pos_tensor = torch.stack(grasp_pos_list)
                    ori_grasp_pos_tensor = torch.stack(ori_grasp_pos_list)
                    return grasp_pos_tensor, ori_grasp_pos_tensor

                indices = torch.randint(0, len(all_grasp_group_array), (valid_grasp_pos_num, ))
                sampled = torch.tensor(all_grasp_group_array)[indices]
                grasp_pos_vector = torch.cat([sampled[:, 13:16], sampled[:, 4:10], sampled[:, 0].unsqueeze(1)], dim=1).to(device)
                rot_matrices = sampled[:, 4:4+9].reshape(-1, 3, 3)
                quats = R.from_matrix(rot_matrices).as_quat()
                ori_grasp_pos_vector = torch.cat([sampled[:, 13:16], torch.from_numpy(quats).to(sampled.device), sampled[:, 0].unsqueeze(1)], dim=1).to(device)
        grasp_pos_list.append(grasp_pos_vector)
        ori_grasp_pos_list.append(ori_grasp_pos_vector)
        grasp_pos_tensor = torch.stack(grasp_pos_list)
        ori_grasp_pos_tensor = torch.stack(ori_grasp_pos_list)
        return grasp_pos_tensor, ori_grasp_pos_tensor

    def depth_to_pointcloud(self, rgb, depth, camera_name):
        cam_id = self._get_part_id(camera_name, 'cam')
        fovy = self.robot_model.cam_fovy[cam_id]
        vfov = fovy / 360. * 2. * np.pi
        H, W = depth.shape
        fy = H / (2 * np.tan(vfov / 2))
        hfov = 2. * np.arctan(np.tan(vfov / 2) * W / H)
        fx = W / (2. * np.tan(hfov / 2.))
        cx, cy = W / 2.0, H / 2.0

        xs, ys = np.meshgrid(np.arange(W), np.arange(H))
        X = (xs - cx) * depth / fx
        Y = (ys - cy) * depth / fy
        Z = depth

        points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        colors = rgb.reshape(-1, 3) / 255.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    def seg_to_pointcloud(self, rgb, depth, seg, obj_name, camera_name):
        cam_id = self._get_part_id(camera_name, 'cam')
        obj_id = self._get_part_id(obj_name, 'GEOM')
        fovy = self.robot_model.cam_fovy[cam_id]
        H, W = depth.shape
        fy = H / (2.0 * np.tan(np.deg2rad(fovy) / 2.0))
        fx = fy * (W / H)
        cx, cy = W / 2.0, H / 2.0

        mask = (seg[..., 1] == obj_id)
        ys, xs = np.where(mask)
        depth_vals = depth[ys, xs]

        X = (xs - cx) * depth_vals / fx
        Y = (ys - cy) * depth_vals / fy
        Z = depth_vals
        points = np.stack([X, Y, Z], axis=-1)

        colors = rgb[ys, xs] / 255.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    # 从相机中读取图像
    def _get_image(self):
        # 确保 MuJoCo 状态更新

        self.cam_viewer.make_context_current()
        rgb = self.cam_viewer.render(
            render_mode='rgbd_tuple', camera_id=0, segmentation=False
        )
        depth = self.cam_viewer.render(
            render_mode='depth_array', camera_id=0,
        )
        seg = self.cam_viewer.render(
            render_mode='rgbd_tuple', camera_id=0, segmentation=True
        )

        depth = self.depth_2_meters(depth)
        # depth = (2.0 * znear * zfar) / (zfar + znear - (2.0 * depth - 1.0) * (zfar - znear))
        mask = (depth <= 5.0)
        depth = np.where(mask, depth, 0)

        # path = './output'
        # import cv2
        # cv2.imwrite(os.path.join(path, 'rgb.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        # cv2.imwrite(os.path.join(path, 'depth.png'), (depth * 255).astype(np.uint8))
        # cv2.imwrite(os.path.join(path, 'seg.png'), seg[:, :, 1].astype(np.uint8) * 80)
        return rgb, depth, seg

    def depth_2_meters(self, depth):
        extend = self.robot_model.stat.extent
        near = self.robot_model.vis.map.znear * extend
        far = self.robot_model.vis.map.zfar * extend

        return near / (1 - depth * (1 - near / far))

    def motor_pd_control(self, target_qpos, kp=1000, kd=40):
        qpos = self.robot_data.qpos[self.arm_joint_ids]
        qvel = self.robot_data.qvel[self.arm_joint_ids]

        # 重力补偿项
        # mujoco.mj_inverse(self.robot_model, self.robot_data)
        # gravity_torque = self.robot_data.qfrc_bias[self.arm_joint_ids]

        # PD 控制 + 重力补偿
        torque = kp * (target_qpos - qpos) - kd * qvel
        print('target_qpos:     {};   qpos:     {}'.format(target_qpos, qpos))
        print('target_qpos - qpos:    {};    qvel:     {}'.format(target_qpos - qpos, qvel))
        # 写入控制信号
        self.robot_data.ctrl[:] = torque

    def transform_delta_ee_pos(self, delta_ee_pos):
        # 输入: delta_ee_pos = [dx, dy, dz, droll, dpitch, dyaw]
        delta_ee_pos = np.array(delta_ee_pos)
        pos = delta_ee_pos[:3]
        rpy = delta_ee_pos[3:]

        # 创建旋转器：绕 Z 轴旋转 180°
        R_z_180 = R.from_euler('z', np.pi)  # 绕Z轴180度

        # 旋转平移部分
        pos_rotated = R_z_180.apply(pos)

        # 将rpy转换为旋转矩阵，执行变换，再转回rpy
        ori = R.from_euler('xyz', rpy)
        ori_rotated = R_z_180 * ori  # 注意旋转组合的顺序
        rpy_rotated = ori_rotated.as_euler('xyz')

        return np.concatenate([pos_rotated, rpy_rotated])

    def control_ik_mujoco(self, dpose, damping, eef_body_name):
        """
        在 MuJoCo 中实现阻尼最小二乘 IK
        参数:
            model: mujoco.MjModel
            data: mujoco.MjData
            dpose: np.ndarray, shape (num_envs, 6)，期望末端位姿增量 (Δx)
            damping: float，阻尼系数
            eef_body_name: str，末端执行器 body 名称
        返回:
            u: np.ndarray, shape (num_envs, 6)，关节速度/增量
        """
        eef_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, eef_body_name)

        J_pos = np.zeros((3, self.robot_model.nv))
        J_rot = np.zeros((3, self.robot_model.nv))
        mujoco.mj_jacBody(self.robot_model, self.robot_data, J_pos, J_rot, eef_id)
        J_eef = np.vstack([J_pos, J_rot])
        J_eef = J_eef[:, self.arm_joint_ids]
        JT = J_eef.T
        lambda_I = np.eye(6) * (damping ** 2)
        dq = JT @ np.linalg.inv(J_eef @ JT + lambda_I) @ dpose

        return dq

    def _move_arm(self, delta_ee_pos):

        pos, orn = self.ik_arm_controller.get_ee_pose()
        self.arm_ee_pos = np.concatenate([pos, orn])
        target_joint_state = self.ik_arm_controller.compute_ik(delta_ee_pos)
        self.ik_arm_controller.set_joint_state(target_joint_state)
        # self.robot_data.qpos[self.arm_joint_ids] = target_joint_state

        # delta_js = self.control_ik_mujoco(delta_ee_pos, damping=0.05, eef_body_name='arm_link6')
        # print('-------------------------------------------')
        # print('delta_ee_pos:  {}'.format(delta_ee_pos))
        # print('delta_js:      {}'.format(delta_js))
        # print('-------------------------------------------')
        # target_joint_state = self.robot_data.qpos[self.arm_joint_ids] + delta_js
        # self.ik_arm_controller.set_joint_state(target_joint_state)

        #### Difference Direct Setting Position
        self.robot_data.qpos[self.arm_joint_ids] = target_joint_state

        #### Try to use torque control
        # print('################################################')
        # print('target_joint_state:       {}'.format(target_joint_state))
        # print('self.robot_data.qpos[self.arm_joint_ids]:       {}'.format(self.robot_data.qpos[self.arm_joint_ids]))
        # print('################################################')
        # mv_arm = self.arm_pid.update(target_joint_state, self.robot_data.qpos[self.arm_joint_ids], self.robot_data.time)
        # self.robot_data.ctrl = mv_arm

    def compute_torque_from_delta_eepos(self, delta_ee_pos, kp=50.0, kd=5.0):
        #1. 获取末端位姿雅可比矩阵
        J_pos = np.zeros((3, self.robot_model.nv))
        J_rot = np.zeros((3, self.robot_model.nv))
        mujoco.mj_jacSite(self.robot_model, self.robot_data, J_pos, J_rot, self.ee_body_id)

        J_full = np.vstack([J_pos, J_rot])  # 6 x nv

        #2. 关节当前状态
        qpos = self.robot_data.qpos[self.arm_joint_ids]
        qvel = self.robot_data.qvel[self.arm_joint_ids]

        #3. 通过雅可比求关节速度增量
        J_arm = J_full[:, self.arm_joint_ids]  # 取出机械臂相关关节
        dq = np.linalg.pinv(J_arm) @ delta_ee_pos  # 伪逆求解 Δq

        # 4️⃣ 计算目标关节位置
        qpos_target = qpos + dq

        # 5️⃣ 生成 PD 力控制
        torque = kp * (qpos_target - qpos) - kd * qvel
        return torque

    def reset(self):
        mujoco.mj_resetData(self.robot_model, self.robot_data)
        # moma robot
        self._set_arm_joint_pos(self.momaDefaultDofPos[:6])
        self._set_gripper_state(1)
        self._set_base_pos(self.momaDefaultStatePos)

        self.arm_pid.reset()
        # test arm
        # self.arm_data.qpos[0:6] = self.momaDefaultDofPos[:6]
        mujoco.mj_forward(self.robot_model, self.robot_data)
        self.viewer.sync()
        # IK URDF ARM
        self.ik_arm_controller.set_joint_state(self.momaDefaultDofPos[:6])
        pos, orn = self.ik_arm_controller.get_ee_pose()
        self.arm_ee_pos = np.concatenate([pos, orn])
        # self.render()

    def step(self, action):
        """
        action = [base_qvel(3), delta_ee_pos(6), gripper_state(1)]
        """

        base_qvel = action[:2]
        delta_ee_pos = action[2:-1]
        gripper_state = action[-1]
        # self._set_base_pos(self.momaDefaultStatePos)
        self._set_base_qvel(base_qvel)
        self._move_arm(delta_ee_pos)
        self._set_gripper_state(1)
        mujoco.mj_forward(self.robot_model, self.robot_data)
        mujoco.mj_step(self.robot_model, self.robot_data)
        # mujoco.mj_rnePostConstraint(self.robot_model, self.robot_data)
        # print('self.robot_data.ctrl:      {}'.format(self.robot_data.ctrl))

        obs = self._get_observation()
        return obs  # 这里你可以返回 obs, reward, done, info

    def run_simulation(self, action):
        obs = self.step(action)
        self.render()
        self.cnt += 1
        return obs

    def run_simulation_set_state(self, state):
        # base_state = state[:7]
        base_state = state[:7]
        self.robot_data.qvel[:] = 0
        arm_dof = state[-6:]
        # base_state = np.array([-0.5, -0.5, 0, 0, 0, 0, 1])
        # arm_dof = np.array([0, 0, 0, 0, 0, 0])
        gripper_state = 1
        base_pos = np.array(base_state[:3])
        base_quat = np.roll(base_state[3:], 1)
        # self._set_base_pos(self.momaDefaultStatePos)
        self._set_base_pos(np.hstack([base_pos, base_quat]))
        self._set_arm_joint_pos(arm_dof)
        self._set_gripper_state(gripper_state)

        mujoco.mj_forward(self.robot_model, self.robot_data)
        mujoco.mj_step(self.robot_model, self.robot_data)
        # mujoco.mj_rnePostConstraint(self.robot_model, self.robot_data)
        self.render()
        self.cnt += 1




if __name__ == '__main__':
    cfg_path = '/home/island/Desktop/mobile_manipulation/realrobot/moma_mujoco/cfg'
    cfg_yaml = 'config.yaml'
    config = load_config(os.path.join(cfg_path, cfg_yaml))
    mujoco_robot_env = MujocoRobotEnv(config)
    mujoco_robot_env.run_simulation()



