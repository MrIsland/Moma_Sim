#!/home/island/.conda/envs/ovir3d/bin/python3
import numpy as np
import rospkg
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from model.PointNavResNetNet import *
from rl_games.algos_torch import model_builder, torch_ext
from copy import copy
from model.common import *
import pybullet as p
from scipy.spatial.transform import Rotation as R
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_rotation_6d
from cv_bridge import CvBridge

class ModelInference:
    def __init__(self):
        config_path = './configs/model_config.yaml'
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('moma_perception')
        self.cfg = load_config(os.path.join(package_path, config_path))
        self.img_h, self.img_w = self.cfg['params']['network']['visual_encoder']['img_h'], self.cfg['params']['network']['visual_encoder']['img_w']

        self.device = 'cuda:0'
        self.ckpt_path = self.cfg['params']['checkpoint'] if 'checkpoint' in self.cfg['params'] and self.cfg['params']['checkpoint'] != '' else None
        self.ckpt_path = os.path.join()
        model_builder.register_network('pointnavresnetnet', lambda **kwargs: PointNavResNetBuilder())
        # 观测向量维度: eef_pos(3)  + eef_6d(6) + robot_vel(6) + grasp_pos_vector(640) = 661
        obs_shape = 3 + 6 + 6 + 640
        obs_shape = np.zeros(obs_shape).shape
        self.config = {
            'actions_num': 2 + 6 + 1,
            'input_shape': obs_shape,
            'num_seqs': 1,
            'value_size': 1,
            'normalize_value': True,
            'normalize_input': True,
        }
        builder = model_builder.ModelBuilder()
        self.model = builder.load(self.cfg['params']).build(self.config)
        self.model.to(self.device)
        self.model.eval()
        self.states = {}
        self.clip_actions = True
        self.actions_low = np.ones(self.config['actions_num']) * -1
        self.actions_high = np.ones(self.config['actions_num'])
        self.actions_low = to_torch(self.actions_low)
        self.actions_high = to_torch(self.actions_high)

    def restore(self, ckpt_path=None):
        if ckpt_path is None:
            ckpt_path = self.ckpt_path
        checkpoint = torch_ext.load_checkpoint(ckpt_path)
        self.model.load_state_dict(checkpoint['model'])
        if self.normalize_input and 'running_mean_std' in checkpoint:
            self.model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

    def refresh_observation(self, data_package, mode, graspnet_result=None):
        bridge = CvBridge()
        # depth = bridge.imgmsg_to_cv2(data_package['depth_msg'], desired_encoding="passthrough")
        eepos = [
            data_package['epos_msg'].position.x,
            data_package['epos_msg'].position.y, 
            data_package['epos_msg'].position.z,
            *R.from_quat([
                data_package['epos_msg'].orientation.x,
                data_package['epos_msg'].orientation.y,
                data_package['epos_msg'].orientation.z,
                data_package['epos_msg'].orientation.w
            ]).as_euler('xyz')
        ]
        twist = [
            data_package['twist_msg'].linear.x,
            data_package['twist_msg'].linear.y,
            data_package['twist_msg'].linear.z,
            data_package['twist_msg'].angular.x,
            data_package['twist_msg'].angular.y,
            data_package['twist_msg'].angular.z
        ]
        # gripper_state = np.array(data_package['gripper_state_msg'].data)
        # gripper_state = np.array([gripper_state, -1 * gripper_state, gripper_state, gripper_state, -1 * gripper_state, gripper_state])

        
        if mode =='state':
            bottle_pos = np.array([0.6, 0.6, 0.84])
            bottle_mat = np.array([[1,0,0],[0,1,0],[0,0,1]])
            
            grasp_poses_factory = self.compute_good_grasp_pos(bottle_pos[None, :], bottle_mat)
            grasp_pos_tensor = self.convert_grasp_pose_to_features(to_torch(grasp_poses_factory))
            # grasp_pos_tensor, _ = self.generate_random_transform_batch(bottle_pos, num_samples=64)
            epos_quat = np.roll((p.getQuaternionFromEuler(eepos[3:])), 1)
            self.states.update({
                'eef_pos': to_torch(eepos[:3]).unsqueeze(0),
                'eef_6d':matrix_to_rotation_6d(quaternion_to_matrix(to_torch(epos_quat))).unsqueeze(0),
                'robot_vel': to_torch(twist[:6]).unsqueeze(0),
                # 'q_gripper': to_torch(gripper_state[-6:]).unsqueeze(0),
                'grasp_pos_vector': grasp_pos_tensor.reshape(1, -1)
            })
            # print('eef_pos:{};eef_6d:{}'.format(self.states['eef_pos'], self.states['eef_6d']))
            
            
        if mode == 'vision':
            grasp_pos_tensor = self.extract_grasp_poses_from_graspnet(graspnet_result)
            
            curr_depth_tensor = to_torch(depth).unsqueeze(0)
            curr_depth_tensor = torch.nan_to_num(curr_depth_tensor, posinf=1e10, neginf=0)
            curr_depth_tensor = curr_depth_tensor.unsqueeze(-1).permute(0, 3, 1, 2)
            curr_depth_tensor = F.interpolate(curr_depth_tensor,
                                            size=(self.img_h, self.img_w),
                                            mode='bilinear',
                                            align_corners=False)
            self.states.update({
                'eef_pos': to_torch(eepos[:3]).unsqueeze(0),
                'eef_6d':matrix_to_rotation_6d(quaternion_to_matrix(to_torch(epos_quat))).unsqueeze(0),
                'robot_vel': to_torch(twist[:6]).unsqueeze(0),
                # 'q_gripper': to_torch(gripper_state[-6:]).unsqueeze(0),
                'curr_depth_img_tensor_batch': curr_depth_tensor.reshape(1, -1),
                'grasp_pos_vector': grasp_pos_tensor.reshape(-1).unsqueeze(0)
            })
            
    def extract_grasp_poses_from_graspnet(self, graspnet_result):
        graspnet_result.sort_by_score()
        translations = graspnet_result.translations  # shape: [N, 3]
        rotation_matrices = graspnet_result.rotation_matrices  # shape: [N, 3, 3]
        widths = graspnet_result.widths  # shape: [N,]

        best_index = self.fusion_graspnet_group(graspnet_result)
        selected_translation = translations[best_index]  # [3]
        selected_rotation = rotation_matrices[best_index]  # [3, 3]
        selected_width = widths[best_index]  # scalar
        
        # 创建64个相同的抓取姿态
        pos_tensor = to_torch(selected_translation).unsqueeze(0).repeat(64, 1)  # [64, 3]
        rot_tensor = to_torch(selected_rotation).unsqueeze(0).repeat(64, 1, 1)  # [64, 3, 3]
        width_tensor = to_torch(selected_width).unsqueeze(0).unsqueeze(1).repeat(64, 1)  # [64, 1]

        rot_cols = rot_tensor[:, :, :2].reshape(64, 6)  # [64, 6]
        transforms = torch.cat([pos_tensor, rot_cols, width_tensor], dim=1)  # [64, 10]
    
        return transforms
    
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

        
    def fusion_graspnet_group(self, graspnet_result):
        """抓取pos融合函数"""
        if len(graspnet_result) > 0:
            return 0  # 返回最高分数的抓取索引
        else:
            return None  # 如果没有抓取，返回None
    
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
    
    def preproc_obs(self, obs_batch):
        if type(obs_batch) is dict:
            obs_batch = copy.copy(obs_batch)
            for k, v in obs_batch.items():
                if v.dtype == torch.uint8:
                    obs_batch[k] = v.float() / 255.0
                else:
                    obs_batch[k] = v
        else:
            if obs_batch.dtype == torch.uint8:
                obs_batch = obs_batch.float() / 255.0
        return obs_batch
    
    def get_action(self, data_package, graspnet_result, mode, is_deterministic = True):
        obs = ["eef_pos", "eef_6d", "robot_vel", "grasp_pos_vector"]
        if data_package is not None:
            self.refresh_observation(data_package, mode, graspnet_result)
        
        # print('#############################################')
        # print('eef_pos:       {}'.format(self.states['eef_pos']))
        # print('eef_6d:        {}'.format(self.states['eef_6d']))
        # print('robot_vel:     {}'.format(self.states['robot_vel']))
        # print('grasp_pos_vector:     {}'.format(self.states['grasp_pos_vector']))
        obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)
        obs = self.preproc_obs(obs_buf)
        
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs' : obs,
            'rnn_states' : self.states
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)
        mu = res_dict['mus']
        action = res_dict['actions']
        self.states = res_dict['rnn_states']
        if is_deterministic:
            current_action = mu
        else:
            current_action = action

        # print('get_action - mu:      {}'.format(mu))

        if self.clip_actions:
            return rescale_actions(self.actions_low, self.actions_high, torch.clamp(current_action, -1.0, 1.0)).cpu().numpy()
        else:
            return current_action.cpu().numpy()
