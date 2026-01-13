import math
import os, sys
import numpy as np
import cv2
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import torch
import pdb
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(ROOT_DIR)
sys.path.append(ROOT_DIR)
from utils.GraspFusionUnion import GraspFusion
# from utils.GraspFusion import GraspFusion
# from utils.GraspFusion import GraspFusion
from utils.common import to_torch, mov
import yaml
import open3d as o3d



class GymGraspSensor:
    def __init__(self, config, asset_root, urdf_path, output_dir="output"):
        custom_parameters = [
            {"name": "--controller", "type": str, "default": "ik",
             "help": "Controller to use for Franka. Options are {ik, osc}"},
            {"name": "--num_envs", "type": int, "default": 2, "help": "Number of environments to create"},
        ]
        self.args = gymutil.parse_arguments(
            description="Franka Jacobian Inverse Kinematics (IK) + Operational Space Control (OSC) Example",
            custom_parameters=custom_parameters,
        )
        self.config = config['graspfusioner']
        self.gym = gymapi.acquire_gym()
        self.asset_root = asset_root
        self.urdf_path = urdf_path
        self.output_dir = output_dir
        self.sim = None
        self.viewer = None
        self.env = None
        self.camera_handle = None
        self.camera_props = gymapi.CameraProperties()
        self.frame_idx = 0
        self.radius = 0.5
        self.angle = 0.0
        self.angle_speed = 0.01
        self.depth_bar = 1.2
        self.segmentation_id = {
            'table': 1,
            'bottle': 2
        }
        self.device = "cuda:0"

        self._init_sim()
        self._create_env()
        self._create_viewer()
        self._load_cameras()
        os.makedirs(self.output_dir, exist_ok=True)

        camera_u = torch.arange(0, self.camera_props.width)
        camera_v = torch.arange(0, self.camera_props.height)
        self.camera_v2, self.camera_u2 = torch.meshgrid(camera_v, camera_u, indexing='ij')
        self.camera_u2 = to_torch(self.camera_u2, device=self.device)
        self.camera_v2 = to_torch(self.camera_v2, device=self.device)
        self.grasp_pose_factory = GraspFusion(self.config)

    def _init_sim(self):
        device = self.args.sim_device if self.args.use_gpu_pipeline else 'cpu'
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        sim_params.dt = 1.0 / 60.0
        sim_params.substeps = 2
        sim_params.use_gpu_pipeline = self.args.use_gpu_pipeline
        if self.args.physics_engine == gymapi.SIM_PHYSX:
            sim_params.physx.solver_type = 1
            sim_params.physx.num_position_iterations = 8
            sim_params.physx.num_velocity_iterations = 1
            sim_params.physx.rest_offset = 0.0
            sim_params.physx.contact_offset = 0.001
            sim_params.physx.friction_offset_threshold = 0.001
            sim_params.physx.friction_correlation_distance = 0.0005
            sim_params.physx.num_threads = self.args.num_threads
            sim_params.physx.use_gpu = self.args.use_gpu
        else:
            raise Exception("This example can only be used with PhysX")


        self.sim = self.gym.create_sim(self.args.compute_device_id, self.args.graphics_device_id, self.args.physics_engine, sim_params)
        if self.sim is None:
            raise Exception("Failed to create sim")

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

    def _load_cameras(self):
        raw_depth_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.env, self.camera_handle, gymapi.IMAGE_DEPTH)
        self.depth_tensor = gymtorch.wrap_tensor(raw_depth_tensor)

        raw_rgb_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.env, self.camera_handle, gymapi.IMAGE_COLOR)
        self.rgb_tensor = gymtorch.wrap_tensor(raw_rgb_tensor)
        # print('1.  self.rgb_tensor:    {}'.format(self.rgb_tensor))
        self.rgb_tensor = torch.ones_like(self.rgb_tensor) * 155
        # print('2.  self.rgb_tensor:    {}'.format(self.rgb_tensor))

        raw_seg_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.env, self.camera_handle, gymapi.IMAGE_SEGMENTATION)
        self.seg_tensor = gymtorch.wrap_tensor(raw_seg_tensor)

    def _create_viewer(self):
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())

    def _create_env(self):
        self.env = self.gym.create_env(self.sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1)

        # 添加桌子
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_opts.disable_gravity = True
        table_asset = self.gym.create_box(self.sim, 0.6, 1.0, 0.4, table_opts)

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.0, 0.0, 0.2)
        print('self.segmentation_id-table:  {}'.format(self.segmentation_id['table']))
        table_handle = self.gym.create_actor(self.env, table_asset, table_pose, "table", 0, -1, segmentationId=self.segmentation_id['table'])
        self.gym.set_rigid_body_color(self.env, table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                      gymapi.Vec3(0.5, 0.5, 0.5))

        # 添加瓶子
        bottle_opts = gymapi.AssetOptions()
        bottle_opts.fix_base_link = False
        bottle_opts.disable_gravity = False
        bottle_asset = self.gym.load_asset(self.sim, self.asset_root, self.urdf_path, bottle_opts)

        bottle_pose = gymapi.Transform()
        bottle_pose.p = gymapi.Vec3(0.0, 0.0, 0.545)
        print('self.segmantation_id-bottle:   {}'.format(self.segmentation_id['bottle']))
        bottle_handle = self.gym.create_actor(self.env, bottle_asset, bottle_pose, "bottle", 0, -1, segmentationId=self.segmentation_id['bottle'])
        self.gym.set_rigid_body_color(self.env, bottle_handle, 0, gymapi.MESH_VISUAL,
                                      gymapi.Vec3(238 / 255, 162 / 255, 164 / 255))

        # 创建相机
        self.camera_props.width = 640
        self.camera_props.height = 480
        self.camera_props.horizontal_fov = 60.0
        self.camera_props.enable_tensors = True
        self.camera_handle = self.gym.create_camera_sensor(self.env, self.camera_props)

        # 准备仿真
        self.gym.prepare_sim(self.sim)
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

    #@Brief:
    #-return: point: (batch_size, point_num, 7)   [x, y, z, r, g, b, label]
    #-return: valid: (batch_size, point_num)
    def _depth_image_to_point_cloud_GPU_batch(self,
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

    def _update_camera_orbit(self):
        cam_pos = gymapi.Vec3(self.radius * math.cos(self.angle), self.radius * math.sin(self.angle), 0.5)
        cam_transform = gymapi.Transform(cam_pos, gymapi.Quat.from_euler_zyx(0, 0, 0))
        # self.gym.set_camera_transform(self.camera_handle, self.env, cam_transform)
        self.gym.set_camera_location(self.camera_handle, self.env, cam_pos, gymapi.Vec3(0, 0, 0.5))

        # print(self.gym.get_camera_view_matrix(self.sim, self.env, self.camera_handle))
        self.vinv_mat = torch.inverse(
            (to_torch(self.gym.get_camera_view_matrix(self.sim, self.env, self.camera_handle), device=self.device)))

        self.proj_mat = to_torch(self.gym.get_camera_proj_matrix(self.sim, self.env, self.camera_handle), device=self.device)

    def _get_frame_pcd(self):
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        point, valid = self._depth_image_to_point_cloud_GPU_batch(self.depth_tensor.unsqueeze(0),
                                                                  self.rgb_tensor.unsqueeze(0),
                                                                  self.seg_tensor.unsqueeze(0),
                                                                  self.vinv_mat.unsqueeze(0),
                                                                  self.proj_mat.unsqueeze(0),
                                                                  self.camera_u2,
                                                                  self.camera_v2,
                                                                  self.camera_props.width,
                                                                  self.camera_props.height,
                                                                  self.depth_bar,
                                                                  self.device)
        # if self.rgb_tensor is not None:
        #     color_np = self.rgb_tensor.cpu().numpy()
        #     print('color_np.shape:  {}'.format(color_np.shape))
        #     color_np = np.reshape(color_np, (self.camera_props.height, self.camera_props.width, 4))[:, :, :3]
        #     cv2.imwrite(f"{self.output_dir}/rgb_{self.frame_idx:03d}.png", cv2.cvtColor(color_np, cv2.COLOR_RGB2BGR))
        #
        # if self.depth_tensor is not None:
        #     depth_np = self.depth_tensor.cpu().numpy()
        #     depth_np = np.reshape(depth_np, (self.camera_props.height, self.camera_props.width))
        #     cv2.imwrite(f"{self.output_dir}/depth_{self.frame_idx:03d}.png", (depth_np * 255).astype(np.uint8))
        #
        # if self.seg_tensor is not None:
        #     seg_np = self.seg_tensor.cpu().numpy()
        #     unique_ids = np.unique(seg_np)
        #     print('unique_ids:   {}'.format(unique_ids))
        #     seg_np = np.reshape(seg_np, (self.camera_props.height, self.camera_props.width))
        #     cv2.imwrite(f"{self.output_dir}/seg_{self.frame_idx:03d}.png", seg_np.astype(np.uint8) * 80)

        self.gym.end_access_image_tensors(self.sim)
        return point, valid

    def run(self):
        # sphere_asset_options = gymapi.AssetOptions()
        # sphere_asset_options.disable_gravity = True
        # sphere_asset = self.gym.create_sphere(self.sim, 0.02, sphere_asset_options)
        # sphere_actor = self.gym.create_actor(self.env, sphere_asset, pose, f"point_marker_{self.frame_idx}", self.frame_idx, 0, 0)
        while not self.gym.query_viewer_has_closed(self.viewer):
            self._update_camera_orbit()

            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            # 更新传感器数据
            self._refresh()
            frame_pcd, frame_valid = self._get_frame_pcd()
            if self.frame_idx % 10 == 0:
                frame_cloud = self.grasp_pose_factory.add_new_frame(frame_pcd.squeeze(0), frame_valid.squeeze(0), self.frame_idx)

            self.angle = self.frame_idx * self.angle_speed
            self.angle = (self.angle + math.pi) % (2 * math.pi) - math.pi

            self.frame_idx += 1
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)

            # if self.frame_idx % 10 == 0:
            #     # o3d.visualization.draw_geometries(self.grasp_pose_factory.pcd_list)
            #     tmp_frame_grasp_group = self.grasp_pose_factory._get_gg_from_cur_scene_node()
            #     self.grasp_pose_factory.grasp_pos_detector.vis_grasp_pos(tmp_frame_grasp_group, frame_cloud)

        self.cleanup()

    def _refresh(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

    def cleanup(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

if __name__ == "__main__":
    asset_root = "../../assets"
    urdf_path = "urdf/object/meshdatav3_scaled/sem/Bottle-1d4093ad2dfad9df24be2e4f911ee4af/coacd/coacd_015.urdf"

    with open('../cfg/example/grasp_fusion.yaml', 'r') as file:
        config = yaml.safe_load(file)

    sensor = GymGraspSensor(config, asset_root, urdf_path)
    sensor.run()


