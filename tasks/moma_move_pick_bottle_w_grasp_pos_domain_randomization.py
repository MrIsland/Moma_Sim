import pdb
import time

import numpy as np
import os
import copy

from isaacgym import gymtorch
from isaacgym import gymapi
import torch
import torch.nn.functional as F
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_rotation_6d
import math

from isaacgym.torch_utils import quat_rotate
from isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, tensor_clamp, quat_apply
from isaacgymenvs.tasks.base.vec_task import VecTask
from utils.common import random_z_rotation_quaternion, mov, pose_to_matrix, pose_to_matrix_batch, \
    rotmat_to_euler_xyz_batch, opengl_trans_opencv, get_quat_180, pre_process_rgb_img, accumulate_reward, debug_vis_draw_scalar
# from utils.GraspFusion import GraspFusion
from utils.GraspFusionUnion import GraspFusion
from scipy.spatial.transform import Rotation as R
import cv2
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
from utils.common import load_urdf_with_txt, sample_from_regions_np, low_pass_filter

from learning.depth_anything_v2_encoder import DepthAnythingV2Encoder

DEBUG_REWARD = False
######################################################
# @brief:
######################################################

"""
self.gym.get_actor_joint_dict(env_ptr, moma_handle): [
    ## Active joint
    arm:  'joint1': 1, 'joint2': 2, 'joint3': 3, 'joint4': 4, 'joint5': 5, 'joint6': 6, 
    base:   'joint_bucl': 17,'joint_bclw': 18, 'joint_bucr': 19, 'joint_bcrw': 20, 'joint_fucl': 21, 'joint_fclw': 22, 'joint_fucr': 23, 'joint_fcrw': 24, 'joint_lw': 25, 'joint_rw': 26,
    gripper: 'finger_joint': 9, 'leftinn_joint': 10, 'left_kckle_joint': 12,  'right_kckle_joint': 13, 'rightout_joint': 14 'rightinn_joint': 15,

    ## Non-active joint
    'joint_arm_base': 0, 'gripper_site_joint': 7, 'joint_7': 8, 'left_pad_joint': 11, 'right_pad_joint': 16,
]

get_actor_dof_names:[
    arm:  'joint1': 0, 'joint2': 1, 'joint3': 2, 'joint4': 3, 'joint5': 4, 'joint6': 5, 
    gripper: 'finger_joint': 6, 'leftinn_joint': 7, 'left_kckle_joint': 8,  'right_kckle_joint': 9, 'rightout_joint': 10, 'rightinn_joint': 11,
    base:   'joint_bucl': 12,'joint_bclw': 13, 'joint_bucr': 14, 'joint_bcrw': 15, 'joint_fucl': 16, 'joint_fclw': 17, 'joint_fucr': 18, 'joint_fcrw': 19, 'joint_lw': 20, 'joint_rw': 21,
]

get_actor_rigid_body_names:[
    'base_link': 0, 'arm_base_link': 1, 'arm_link1': 2, 'arm_link2': 3, 'arm_link3': 4, 'arm_link4': 5, 'arm_link5': 6, 'arm_link6': 7, 'gripper_site': 8,
    'link_7': 9, 'leftout_Link': 10, 'leftinn_Link': 11, 'left_pad': 12, 'left_kckle': 13, 'right_kckle': 14, 'rightout_Link': 15, 'rightinn_Link': 16, 'right_pad': 17,
    'link_bucl': 18, 'link_bclw': 19, 'link_bucr': 20, 'link_bcrw': 21, 'link_fucl': 22, 'link_fclw': 23, 'link_fucr': 24, 'link_fcrw': 25, 'link_lw': 26, 'link_rw': 27
]
"""


@torch.jit.script
def axisangle2quat(vec, eps=1e-6):
    """
    Converts scaled axis-angle to quat.
    Args:
        vec (tensor): (..., 3) tensor where final dim is (ax,ay,az) axis-angle exponential coordinates
        eps (float): Stability value below which small values will be mapped to 0

    Returns:
        tensor: (..., 4) tensor where final dim is (x,y,z,w) vec4 float quaternion
    """
    # type: (Tensor, float) -> Tensor
    # store input shape and reshape
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    # Grab angle
    angle = torch.norm(vec, dim=-1, keepdim=True)

    # Create return array
    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 3] = 1.0

    # Grab indexes where angle is not zero an convert the input to its quaternion form
    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat([
        vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
        torch.cos(angle[idx, :] / 2.0)
    ], dim=-1)

    # Reshape and return output
    quat = quat.reshape(list(input_shape) + [4, ])
    return quat


class MomaMovePickBottleWGraspPosDR(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.experiment_name = self.cfg['name'] + datetime.now().strftime("_%d-%H-%M-%S")

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.moma_position_noise = self.cfg["env"]["momaPositionNoise"]
        self.moma_rotation_noise = self.cfg["env"]["momaRotationNoise"]
        self.table_height_noise = self.cfg["env"]["tableHeightNoise"]
        self.table_position_noise = self.cfg["env"]["tablePositionNoise"]
        self.table_rotation_noise = self.cfg["env"]["tableRotationNoise"]
        self.moma_dof_noise = self.cfg["env"]["momaDofNoise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.cfg_visual_encoder = self.cfg["visual_encoder"]

        # Create dicts to pass to reward function
        self.reward_settings = {
            "r_dist_scale": self.cfg["env"]["distRewardScale"],
            "r_lift_scale": self.cfg["env"]["liftRewardScale"],
            "r_lift_always_scale": self.cfg["env"]["liftRewardScale_always"],
            "r_target_pos_scale": self.cfg["env"]["targetposRewardScale"],
            "r_o2g_reward_scale": self.cfg["env"]["o2gRewardScale"],
            "r_d_gripper_grasp_pos": self.cfg["env"]["flagSuccessRewardGripperGraspRewardScale"],
            "o2g_beta1": self.cfg["env"]["o2gBeta1"],
            "o2g_beta2": self.cfg["env"]["o2gBeta2"],
            "p_gripper_scale": self.cfg["env"]["gripperPenaltyScale"],
            "p_collision_scale": self.cfg["env"]["collisionPenaltyScale"],
            "p_action_jitter_scale": self.cfg["env"]["actionjitterPenaltyScale"],
        }

        # Controller type
        self.control_type = self.cfg["env"]["controlType"]
        assert self.control_type in {"osc", "joint_tor", "ik"}, \
            "Invalid control type specified. Must be one of: {osc, joint_tor, ik}"

        # dimensions
        # obs include: bottleA_pose (7) + base_pose(7) + eef_pose (6) + q_gripper (6)
        # obs include: bottleA_pose (7) + eef_pose (7) + q_gripper (6)  20
        # obs include: bottleA_pose (7) + eef_pose (7) + q_gripper (6) + target_bottleA_pose(7) 28
        # obs include: bottleA_pose (7) + eef_pose (7) + q_gripper (6) + bottleA_pos_relative_robot_2d(2) + robot_vel(6)  = 28
        num_visual_observation = self.cfg_visual_encoder['img_h'] / 14 * self.cfg_visual_encoder['img_w'] / 14 * 64   # (patch_h * patch_w * vits)
        self.cfg["env"]["numObservations"] = 15 + 640
        self.cfg["env"]["numGraspObservations"] = 640
        self.cfg["env"]["numVisionObservations"] = int(num_visual_observation)
        # actions include: delta EEF if OSC (6) or joint torques (7) + bool gripper (1)
        # actions include: base_pose(2) + delta EEF if OSC (6) or joint torques (7) + bool gripper (1)
        self.cfg["env"]["numActions"] = 9
        self.lpf_alpha = self.cfg['env']['LPF']['alpha']

        # Values to be filled in at runtime
        self.states = {}  # will be dict filled with relevant states to use for reward calculation
        self.handles = {}  # will be dict mapping names to relevant sim handles
        self.num_dofs = None  # Total number of DOFs per env
        self.actions = None  # Current actions to be deployed
        self.prev_actions = None  # Prevent actions to be deployed
        self._init_bottleA_state = None  # Initial state of bottleA for the current env
        # self._init_cubeB_state = None           # Initial state of cubeB for the current env
        self._bottleA_state = None  # Current state of bottleA for the current env
        # self._cubeB_state = None                # Current state of cubeB for the current env
        self._bottleA_id = None  # Actor ID corresponding to bottleA for a given env
        # self._cubeB_id = None                   # Actor ID corresponding to cubeB for a given env
        self._table_id = None
        self._obstacle_ids = None
        # Tensor placeholders
        self._root_state = None  # State of root body        (n_envs, 13)
        self._dof_state = None  # State of all joints       (n_envs, n_dof)
        self._q = None  # Joint positions           (n_envs, n_dof)
        self._qd = None  # Joint velocities          (n_envs, n_dof)
        self._rigid_body_state = None  # State of all rigid bodies             (n_envs, n_bodies, 13)
        self._contact_forces = None  # Contact forces in sim
        self._eef_state = None  # end effector state (at grasping point)
        self._eef_lf_state = None  # end effector state (at left fingertip)
        self._eef_rf_state = None  # end effector state (at left fingertip)
        self._j_eef = None  # Jacobian for end effectorf_arm_control
        self._mm = None  # Mass matrix
        self._arm_control = None  # Tensor buffer for controlling arm
        self._gripper_control = None  # Tensor buffer for controlling gripper
        self._pos_control = None  # Position actions
        self._effort_control = None  # Torque actions
        self._moma_effort_limits = None  # Actuator effort limits for moma
        self._global_indices = None  # Unique indices corresponding to all envs in flattened array
        self._init_states = None  # Initial states
        self._constant_base_vel = None # Constant vel of the base
        self.flag_collision = False
        self._target_visible = None # Whether the object is in the field of view
        self.last_ori_grasp_pos_tensor = None
        self.ori_grasp_pos_tensor = None
        self.curr_robot_vel = None
        self.frame_idx = 0
        self.global_idx = 0
        # self.vinv_matrices = []
        # self.proj_matrices = []
        self.camera_actors = []
        self.camera_rgb_tensor_list = []
        self.camera_depth_tensor_list = []
        self.camera_seg_tensor_list = []

        self.obj_asset_list = []

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.debug_vis_grasp_pos = self.cfg["env"]["enableDebugVis_grasppos"]

        self.up_axis = "z"
        self.up_axis_idx = 2
        self.temperature = 0.5

        self._contact_bodies = ['arm_base_link', 'arm_link1', 'arm_link2', 'arm_link3',
                                'arm_link4', 'arm_link5', 'arm_link6', 'link_7',
                                'leftout_Link', 'leftinn_Link', 'left_pad', 'left_kckle', 'right_kckle', 'rightout_Link',
                                'rightinn_Link', 'right_pad']
        self._contact_obs = ['obs_0', 'obs_1', 'obs_2']
        self._contact_wall = ['wall_0', 'wall_1', 'wall_2', 'wall_3']
        self.segmentation_id = {
            'bottleA': 1
        }
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device,
                         graphics_device_id=graphics_device_id, headless=headless,
                         virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        self.moma_default_dof_pos = to_torch(
            # [-0.67, 0.0, 0.65, 0.0, 1.36, 1.57, 0.5, -0.5, 0.5, 0.5, 0.5, -0.5], device=self.device
            [-0.67, 0.0, 0.8, 0.0, 1.36, 3.14, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0], device=self.device
            # [0.0, 0.0, 0.0, 0.0, 2.0, 3.14, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0], device=self.device
        )
        self.moma_default_state_pos = to_torch(
            # [-0.2, -0.5, 0.0, 0.0, 0.0, 0.707, 0.707], device=self.device
            # [-0.6, -0.6, 0.0, 0.0, 0.0, 0.707, 0.707], device=self.device
            # [-0.1, -0.8, 0.0, 0.0, 0.0, 0.707, 0.707], device=self.device
            # [-0.6, 0.6, 0.0, 1.0, 0.0, 0.0, 0.0], device=self.device
            [-0.8, 0.8, 0.0, 0.0, 0.0, 0.0, 1.0], device=self.device
        )

        self.grasp_pose_factories = []
        for i in range(self.num_envs):
            grasp_pose_factory = GraspFusion(self.cfg['graspfusioner'])
            self.grasp_pose_factories.append(grasp_pose_factory)

        # 需要检测碰撞的bodies
        # OSC Gains  eef_pose | joint
        self.kp = to_torch([150.] * 6, device=self.device)
        self.kd = 2 * torch.sqrt(self.kp)
        self.kp_null = to_torch([10.] * 6, device=self.device)
        self.kd_null = 2 * torch.sqrt(self.kp_null)
        # self.cmd_limit = None                   # filled in later
        #
        self.cmd_arm_limit = to_torch([0.05, 0.05, 0.05, 0.1, 0.1, 0.1], device=self.device).unsqueeze(0) if \
            self.control_type == "osc" or self.control_type == "ik" else self._moma_effort_limits[:7].unsqueeze(0)
        self.cmd_base_limit = to_torch([0.3, 0.6], device=self.device) # 0.2 m/s, 2 rad/s

        # Reset all environments
        self.score_pool, self.d_gg_pool, self.r_o2g_pool \
            = torch.zeros((self.num_envs, self.max_episode_length)), torch.zeros((self.num_envs, self.max_episode_length)), torch.zeros((self.num_envs, self.max_episode_length))
        self.score_pool_exp = torch.zeros((self.num_envs, self.max_episode_length))
        self.d_gripper_grasp_pos = torch.zeros((self.num_envs, self.max_episode_length))
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        self.depth_anything = DepthAnythingV2Encoder(encoder='vits', device=self.device)
        self.summaries_dir = os.path.join('runs', self.experiment_name, 'summaries')
        os.makedirs(self.summaries_dir, exist_ok=True)
        self.reward_writer = SummaryWriter(self.summaries_dir)
        self.reward_dict = {
            'reward_dist': None,
            'reward_dist_scale': None,
            'r_gg': None,
            'r_go': None,
            'r_o2g': None,
            'r_o2g_scale': None,
            'flag_success_gripper_grasp': None,
            'flag_success_gripper_grasp_scale': None,
        }
        self.reward_dict_episode = {
            'reward_dist': to_torch(torch.zeros(self.max_episode_length), device=self.device),
            'reward_dist_count': to_torch(torch.zeros(self.max_episode_length), device=self.device),

            'reward_dist_scale': to_torch(torch.zeros(self.max_episode_length), device=self.device),
            'reward_dist_scale_count': to_torch(torch.zeros(self.max_episode_length), device=self.device),

            'r_gg': to_torch(torch.zeros(self.max_episode_length), device=self.device),
            'r_gg_count': to_torch(torch.zeros(self.max_episode_length), device=self.device),

            'r_gg_scale': to_torch(torch.zeros(self.max_episode_length), device=self.device),
            'r_gg_scale_count': to_torch(torch.zeros(self.max_episode_length), device=self.device),

            'r_go': to_torch(torch.zeros(self.max_episode_length), device=self.device),
            'r_go_count': to_torch(torch.zeros(self.max_episode_length), device=self.device),

            'r_go_scale': to_torch(torch.zeros(self.max_episode_length), device=self.device),
            'r_go_scale_count': to_torch(torch.zeros(self.max_episode_length), device=self.device),

            'r_o2g': to_torch(torch.zeros(self.max_episode_length), device=self.device),
            'r_o2g_count': to_torch(torch.zeros(self.max_episode_length), device=self.device),

            'r_o2g_scale': to_torch(torch.zeros(self.max_episode_length), device=self.device),
            'r_o2g_scale_count': to_torch(torch.zeros(self.max_episode_length), device=self.device),

            'flag_success_gripper_grasp': to_torch(torch.zeros(self.max_episode_length), device=self.device),
            'flag_success_gripper_grasp_count': to_torch(torch.zeros(self.max_episode_length), device=self.device),

            'flag_success_gripper_grasp_scale': to_torch(torch.zeros(self.max_episode_length), device=self.device),
            'flag_success_gripper_grasp_scale_count': to_torch(torch.zeros(self.max_episode_length), device=self.device),
        }

        self.gym.simulate(self.sim)  # 物理仿真
        self.gym.fetch_results(self.sim, True)  # 等待物理仿真结束
        self.gym.step_graphics(self.sim)  # ⭐ 更新图形渲染
        self.gym.draw_viewer(self.viewer, self.sim, True)  # 在 viewer 中显示帧
        self.gym.sync_frame_time(self.sim)  # 同步到固定 frame rate
        # Refresh tensors
        self._refresh()

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._pre_load_asset()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.static_friction = 0.0  # 静摩擦系数
        plane_params.dynamic_friction = 0.0  # 动摩擦系数
        plane_params.restitution = 0.0  # 弹性（反弹程度）
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_walls(self):
        env_spacing = self.cfg['env']['envSpacing']  # 两个环境之间的距离
        self.wall_thickness = 0.05
        self.wall_length = 2 * env_spacing - 2 * self.wall_thickness
        self.wall_height = 1.0
        wall_asset_options = gymapi.AssetOptions()
        wall_asset_options.fix_base_link = True
        wall_asset = self.gym.create_box(self.sim, self.wall_length, self.wall_thickness, self.wall_height, wall_asset_options)
        return wall_asset

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        Moma_asset_file = "urdf/IslandMomaRobot/urdf/island_moma_robot.urdf"

        bottle_asset_file = "urdf/object/meshdatav3_scaled/sem/Bottle-f452c1053f88cd2fc21f7907838a35d1/coacd/coacd_012.urdf"
        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      self.cfg["env"]["asset"].get("assetRoot", asset_root))
            Moma_asset_file = self.cfg["env"]["asset"].get("assetFileNameMoma", Moma_asset_file)
            bottle_asset_file = self.cfg["env"]["asset"].get("assetFileNamebottle", bottle_asset_file)
        ## asset loading !!
        #1. load moma asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = False
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.use_mesh_materials = True
        self.moma_asset = self.gym.load_asset(self.sim, asset_root, Moma_asset_file, asset_options)

        bottle_asset_options = gymapi.AssetOptions()
        bottle_asset_options.density = 150
        bottle_asset_options.fix_base_link = True
        bottle_asset_options.disable_gravity = False
        bottle_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX  # or NONE if normals already exist
        bottle_asset_options.use_mesh_materials = True
        bottle_asset = self.gym.load_asset(self.sim, asset_root, bottle_asset_file, bottle_asset_options)

        moma_dof_stiffness = to_torch([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000], dtype=torch.float,
                                      device=self.device)
        moma_dof_damping = to_torch([50, 50, 50, 50, 50, 50, 100, 100, 100, 100, 100, 100], dtype=torch.float,
                                    device=self.device)

        #2. Create table asset
        table_thickness = self.cfg['env']['asset'].get('tableHeight', 0.7)  # 0.7
        table_pos = [0.0, 0.0, table_thickness / 2]
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True

        #3. Create wall asset
        wall_asset = self._create_walls()

        #4. Create camera asset
        link_to_camera_transform = gymapi.Transform()
        link_to_camera_transform.p = gymapi.Vec3(0.106, 0, 0)
        link_to_camera_transform.r = gymapi.Quat.from_euler_zyx(0, -np.pi / 2, np.pi)

        self.camera_asset = gymapi.CameraProperties()
        self.camera_asset.width = 640
        self.camera_asset.height = 480
        self.camera_asset.horizontal_fov = 60.0
        self.camera_asset.enable_tensors = True
        camera_u = torch.arange(0, self.camera_asset.width)
        camera_v = torch.arange(0, self.camera_asset.height)
        self.camera_v2, self.camera_u2 = torch.meshgrid(camera_v, camera_u, indexing='ij')
        self.camera_u2 = to_torch(self.camera_u2, device=self.device)
        self.camera_v2 = to_torch(self.camera_v2, device=self.device)

        ######################################################################
        self.bottleA_height = None

        # Create bottleA asset
        # bottleA_opts = gymapi.AssetOptions()
        # bottleA_asset = self.gym.create_box(self.sim, *([self.bottleA_size] * 3), bottleA_opts)
        bottleA_color = gymapi.Vec3(20 / 255, 74 / 255, 116 / 255)

        self.num_moma_bodies = self.gym.get_asset_rigid_body_count(self.moma_asset)
        self.num_moma_dofs = self.gym.get_asset_dof_count(self.moma_asset)

        print("num moma bodies: ", self.num_moma_bodies)
        print("num moma dofs: ", self.num_moma_dofs)

        moma_dof_props = self.gym.get_asset_dof_properties(self.moma_asset)
        self.moma_dof_lower_limits = []
        self.moma_dof_upper_limits = []
        self._moma_effort_limits = []

        for i in range(self.num_moma_dofs):
            if i > 11:
                break
            moma_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS if i > 5 else gymapi.DOF_MODE_POS
            if self.physics_engine == gymapi.SIM_PHYSX:
                moma_dof_props['stiffness'][i] = moma_dof_stiffness[i]
                moma_dof_props['damping'][i] = moma_dof_damping[i]
            else:
                moma_dof_props['stiffness'][i] = 7000.0
                moma_dof_props['damping'][i] = 50.0

            self.moma_dof_lower_limits.append(moma_dof_props['lower'][i])
            self.moma_dof_upper_limits.append(moma_dof_props['upper'][i])
            self._moma_effort_limits.append(moma_dof_props['effort'][i])

        self.moma_dof_lower_limits = to_torch(self.moma_dof_lower_limits, device=self.device)
        self.moma_dof_upper_limits = to_torch(self.moma_dof_upper_limits, device=self.device)
        self._moma_effort_limits = to_torch(self._moma_effort_limits, device=self.device)
        print('self.moma_dof_lower_limits:    {}'.format(self.moma_dof_lower_limits))
        print('self.moma_dof_upper_limits:    {}'.format(self.moma_dof_upper_limits))
        print('self._moma_effort_limits:    {}'.format(self._moma_effort_limits))
        self.moma_dof_speed_scales = torch.ones_like(self.moma_dof_lower_limits)
        # self.moma_dof_speed_scales[[6, 7, 8, 9, 10, 11]] = 0.1
        moma_dof_props['effort'][[6, 7, 8, 9, 10, 11]] = 200
        # moma_dof_props['effort'][8] = 200

        # Define start pose for moma
        moma_start_pose = gymapi.Transform()
        # moma_start_pose.p = gymapi.Vec3(-0.45, 0.0, 1.0 + table_thickness / 2 + table_stand_height)
        moma_start_pose.p = gymapi.Vec3(-0.7, -0.5, 0.0)
        moma_start_pose.r = gymapi.Quat(0.0, 0.0, 0.707, 0.707)

        # Define start pose for table
        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(*table_pos)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        table_x, table_y = self.cfg['env']['asset'].get('tableX', 0.0), self.cfg['env']['asset'].get('tableY', 0.0)
        self._table_surface_pos = np.array([table_x, table_y, table_thickness])
        self.reward_settings["table_height"] = self._table_surface_pos[2]

        # Define start pose for bottles (doesn't really matter since they're get overridden during reset() anyways)
        bottleA_start_pose = gymapi.Transform()
        bottleA_start_pose.p = gymapi.Vec3(-0.1, 0.0, 0.2)
        bottleA_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # compute aggregate size
        num_moma_bodies = self.gym.get_asset_rigid_body_count(self.moma_asset)
        num_moma_shapes = self.gym.get_asset_rigid_shape_count(self.moma_asset)
        # todo: 增加墙的时候这部分要修改
        max_agg_bodies = num_moma_bodies + 2 + 4 + 3  # 1 for table, 1 for bottle, 4 for walls, 3 for obstacles
        max_agg_shapes = num_moma_shapes + 2 + 4 + 3  # 1 for table, 1 for bottle, 4 for walls, 3 for obstacles
        print('max_agg_bodies:    {}'.format(max_agg_bodies))
        print('max_agg_shapes:    {}'.format(max_agg_shapes))
        self.momas = []
        self.envs = []
        self.ln_obj_asset = len(self.obj_asset_list)

        # Create environments
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: moma should ALWAYS be loaded first in sim!
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            moma_actor = self.gym.create_actor(env_ptr, self.moma_asset, moma_start_pose, "moma", i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, moma_actor, moma_dof_props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create table
            length, width = np.random.rand() * 0.2 + 0.1, np.random.rand() * 0.2 + 0.1
            table_asset = self.gym.create_box(self.sim, *[length, width, table_thickness], table_opts)
            self._table_id = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0)
            # self.gym.set_rigid_body_color(env_ptr, table_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, bottleA_color)
            # table_stand_actor = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_start_pose, "table_stand",
            #                                           i, 1, 0)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            center_x, center_y = 0, 0
            wall_poses = [
                gymapi.Transform(p=gymapi.Vec3(center_x, center_y - self.wall_length / 2, self.wall_height / 2)),  # 南墙
                gymapi.Transform(p=gymapi.Vec3(center_x, center_y + self.wall_length / 2, self.wall_height / 2)),  # 北墙
                gymapi.Transform(p=gymapi.Vec3(center_x - self.wall_length / 2, center_y, self.wall_height / 2),
                                 r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 1.5708)),  # 西墙，旋转90度
                gymapi.Transform(p=gymapi.Vec3(center_x + self.wall_length / 2, center_y, self.wall_height / 2),
                                 r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 1.5708)),  # 东墙，旋转90度
            ]
            self._wall_ids = torch.zeros(4).long()
            for j, pose in enumerate(wall_poses):
                wall_id = self.gym.create_actor(env_ptr, wall_asset, pose, f"wall_{j}", i, 3, 0)
                self._wall_ids[j] = wall_id

            # scene layout obstacle
            self._obstacle_ids = torch.zeros(3).long()
            for j in range(3):
                length, width = np.random.rand() * 0.3 + 0.1, np.random.rand() * 0.3 + 0.1
                table_asset = self.gym.create_box(self.sim, *[length, width, table_thickness], table_opts)
                obstacle_id = self.gym.create_actor(env_ptr, table_asset, table_start_pose, f"obstacle_{j}", i, 3, 0)
                self._obstacle_ids[j] = obstacle_id
            # load_camera
            link6_id = self.gym.find_actor_rigid_body_handle(env_ptr, moma_actor, 'arm_link6')
            camera_actor = self.gym.create_camera_sensor(env_ptr, self.camera_asset)
            self.gym.attach_camera_to_body(camera_actor, env_ptr, link6_id, link_to_camera_transform, gymapi.FOLLOW_TRANSFORM)
            self._load_camera(env_ptr, camera_actor)
            self.camera_actors.append(camera_actor)

            self._bottleA_id = self.gym.create_actor(env_ptr, self.obj_asset_list[i % self.ln_obj_asset], bottleA_start_pose, 'bottleA', i, 2, self.segmentation_id['bottleA'])
            if self.cfg['env']['asset']['assetFlagBottleOnly'] is False:
                self.gym.set_actor_scale(env_ptr, self._bottleA_id, self.scale_list[i % self.ln_obj_asset])
            self.gym.set_rigid_body_color(env_ptr, self._bottleA_id, 0, gymapi.MESH_VISUAL, bottleA_color)


            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.momas.append(moma_actor)

        # Setup init state buffer
        self._init_bottleA_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._init_table_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._init_obstacle_state = torch.zeros(self.num_envs, 3, 13, device=self.device)
        self._force_tensor = torch.zeros(max_agg_bodies * self.num_envs, 3, device=self.device)
        self._torque_tensor = torch.zeros(max_agg_bodies * self.num_envs, 3, device=self.device)
        self.base_body_indices = [i * max_agg_bodies for i in range(self.num_envs)]
        self.prev_actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self.filtered_actions = torch.zeros(self.num_envs, 2, device=self.device)   # [Low-pass filter in base velocity]
        # Setup data
        self.init_data()

    def _load_camera(self, env_ptr, camera_actor):
        raw_rgb_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_actor, gymapi.IMAGE_COLOR)
        rgb_tensor = gymtorch.wrap_tensor(raw_rgb_tensor)
        raw_depth_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_actor, gymapi.IMAGE_DEPTH)
        depth_tensor = gymtorch.wrap_tensor(raw_depth_tensor)
        raw_seg_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_actor, gymapi.IMAGE_SEGMENTATION)
        seg_tensor = gymtorch.wrap_tensor(raw_seg_tensor)
        self.camera_rgb_tensor_list.append(rgb_tensor)
        self.camera_depth_tensor_list.append(depth_tensor)
        self.camera_seg_tensor_list.append(seg_tensor)

    def _pre_load_asset(self):
        self.urdf_dataset_path = self.cfg['env']['asset']['assetDatasetPath']
        self.obj_list_txt = self.cfg['env']['asset']['assetDatasetObjListTxT']
        self.obj_scale_txt = self.cfg['env']['asset']['assetDatasetObjScaleListTxT']
        self.obj_height_txt = self.cfg['env']['asset']['assetDatasetObjHeightListTxT']
        self.urdf_list, self.scale_list, self.height_list = None, None, None
        if self.cfg['env']['asset']['assetFlagBottleOnly'] is False:
            self.urdf_list, self.scale_list, self.height_list = load_urdf_with_txt(self.urdf_dataset_path,
                                                                                   self.obj_list_txt,
                                                                                   self.obj_scale_txt,
                                                                                   self.obj_height_txt)
            for obj_asset_file in self.urdf_list:
                obj_asset_options = gymapi.AssetOptions()
                obj_asset_options.density = 150
                obj_asset_options.fix_base_link = True
                obj_asset_options.disable_gravity = False
                obj_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX  # or NONE if normals already exist
                obj_asset_options.use_mesh_materials = False
                obj_asset = self.gym.load_asset(self.sim, self.urdf_dataset_path, obj_asset_file, obj_asset_options)
                self.obj_asset_list.append(obj_asset)
        else:
            obj_asset_options = gymapi.AssetOptions()
            obj_asset_options.density = 150
            obj_asset_options.fix_base_link = True
            obj_asset_options.disable_gravity = False
            obj_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX  # or NONE if normals already exist
            obj_asset_options.use_mesh_materials = True
            asset_root, bottle_asset_file = None, None
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      self.cfg["env"]["asset"].get("assetRoot", asset_root))
            bottle_asset_file = self.cfg["env"]["asset"].get("assetFileNamebottle", bottle_asset_file)
            obj_asset = self.gym.load_asset(self.sim, asset_root, bottle_asset_file, obj_asset_options)
            self.obj_asset_list.append(obj_asset)

    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        moma_handle = 0
        self.handles = {
            # For calculation
            "left_pad": self.gym.find_actor_rigid_body_handle(env_ptr, moma_handle, "left_pad"),
            "right_pad": self.gym.find_actor_rigid_body_handle(env_ptr, moma_handle, "right_pad"),
            "gripper_site": self.gym.find_actor_rigid_body_handle(env_ptr, moma_handle, "gripper_site"),
            'arm_base_link': self.gym.find_actor_rigid_body_handle(env_ptr, moma_handle, 'arm_base_link'),

            # Table
            "table_handle": self.gym.find_actor_rigid_body_handle(env_ptr, self._table_id, 'box'),

            # Cubes
            "bottleA_body_handle": self.gym.find_actor_rigid_body_handle(env_ptr, self._bottleA_id, "link_001"),
        }

        for body in self._contact_bodies:
            # print('body:   {}'.format(body))
            self.handles.update({body: self.gym.find_actor_rigid_body_handle(env_ptr, moma_handle, body)})
        # Obstacle
        for i, obs in enumerate(self._obstacle_ids):
            self.handles.update({'obs_{}'.format(i): self.gym.find_actor_rigid_body_handle(env_ptr, obs, 'box')})
        # Wall
        for i, wall in enumerate(self._wall_ids):
            self.handles.update({'wall_{}'.format(i): self.gym.find_actor_rigid_body_handle(env_ptr, wall, 'box')})


        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)  # gymTensor(6, 13) 获得所有actor的根状态张量(13)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)  # gymTensor(44, 2) 获得每个关节的状态张量 (pos, vel)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)  # gymTensor(60, 13) 获取所有刚体的状态张量(13)
        _contact_force = self.gym.acquire_net_contact_force_tensor(self.sim) # gymTensor(num_env * num_bodies, 3)

        # self._contacts = self.gym.get_rigid_contacts(self.sim)   #  只用于CPU版本
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)  # Tensor(num_env, num_actor, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)  # Tensor(num_env, num_joint, 13)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)  # Tensor(num_env, num_link, 13)
        self._contact_force = gymtorch.wrap_tensor(_contact_force).view(self.num_envs, -1, 3) # Tensor(num_env, num_bodies, 3)

        self._q = self._dof_state[..., 0]  # Tensor(2, 22) 位置
        self._qd = self._dof_state[..., 1]  # Tensor(2, 22) 速度
        self._eef_state = self._rigid_body_state[:, self.handles["gripper_site"], :]
        self._eef_lf_state = self._rigid_body_state[:, self.handles["left_pad"], :]
        self._eef_rf_state = self._rigid_body_state[:, self.handles["right_pad"], :]
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "moma")
        jacobian = gymtorch.wrap_tensor(_jacobian)  # Tensor(num_env, num_bodies, 6, num_dofs)
        gripper_joint_index = self.gym.get_actor_joint_dict(env_ptr, moma_handle)['joint_7']  # 包括所有的joint
        ee_index = self.gym.find_actor_rigid_body_index(env_ptr, moma_handle, "arm_link6", gymapi.DOMAIN_ENV)
        self._j_eef = jacobian[:, ee_index, :, 6:12]
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "moma")
        mm = gymtorch.wrap_tensor(_massmatrix)  # Tensor(num_env, num_dof, num_dof)
        self._mm = mm[:, :6, :6]
        self._bottleA_state = self._root_state[:, self._bottleA_id, :]
        self._table_state = self._root_state[:, self._table_id, :]
        self._obstacle_state = self._root_state[:, self._obstacle_ids[0]:self._obstacle_ids[-1]+1, :]
        self._moma_robot_state = self._root_state[:, self.momas[0], :]
        # Initialize states
        self.states.update({
            "bottleA_height": None,
        })

        # Initialize actions
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)  # num_dofs是活动关节数量
        self._effort_control = torch.zeros_like(self._pos_control)
        # self._robot_state_control = self._moma_robot_state.clone()

        # Initialize control
        self._arm_control = self._effort_control[:, :6]
        self._gripper_control = self._pos_control[:, 6:12]
        self._robot_pos_control = self._moma_robot_state[:, :7]
        self._robot_vel_control = self._moma_robot_state[:, 7:]

        # Initialize indices
        self._global_indices = torch.arange(self.num_envs * (7 + 3), dtype=torch.int32, device=self.device).view(self.num_envs, -1)  # 1 for robot table bottleA

        # Initialize base vel
        # self._robot_state_vel_control = self._moma_robot_state[:, 7:]
        self._constant_base_vel = torch.tensor(self.cfg["constant_action"]["base_vel"])

        #### end effector pose
        self.pos_eef = self._rigid_body_state[:, self.handles['arm_link6'], 0:3]
        self.quat_eef = self._rigid_body_state[:, self.handles['arm_link6'], 3:7]

        self.pos_base = self._rigid_body_state[:, self.handles['arm_base_link'], 0:3]
        self.quat_base = self._rigid_body_state[:, self.handles['arm_base_link'], 3:7]

    def _update_states(self):
        if self.bottleA_height == None:
            self.bottleA_height = self._bottleA_state[0, 2]
            self.states.update({
                "bottleA_height": self.bottleA_height * torch.ones_like(self._eef_state[:, 0]),
            })

        # state_camera_depth_tensor_batch = copy.deepcopy(self.curr_camera_depth_tensor_batch)
        # state_camera_depth_tensor_batch = -torch.nan_to_num(state_camera_depth_tensor_batch, posinf=1e10, neginf=0)
        # state_camera_depth_tensor_batch = state_camera_depth_tensor_batch.unsqueeze(-1).permute(0, 3, 1, 2)
        # state_camera_depth_tensor_batch = F.interpolate(state_camera_depth_tensor_batch,
        #                                                 size=(self.cfg_visual_encoder['img_h'], self.cfg_visual_encoder['img_w']),
        #                                                 mode='bilinear',
        #                                                 align_corners=False)
        curr_rgb_img_tensor_batch = self.curr_camera_rgb_tensor_batch[..., :3].float()
        curr_rgb_img_tensor_batch = pre_process_rgb_img(curr_rgb_img_tensor_batch,
                                                        img_h=self.cfg_visual_encoder['img_h'],
                                                        img_w=self.cfg_visual_encoder['img_w'])
        ve_feature = self.depth_anything(curr_rgb_img_tensor_batch)
        # print('score:   {}'.format(self.grasp_pos_tensor[:, :, -1]))
        self.score_pool = to_torch(self.score_pool, device=self.device)
        self.states.update({
            # Robot
            "q": self._q[:, :],
            "q_gripper": self._q[:, 6:12],
            "eef_pos": self._eef_state[:, :3],
            "eef_quat": self._eef_state[:, 3:7],
            "eef_6d": matrix_to_rotation_6d(quaternion_to_matrix(torch.roll(self._eef_state[:, 3:7], 1))),
            "eef_vel": self._eef_state[:, 7:],
            "eef_lf_pos": self._eef_lf_state[:, :3],
            "eef_rf_pos": self._eef_rf_state[:, :3],
            "robot_pos": self._moma_robot_state[:, 0:7],
            "robot_vel": self._moma_robot_state[:, 7:],
            # "robot_vel": self.curr_robot_vel,
            # Cubes
            "bottleA_pos": self._bottleA_state[:, :3],
            "bottleA_quat": self._bottleA_state[:, 3:7],
            "bottleA_pos_relative": self._bottleA_state[:, :3] - self._eef_state[:, :3],
            "target_bottleA_pos": self._init_bottleA_state[:, :3] + torch.tensor([[0, 0, 0.2]], device=self.device),
            "target_bottleA_quat": self._init_bottleA_state[:, 3:7],
            "bottleA_pos_relative_robot_2d": self._bottleA_state[:, :2] - self._moma_robot_state[:, :2],
            # todo orientation
            # state:
            "flag_collision": self.flag_collision.view(-1, 1),
            # vision info
            'grasp_pos_vector': self.grasp_pos_tensor.reshape(self.num_envs, -1),
            'perfect_grasp_pos_vector': self.perfect_grasp_pos_tensor.reshape(self.num_envs, -1),
            'prev_actions': self.prev_actions,
            'filtered_actions': self.filtered_actions,
            # 'curr_depth_img_tensor_batch': state_camera_depth_tensor_batch.reshape(self.num_envs, -1),
            'curr_rgb_img_tensor_batch': curr_rgb_img_tensor_batch.reshape(self.num_envs, -1),
            've_feature': ve_feature.reshape(self.num_envs, -1),
            # 'ori_grasp_pos_vector': self.ori_grasp_pos_tensor.reshape(self.num_envs, -1),
            'r_go': self.score_pool[torch.arange(self.num_envs), self.progress_buf] / 10,
            'r_gg': self.d_gg_pool[torch.arange(self.num_envs), self.progress_buf],
            'r_o2g': self.r_o2g_pool[torch.arange(self.num_envs), self.progress_buf],
            'd_gripper_grasp_pos': self.d_gripper_grasp_pos[torch.arange(self.num_envs), self.progress_buf],
            "flag_gripper_grasp_pos": (self.d_gripper_grasp_pos[torch.arange(self.num_envs), self.progress_buf] < 0.1).view(-1, 1)
        })
        # print('self.states[bottleA_pos]:    {}'.format(self.states['bottleA_pos']))

    def get_arm_ee_pos(self):
        T_world_eef = pose_to_matrix(self.pos_eef[0], self.quat_eef[0])
        T_world_base = pose_to_matrix(self.pos_base[0], self.quat_base[0])
        T_base_eef = torch.linalg.inv(T_world_base) @ T_world_eef
        pos_rel = T_base_eef[:3, 3]
        rot_rel_mat = T_base_eef[:3, :3]
        rot_rel_euler = R.from_matrix(rot_rel_mat.numpy()).as_euler('xyz')
        return torch.cat([pos_rel, torch.from_numpy(rot_rel_euler)]).unsqueeze(0)

    def get_arm_ee_pos_batch(self, pos_eef, quat_eef, pos_base, quat_base):
        T_world_eef = pose_to_matrix(pos_eef, quat_eef)
        T_world_base = pose_to_matrix(pos_base, quat_base)

        T_base_eef = torch.linalg.inv(T_world_base) @ T_world_eef

        pos_rel = T_base_eef[:, :3, 3]
        rot_rel_mat = T_base_eef[:, :3, :3]
        rot_rel_euler = rotmat_to_euler_xyz_batch(rot_rel_mat)

        return torch.cat([pos_rel, rot_rel_euler], dim=-1)

    def compute_score_sum(self):
        curr_score = self.grasp_pos_tensor[:, :, -1].sum(1)  # Tensor(num_envs, 64)
        # self.score_pool.append(curr_score)
        self.score_pool = to_torch(self.score_pool, device=self.device)
        self.score_pool[torch.arange(self.num_envs), self.progress_buf] = curr_score

    def theta(self, q1, q2):
        q1 = q1 / q1.norm(dim=-1, keepdim=True)
        q2 = q2 / q2.norm(dim=-1, keepdim=True)

        dot = torch.sum(q1 * q2, dim=-1)
        dot = torch.clamp(dot, -1, 1)

        angle = 2 * torch.acos(torch.abs(dot))
        return angle

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


    def compute_d_gg(self):
        curr_grasp_pos_tensor = self.ori_grasp_pos_tensor  # Tensor(num_envs, 64, 8): [pos, quat, score]
        pos, quat = curr_grasp_pos_tensor[:, :, :3], curr_grasp_pos_tensor[:, :, 3:3+4]
        grasp_score = curr_grasp_pos_tensor[:, :, -1]
        p_gripper, q_gripper = self._eef_state[:, :3], self._eef_state[:, 3:7]
        pos_dist = (p_gripper.unsqueeze(1) - pos).norm(dim=-1)
        quat_dist_0 = self.theta(q_gripper.unsqueeze(1), quat)
        q_gripper_180 = get_quat_180(q_gripper)
        quat_dist_180 = self.theta(q_gripper_180.unsqueeze(1), quat)
        quat_dist = torch.min(quat_dist_0, quat_dist_180)
        pos_quat_dist = self.reward_settings["o2g_beta1"] * pos_dist + self.reward_settings["o2g_beta2"] * quat_dist # (num_envs, 64)
        min_pos_quat_dist, _ = torch.min(pos_quat_dist, dim=1)
        d_gg, _ = torch.min(torch.exp(-1 * grasp_score) * pos_quat_dist, dim=1)
        self.score_pool_exp = to_torch(self.score_pool_exp, device=self.device)
        # pdb.set_trace()
        # d_gg, _ = torch.max(
        #     (torch.exp(grasp_score / self.temperature) / self.score_pool_exp[torch.arange(self.num_envs), self.progress_buf].unsqueeze(1))
        #   * (1 - torch.tanh(pos_quat_dist)), dim=1)
        # d_gg = torch.min()
        self.d_gg_pool, self.d_gripper_grasp_pos = to_torch(self.d_gg_pool, device=self.device), to_torch(self.d_gripper_grasp_pos, device=self.device)
        self.d_gg_pool[torch.arange(self.num_envs), self.progress_buf] = d_gg.to(self.d_gg_pool.dtype)
        self.d_gripper_grasp_pos[torch.arange(self.num_envs), self.progress_buf] = min_pos_quat_dist.to(self.d_gripper_grasp_pos.dtype)

    def compute_o2g_reward(self):
        self.compute_d_gg()
        t = self.progress_buf.clone()
        t_1 = t - 1
        valid_mask = t >= 0
        self.r_o2g_pool = to_torch(self.r_o2g_pool, device=self.device)
        self.r_o2g_pool[torch.arange(self.num_envs), t] = 0.0
        if valid_mask.any():
            env_ids = to_torch(torch.arange(self.num_envs), dtype=torch.int32, device=self.device)[valid_mask]
            t_valid = t[valid_mask]
            t_1_valid = t_1[valid_mask]
            sigma = 1 / (1 + torch.exp(0.5 - (t_valid / self.max_episode_length)))
            self.states.update({'sigma': sigma})
            self.score_pool = to_torch(self.score_pool, device=self.device)
            # r_go = self.score_pool[env_ids, t_valid] / 10
            # r_gg = self.d_gg_pool[env_ids, t_valid]

            r_go = self.score_pool[env_ids, t_valid] - self.score_pool[env_ids, t_1_valid]
            r_gg = self.d_gg_pool[env_ids, t_1_valid] - self.d_gg_pool[env_ids, t_valid]
            # pdb.set_trace()
            r_o2g = (1 - sigma) * r_go + sigma * r_gg
            self.r_o2g_pool[env_ids, t_valid] = r_o2g

        # if self.frame_idx == 0:
        #     self.r_o2g_pool.append(0)
        # else:
        #     t_1 = self.frame_idx - 1
        #     t = self.frame_idx
        #     sigma = 1 / (1 + math.exp(0.5 - (self.frame_idx / self.max_episode_length)))
        #     r_go = self.score_pool[t] - self.score_pool[t_1]
        #     r_gg = self.d_gg_pool[t_1] - self.d_gg_pool[t]
        #     r_o2g = (1 - sigma) * r_go + sigma * r_gg
        #     self.r_o2g_pool.append(r_o2g)

    def compute_good_grasp_pos_batch(self, radius=0.1, num_grasps=20):
        bottle_pos = self._bottleA_state[:, :3]  # (N, 3)
        bottle_quat = self._bottleA_state[:, 3:7]  # (N, 4)

        # Convert torch quaternion to z_axis (N, 3)
        z_axis = self.quat_apply_batch(bottle_quat, np.array([0, 0, -1]))  # (N, 3)

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
            bottle_pos.cpu().numpy(), x_axis, y_axis, z_axis, radius=radius, num_grasps=num_grasps
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

    def quat_apply_batch(self, quat_batch, vec):
        """
        Applies quaternion to vector in batch form.
        quat_batch: (N, 4) torch or numpy
        vec: (3,) np array
        returns: (N, 3)
        """
        if isinstance(quat_batch, torch.Tensor):
            quat_batch = quat_batch.cpu().numpy()

        r = R.from_quat(quat_batch)
        return r.apply(vec)


    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.flag_collision = self.collision_detection()
        self.curr_camera_rgb_tensor_batch, self.curr_camera_depth_tensor_batch, self.curr_frame_pcd, \
            self.curr_frame_valid, self.camera_view_camera_matric_inv_batch, self.delta_pos_env \
            = self.acquire_camera_image(self.frame_idx)

        if self.frame_idx % 20 == 0:
            if self.ori_grasp_pos_tensor is None:
                last_ori_grasp_pos_list = []
                for i in range(self.num_envs):
                    _, ori_grasp_pos_vector = self.generate_random_transform_batch(self._bottleA_state[i, :3])
                    last_ori_grasp_pos_list.append(ori_grasp_pos_vector)
                self.last_ori_grasp_pos_tensor = torch.stack(last_ori_grasp_pos_list)
            else:
                self.last_ori_grasp_pos_tensor = self.ori_grasp_pos_tensor.clone()
            self.grasp_pos_tensor, self.ori_grasp_pos_tensor = self.grasp_pos_extraction()
            # print('-----------------------------------------')
            # print('!!!!!!!!!! global_idx:    {}'.format(self.global_idx))
            # print('-----------------------------------------')
        else:
            self.score_pool, self.score_pool_exp = to_torch(self.score_pool, device=self.device), to_torch(self.score_pool_exp, device=self.device)
            self.score_pool[:, self.progress_buf] = self.score_pool[:, self.progress_buf-1]
            self.score_pool_exp[:, self.progress_buf] = self.score_pool_exp[:, self.progress_buf-1]


        self.grasp_poses_factory = self.compute_good_grasp_pos_batch()
        self.perfect_grasp_pos_tensor = self.convert_grasp_pose_to_features(to_torch(self.grasp_poses_factory, device=self.device))

        # pdb.set_trace()
        self.compute_o2g_reward()
        # Refresh states
        self._update_states()


    def collision_detection(self, collision_threshold = 0.1):
        target_indices = [self.handles[link_name] for link_name in self._contact_bodies]
        target_forces = self._contact_force[:, target_indices, :]
        # print('self._contact_force:    {}'.format(self._contact_force))
        force_norms = torch.norm(target_forces, dim=2)
        collision_flags = (force_norms > collision_threshold).any(dim=1)
        return collision_flags

    def collision_detecton_with_table(self, collision_threshold = 1):
        table_id = [self.handles['table_handle']] # 这里确定了编号，在程序上可以优化
        table_contact = self._contact_force[:, table_id, :2]
        table_contact_force = (torch.norm(table_contact, dim=2) > collision_threshold).any(dim=1)
        return table_contact_force

    def collision_detection_with_bottle(self, collision_threshold=0.1):
        bottle_id = [self._bottleA_id]
        bottle_contact = self._contact_force[:, bottle_id, :]
        bottle_contact_force = (torch.norm(bottle_contact, dim=2) > collision_threshold).any(dim=1)
        return bottle_contact_force

    def collision_detecton_with_wall(self, collision_threshold = 0.1):
        wall_ids = [self.handles[wall_name] for wall_name in self._contact_wall] # 这里确定了编号，在程序上可以优化
        wall_contact = self._contact_force[:, wall_ids, :]
        wall_contact_force = (torch.norm(wall_contact, dim=2) > collision_threshold).any(dim=1)
        return wall_contact_force

    def collision_detection_with_obs(self, collision_threshold=0.1):
        obs_ids = [self.handles[obs_name] for obs_name in self._contact_obs]
        obs_contact = self._contact_force[:, obs_ids, :]
        obs_contact_force = (torch.norm(obs_contact, dim=2) > collision_threshold).any(dim=1)
        return obs_contact_force

    #@brief: 根据点云提取grasp pos，并根据物体是否可视对grasp pos进行调整
    def grasp_pos_extraction(self, valid_grasp_pos_num=64):
        grasp_pos_list = []  # tensor(pos, F(quat), score) -> 10
        ori_grasp_pos_list = [] # tensor(pos, quat, score) -> 8
        for i in range(self.num_envs):
            #@ 1. 将当前帧的点云加入到该环境下的grasp pose factories.
            frame_cloud = self.grasp_pose_factories[i].add_new_frame(self.curr_frame_pcd[i], self.curr_frame_valid[i], i,
                                                                     cam_trans=self.camera_view_camera_matric_inv_batch[i],
                                                                     delta_env_pos=self.delta_pos_env[i],
                                                                     target_label=self.segmentation_id['bottleA'])

            #@ 2. 查看是否有fusion后的grasping pose
            all_grasp_group_array = []
            list_scene_node = list(self.grasp_pose_factories[i].scene_node)
            if len(list_scene_node) == 0:
                #@ 2.1. 如果没有fusion的grasping pose，就生成一些随机的grasping pose
                grasp_pos_vector, ori_grasp_pos_vector = self.generate_random_transform_batch(self._bottleA_state[i, :3])
            else:
                #@ 2.2. 如果有fusion的grasping pose，首先看看有没有合格的grasping pose.
                for node in list_scene_node:
                    tmp_grasp_group_array = node.grasp_group_array
                    if node.score > 0.3:
                        all_grasp_group_array.append(tmp_grasp_group_array)
                if len(all_grasp_group_array) == 0:
                    grasp_pos_vector, ori_grasp_pos_vector = self.generate_random_transform_batch(self._bottleA_state[i, :3])
                    grasp_pos_list.append(grasp_pos_vector)
                    ori_grasp_pos_list.append(ori_grasp_pos_vector)
                    continue
                #@ 2.2.1 如果有的话，生成valid_grasp_pos_num个grasping pose.
                indices = torch.randint(0, len(all_grasp_group_array), (valid_grasp_pos_num, ))
                sampled = torch.from_numpy(np.array(all_grasp_group_array))[indices]
                grasp_pos_vector = torch.cat([sampled[:, 13:16], sampled[:, 4:10], sampled[:, 0].unsqueeze(1)],
                                             dim=1).to(self.device)
                rot_matrices = sampled[:, 4:4+9].reshape(-1, 3, 3)
                quats = R.from_matrix(rot_matrices).as_quat()
                ori_grasp_pos_vector = torch.cat(
                    [sampled[:, 13:16], torch.from_numpy(quats).to(sampled.device), sampled[:, 0].unsqueeze(1)],
                    dim=1).to(self.device)

            grasp_pos_list.append(grasp_pos_vector)
            ori_grasp_pos_list.append(ori_grasp_pos_vector)
            curr_score = sum([sce.score for sce in list_scene_node])
            self.score_pool[i, self.progress_buf[i]] = curr_score
            self.score_pool_exp[i, self.progress_buf[i]] = torch.exp(to_torch(curr_score, device=self.device) / self.temperature)

        grasp_pos_tensor = torch.stack(grasp_pos_list)
        ori_grasp_pos_tensor = torch.stack(ori_grasp_pos_list)
        return grasp_pos_tensor, ori_grasp_pos_tensor

    def compute_reward(self, actions):
        wall_contact_force = self.collision_detecton_with_wall()
        table_contact_force = self.collision_detecton_with_table()
        bottle_contact_force = self.collision_detection_with_bottle()
        obs_contact_force = self.collision_detection_with_obs()
        self.flag_collision |= wall_contact_force
        self.flag_collision |= table_contact_force
        # self.flag_collision |= bottle_contact_force
        self.flag_collision |= obs_contact_force

        self.rew_buf[:], self.reset_buf[:], self.reward_dict = compute_moma_reward(
            self.reset_buf, self.progress_buf, self.actions, self.states, self.reward_settings,
            self.max_episode_length, self.flag_collision, wall_contact_force, bottle_contact_force
        )


    def compute_observations(self):
        self._refresh()
        # obs = ["bottleA_quat", "bottleA_pos", "eef_pos", "eef_quat", "robot_vel",
        #        "bottleA_pos_relative_robot_2d", "flag_collision"]  # 4 + 3 + 3 + 4 + 6 + 2 + 1
        obs = ["eef_pos", "eef_6d", "robot_vel"]
               # "flag_collision"]  # 4 + 3 + 3 + 4 + 6 + 2 + 1
        # obs += ["q_gripper"] if self.control_type != "joint_tor" else ["q"]
        obs += ['grasp_pos_vector']
        # obs += ['curr_depth_img_tensor_batch']
        obs += ['ve_feature']
        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)
        maxs = {ob: torch.max(self.states[ob]).item() for ob in obs}
        return self.obs_buf

    # @ brief: bottleA的位置固定
    def _reset_init_bottle_state(self, name, env_ids, check_valid=True):
        """
        Init the position of cube(env)
        """
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)

        # Initialize buffer to hold sampled values
        num_resets = len(env_ids)
        sampled_cube_state = torch.zeros(num_resets, 13, device=self.device)  # [x y z qx qy qz qw vx vy vz wx wy wz]

        if name.lower() == 'a':
            this_cube_state_all = self._init_bottleA_state
            # other_cube_state = self._init_cubeB_state[env_ids, :]
            cube_heights = self.states["bottleA_height"]
        table_info = self._init_table_state[env_ids, :]
        sampled_cube_state[:, :2] = table_info[:, :2]
        # sampled_cube_state[:, 2] = table_info[:, 2] + self._table_surface_pos[2] / 2 + height_list[env_ids % self.ln_obj_asset] / 2 # 0.14
        if self.cfg['env']['asset']['assetFlagBottleOnly'] is True:
            sampled_cube_state[:, 2] = table_info[:, 2] + self._table_surface_pos[2] / 2 + 0.14 # 0.14
        else:
            height_list = to_torch(self.height_list, device=self.device)
            sampled_cube_state[:, 2] = table_info[:, 2] + self._table_surface_pos[2] / 2 + height_list[env_ids % self.ln_obj_asset] / 2 # 0.14
        sampled_cube_state[:, 6] = 1.0
        # sampled_cube_state[:, :2] = torch.tensor([-0.2, 0.0]).repeat(num_resets, 1)

        this_cube_state_all[env_ids, :] = sampled_cube_state

    def _reset_init_table_state(self, env_ids, check_valid=True):
        """
        Init the sate of table in the envs
        """
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)

        # Initialize buffer to hold sampled values
        num_resets = len(env_ids)
        this_table_state_all = self._init_table_state
        sampled_table_state = torch.zeros(num_resets, 13, device=self.device)  # [x y z qx qy qz qw vx vy vz wx wy wz]'
        # height randomization
        sampled_table_state[:, 2] = self._table_surface_pos[2] / 2 +\
                                   self.table_height_noise * 2.0 * (torch.rand(num_resets, device=self.device) - 0.5)
        # position randomization
        sampled_table_state[:, :2] = to_torch(self._table_surface_pos[:2], device=self.device).repeat(num_resets, 1) +\
                                    self.table_position_noise * 2.0 * (torch.rand(num_resets, 2, device=self.device) - 0.5)

        sampled_table_state[:, 6] = 1.0
        if self.table_rotation_noise > 0:
            aa_rot = torch.zeros(num_resets, 3, device=self.device)
            aa_rot[:, 2] = 2.0 * self.start_rotation_noise * (torch.rand(num_resets, device=self.device) - 0.5)
            sampled_table_state[:, 3:7] = quat_mul(axisangle2quat(aa_rot), sampled_table_state[:, 3:7])
        this_table_state_all[env_ids, :] = sampled_table_state

    def _reset_init_robot_state(self, env_ids):
        # tmp_defalut_state_pos = self.moma_default_state_pos.repeat(len(env_ids), 1)
        # Create moma
        # Potentially randomize start pose
        init_state_pos = torch.zeros_like(self.moma_default_state_pos).repeat(len(env_ids), 1)
        if self.moma_position_noise >= 0:
            rand_xy = self.moma_position_noise * (-1. + np.random.rand(len(env_ids), 2) * 2.0)
            init_state_pos[:, :2] = to_torch(rand_xy, device=self.device) + self.moma_default_state_pos[:2]
            # moma_start_pose.p = gymapi.Vec3(-0.7 + rand_xy[0], -0.7 + rand_xy[1], 0.0)

        if self.moma_rotation_noise > 0:
            rand_angle = self.moma_rotation_noise * (-1.0 + 2.0 * np.random.rand(len(env_ids)))

            rand_rot = np.zeros((len(env_ids), 3))
            rand_rot[:, 2] = rand_angle
            q_noise = axisangle2quat(to_torch(rand_rot, device=self.device))  # (N, 4) xyzw

            q_base = self.moma_default_state_pos[3:7].unsqueeze(0).repeat(len(env_ids), 1)  # (N, 4)

            q_new = quat_mul(q_noise, q_base)  # 局部 z 轴旋转
            init_state_pos[:, 3:7] = q_new
        else:
            init_state_pos[:, 3:7] = self.moma_default_state_pos[3:7]
            # moma_start_pose.r = gymapi.Quat(*new_quat)

        return init_state_pos

    def _reset_init_obstacle_state(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)
        num_resets = len(env_ids)
        this_obstacle_state_all = self._init_obstacle_state
        sampled_obstacle_state = torch.zeros(num_resets, 3, 13, device=self.device)
        regions = [
            {"x": tuple(r["x"]), "y": tuple(r["y"])}
            for r in self.cfg['env']["obstacle_regions"]
        ]
        xs, ys = sample_from_regions_np(regions, num_resets * 3)
        xs, ys = xs.reshape(num_resets, 3), ys.reshape(num_resets, 3)
        sampled_obstacle_state[:, :, 0] = to_torch(xs, device=self.device)
        sampled_obstacle_state[:, :, 1] = to_torch(ys, device=self.device)
        sampled_obstacle_state[:, :, 2] = 0.5
        sampled_obstacle_state[:, :, 6] = 1.0

        this_obstacle_state_all[env_ids, :] = sampled_obstacle_state

    # transforms中保存的是pos + rot6d + score
    # ori_transforms中保存的是pos + quat + score
    #@brief: 在瓶子周边生成随机的grasp pos
    def generate_random_transform_batch(self, pos, num_samples=64):
        rot_matrices = R.random(num_samples).as_matrix()
        rot_matrices = torch.from_numpy(rot_matrices).float()

        rot_cols = rot_matrices[:, :, :2].reshape(num_samples, 6).to(pos)
        pos_tensor = pos.repeat(num_samples, 1)
        transforms = torch.cat([pos_tensor, rot_cols, 0.1 * torch.ones(num_samples, 1).to(pos)], dim=1)
        quats = R.from_matrix(rot_matrices.cpu().numpy()).as_quat()
        ori_transforms = torch.cat([pos_tensor, torch.from_numpy(quats).to(rot_cols.device), 0.1 * torch.ones(num_samples, 1).to(pos)], dim=1)
        # ori_transforms = torch.cat([pos_tensor, ])
        return transforms, ori_transforms

    def _compute_osc_torques(self, dpose):
        # Solve for Operational Space Control # Paper: khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf
        # Helpful resource: studywolf.wordpress.com/2013/09/17/robot-control-4-operation-space-control/
        q, qd = self._q[:, :6], self._qd[:, :6]
        mm_inv = torch.inverse(self._mm)
        m_eef_inv = self._j_eef @ mm_inv @ torch.transpose(self._j_eef, 1, 2)
        m_eef = torch.inverse(m_eef_inv)

        # Transform our cartesian action `dpose` into joint torques `u`
        u = torch.transpose(self._j_eef, 1, 2) @ m_eef @ (
                self.kp * dpose - self.kd * self.states["eef_vel"]).unsqueeze(-1)

        # Nullspace control torques `u_null` prevents large changes in joint configuration
        # They are added into the nullspace of OSC so that the end effector orientation remains constant
        # roboticsproceedings.org/rss07/p31.pdf
        j_eef_inv = m_eef @ self._j_eef @ mm_inv
        u_null = self.kd_null * -qd + self.kp_null * (
                (self.moma_default_dof_pos[:6] - q + np.pi) % (2 * np.pi) - np.pi)
        u_null[:, 6:] *= 0
        u_null = self._mm @ u_null.unsqueeze(-1)
        u += (torch.eye(6, device=self.device).unsqueeze(0) - torch.transpose(self._j_eef, 1, 2) @ j_eef_inv) @ u_null

        # Clip the values to be within valid effort range
        u = tensor_clamp(u.squeeze(-1),
                         -self._moma_effort_limits[:6].unsqueeze(0), self._moma_effort_limits[:6].unsqueeze(0))

        return u

    def _compute_ik(self, dpose, damping=0.05):
        j_eef, num_envs = self._j_eef, self.num_envs
        # solve damped least squares
        j_eef_T = torch.transpose(j_eef, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (damping ** 2)
        u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose.unsqueeze(-1)).view(num_envs, 6)
        return u

    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        for env_id in env_ids:
            self.grasp_pose_factories[env_id].reset_gltree()

        self.bottleA_height = None
        # if not self._i:
        self._reset_init_table_state(env_ids=env_ids, check_valid=True)
        self._reset_init_bottle_state(name='A', env_ids=env_ids, check_valid=True)
        self._reset_init_obstacle_state(env_ids=env_ids)
        # self._i = True

        # Write these new init states to the sim states
        self._bottleA_state[env_ids] = self._init_bottleA_state[env_ids]
        self._table_state[env_ids] = self._init_table_state[env_ids]
        self._obstacle_state[env_ids] = self._init_obstacle_state[env_ids]

        # Reset agent
        reset_noise = torch.rand((len(env_ids), 12), device=self.device)
        pos = tensor_clamp(
            self.moma_default_dof_pos.unsqueeze(0) +
            self.moma_dof_noise * 2.0 * (reset_noise - 0.5),
            self.moma_dof_lower_limits.unsqueeze(0), self.moma_dof_upper_limits)
        # pos = self.moma_default_dof_pos.unsqueeze(0)

        # Overwrite gripper init pos (no noise since these are always position controlled)
        # pos[:, -:] = self.moma_default_dof_pos[-2:]
        # Reset the internal obs accordingly
        # self._q[env_ids, :] = pos
        self._q[env_ids, :12] = pos
        self._qd[env_ids, :] = torch.zeros_like(self._qd[env_ids])

        # Set any position control to the current position, and any vel / effort control to be 0
        # NOTE: Task takes care of actually propagating these controls in sim using the SimActions API
        # self._pos_control[env_ids, :] = pos
        self._pos_control[env_ids, :12] = pos
        self._effort_control[env_ids, :12] = torch.zeros_like(pos)

        self._robot_vel_control[env_ids, :] = torch.zeros_like(self._robot_vel_control[env_ids, :])
        # tmp_defalut_state_pos = self.moma_default_state_pos.repeat(len(env_ids), 1)
        tmp_defalut_state_pos = self._reset_init_robot_state(env_ids)
        # tmp_defalut_state_pos[:, 0] = torch.rand(len(env_ids), device=self.moma_default_state_pos.device) * 1.4 - 0.7
        # tmp_defalut_state_pos[:, 3:] = random_z_rotation_quaternion(len(env_ids))
        self._robot_pos_control[env_ids, :] = tmp_defalut_state_pos

        # Deploy updates
        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._pos_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._effort_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))
        # self.gym.set_actor_root_state_tensor_indexed(self.sim,
        #                                              gymtorch.unwrap_tensor(self._root_state),
        #                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
        #                                              len(multi_env_ids_int32))

        # Update cube states
        multi_env_ids_table_int32 = self._global_indices[env_ids, 1].flatten()
        multi_env_ids_cubes_int32 = self._global_indices[env_ids, -1].flatten()
        multi_env_ids_obstacle_int32 = self._global_indices[env_ids, -4:-1].flatten()
        combined_root_state = torch.cat((multi_env_ids_int32, multi_env_ids_table_int32, multi_env_ids_obstacle_int32, multi_env_ids_cubes_int32), dim=0)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(combined_root_state), len(combined_root_state)
        )

        # self.score_pool, self.r_gg_pool, self.d_gg_pool = [], [], []
        self.score_pool[env_ids] = 0
        self.score_pool_exp[env_ids] = 0
        self.d_gg_pool[env_ids] = 0
        self.r_o2g_pool[env_ids] = 0
        self.d_gripper_grasp_pos[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._init_states = None
        self.flag_collision = False
        self.frame_idx = 0

        # for env_id in env_ids:
        #     self.grasp_pose_factories[env_id].reset_gltree()
        #     print('len(list_scene_node[env_id]):     {}'.format(len(self.grasp_pose_factories[env_id].scene_node)))
        # with open('./test_vis/test_action.txt', 'a') as file:
        #     file.write('\n')
        # with open('./test_vis/test_robot_pos.txt', 'a') as file:
        #     file.write('\n')
        # with open('./test_vis/test_ee_pos.txt', 'a') as file:
        #     file.write('\n')
        # self._flag_self_collision = False
        # self._flag_arm_obs_collision = False

    def move_gripper(self, u_gripper):
        u_fingers = torch.zeros_like(self._gripper_control)
        tmp_u_finger = torch.where(u_gripper >= -1, 1.0, 0.0)
        u_fingers = torch.concatenate([tmp_u_finger.unsqueeze(1),
                                       -1 * tmp_u_finger.unsqueeze(1),
                                       tmp_u_finger.unsqueeze(1),
                                       tmp_u_finger.unsqueeze(1),
                                       tmp_u_finger.unsqueeze(1),
                                       -1 * tmp_u_finger.unsqueeze(1)], dim=1)
        return u_fingers

    # @brief: Base position control
    def move_base_position(self, u_wheel):
        ori_pos = self._moma_robot_state[:, :3].clone()
        ori_quats = self._moma_robot_state[:, 3:7].clone()

        ans_base_position = torch.zeros(self._moma_robot_state.shape[0], 7)  # (vx, vy, vz, wx, wy, wz)
        final_u_wheel = u_wheel * self.cmd_base_limit
        vx, wz = final_u_wheel[:, 0], final_u_wheel[:, 1]
        robot_quat = self._moma_robot_state[:, 3:7]
        vx_local = torch.stack([vx, torch.zeros_like(u_wheel[:, 0]), torch.zeros_like(u_wheel[:, 0])], dim=-1)
        vx_world = quat_rotate(robot_quat, vx_local)
        vx_position = self.sim_params.dt * vx_world
        target_pos = ori_pos + vx_position
        target_pos[:, 2] = 0
        rotvecs = np.zeros((self._moma_robot_state.shape[0], 3))
        rotvecs[:, 2] = wz.cpu().numpy() * self.sim_params.dt
        noise_rots = R.from_rotvec(rotvecs)
        base_rots = R.from_quat(ori_quats.cpu().numpy())

        new_rots = noise_rots * base_rots
        new_quats = new_rots.as_quat()  # (N,4)
        ans_base_position[:, :3] = target_pos
        ans_base_position[:, 3:7] = to_torch(new_quats, device=self.device)
        w_world = to_torch(np.zeros((wz.shape[0], 3)), device=self.device)
        w_world[:, 2] = wz
        self.curr_robot_vel = torch.cat([vx_world, w_world], dim=-1)
        return ans_base_position


    # @brief: Base velocity control
    def move_base_velocity(self, u_wheel):
        ans_base_velocity = torch.zeros(u_wheel.shape[0], 6)
        final_u_wheel = u_wheel * self.cmd_base_limit
        vx, wz = final_u_wheel[:, 0], final_u_wheel[:, 1]
        robot_quat = self._moma_robot_state[:, 3:7]
        vx_local = torch.stack([vx, torch.zeros_like(u_wheel[:, 0]), torch.zeros_like(u_wheel[:, 0])], dim=-1)
        vx_world = quat_rotate(robot_quat, vx_local)
        ans_base_velocity[:, :3] = vx_world
        ans_base_velocity[:, -1] = wz
        return ans_base_velocity

    # @Notice 生成的点云是在相机坐标系下，以opencv格式的点云
    def _depth_image_to_point_cloud_GPU_batch_opencv(self,
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
        position_batch[..., 0:3] = opengl_trans_opencv(position_batch[..., 0:3])
        points_batch = position_batch[..., [0, 1, 2, 4, 5, 6, 7]]
        valid_batch = valid_depth_batch  # * valid_z_p_batch * valid_z_n_batch

        return points_batch, valid_batch

    # @Notice 生成的点云是在世界坐标系下的
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

    #@ Attention:
    def save_camera_image(self, frame_idx):
        for i in range(self.num_envs):
            rgb_tensor = self.camera_rgb_tensor_list[i]
            depth_tensor = self.camera_depth_tensor_list[i]
            seg_tensor = self.camera_seg_tensor_list[i]
            os.makedirs(f'./test_vis/vision/{i}', exist_ok=True)
            if rgb_tensor is not None:
                color_np = rgb_tensor.cpu().numpy()
                # print('color_np.shape:  {}'.format(color_np.shape))
                color_np = np.reshape(color_np, (self.camera_asset.height, self.camera_asset.width, 4))[:, :, :3]
                cv2.imwrite(f"./test_vis/vision/{i}/rgb_{self.frame_idx:03d}.png", cv2.cvtColor(color_np, cv2.COLOR_RGB2BGR))

            if depth_tensor is not None:
                depth_np = depth_tensor.cpu().numpy()
                depth_np = np.reshape(depth_np, (self.camera_asset.height, self.camera_asset.width))
                cv2.imwrite(f"./test_vis/vision/{i}/depth_{self.frame_idx:03d}.png", (depth_np * 255).astype(np.uint8))

            if seg_tensor is not None:
                seg_np = seg_tensor.cpu().numpy()
                # unique_ids = np.unique(seg_np)
                # print('unique_ids:   {}'.format(unique_ids))
                seg_np = np.reshape(seg_np, (self.camera_asset.height, self.camera_asset.width))
                cv2.imwrite(f"./test_vis/vision/{i}/seg_{self.frame_idx:03d}.png", seg_np.astype(np.uint8) * 80)

    def acquire_camera_image(self, frame_idx):
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        vinv_matrices, proj_matrices = [], []
        delta_pos_env = []
        for i in range(self.num_envs):
            env_ptr, camera_actor = self.envs[i], self.camera_actors[i]
            vinv_mat = torch.inverse((to_torch(self.gym.get_camera_view_matrix(self.sim, env_ptr, camera_actor), device=self.device)))
            proj_mat = to_torch(self.gym.get_camera_proj_matrix(self.sim, env_ptr, camera_actor), device=self.device)
            # vinv_mat = to_torch(np.eye(4), device=self.device)
            vinv_matrices.append(vinv_mat)
            proj_matrices.append(proj_mat)
            ## 每个环境中心在世界坐标系下的偏置
            pos = self.gym.get_env_origin(env_ptr)
            delta_pos_env.append(to_torch(np.array([pos.x, pos.y, pos.z]), device=self.device))

        delta_pos_env = torch.stack(delta_pos_env)
        camera_rgb_tensor_batch = torch.stack(self.camera_rgb_tensor_list)
        camera_depth_tensor_batch = torch.stack(self.camera_depth_tensor_list)
        camera_seg_tensor_batch = torch.stack(self.camera_seg_tensor_list)
        camera_view_camera_matric_inv_batch = torch.stack(vinv_matrices)
        camera_proj_matrix_batch = torch.stack(proj_matrices)
        height, width = camera_depth_tensor_batch.shape[1], camera_depth_tensor_batch.shape[2]
        # camera_rgb_tensor_batch = torch.ones((self.num_envs, height, width, 4)) * 155

        # 这里的frame_valid只是depth的valid值
        frame_pcd, frame_valid = self._depth_image_to_point_cloud_GPU_batch_opencv(
        # frame_pcd, frame_valid = self._depth_image_to_point_cloud_GPU_batch(
            camera_depth_tensor_batch, camera_rgb_tensor_batch, camera_seg_tensor_batch,
            camera_view_camera_matric_inv_batch, camera_proj_matrix_batch,
            self.camera_u2, self.camera_v2, self.camera_asset.width, self.camera_asset.height,
            1.2, self.device
        )
        # pdb.set_trace()
        # camera_view_camera_matric_inv_batch[:, 3, :3] = camera_view_camera_matric_inv_batch[:, 3, :3] - delta_pos_env
        # pdb.set_trace()
        # frame_pcd[:, :, :3] = frame_pcd[:, :, :3] - delta_pos_env[:, None, :]

        # self.save_camera_image(frame_idx)
        self.gym.end_access_image_tensors(self.sim)
        return camera_rgb_tensor_batch, camera_depth_tensor_batch, frame_pcd, frame_valid, camera_view_camera_matric_inv_batch, delta_pos_env

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        u_wheel, u_arm, u_gripper = self.actions[:, :2], self.actions[:, 2:-1], self.actions[:, -1]
        u_arm = u_arm * self.cmd_arm_limit / self.action_scale

        if self.control_type == "osc" or self.control_type == "joint_tor":
            if self.control_type == "osc":
                u_arm = self._compute_osc_torques(dpose=u_arm)
            self._arm_control[:, :] = u_arm
        elif self.control_type == "ik":
            arm_base_quats = self._rigid_body_state[:, 1, 3:7].cpu().detach().numpy()
            arm_base_matrices = R.from_quat(arm_base_quats).as_matrix()
            ##### rotation vector  [0]
            u_arm_pos = u_arm[:, :3].cpu().detach().numpy()
            u_arm_rpy = u_arm[:, 3:].cpu().detach().numpy()
            u_arm_rot_vec = R.from_euler('xyz', u_arm_rpy).as_rotvec()
            delta_pos_world = np.einsum('nij,nj->ni', arm_base_matrices, u_arm_pos)  # (num_envs, 3)
            delta_rot_world = np.einsum('nij,nj->ni', arm_base_matrices, u_arm_rot_vec)  # (num_envs, 3)
            delta_rot_world = R.from_rotvec(delta_rot_world).as_quat()
            delta_rot_world = delta_rot_world[:, :3] * np.sign(delta_rot_world[:, 3])[:, None]
            u_arm = to_torch(np.hstack([delta_pos_world, delta_rot_world]), device=self.device)

            #### 这里求解ik的方式，其dpose是在世界坐标系下的
            u_arm = self._compute_ik(dpose=u_arm)
            goal_pos = self._q[:, :6] + u_arm
            self._pos_control[:, :6] = goal_pos

        self._gripper_control[:, :] = self.move_gripper(u_gripper)
        final_base_velocity = low_pass_filter(u_wheel, self.filtered_actions, self.progress_buf, alpha=self.lpf_alpha)
        self.filtered_actions = final_base_velocity.clone()
        current_base_vel_ = self.move_base_velocity(final_base_velocity)
        self._robot_vel_control[:, :] = current_base_vel_
        # Deploy actions
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))

        env_ids = torch.arange(self.num_envs, device=self.device)
        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_state),
                                                     gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                     len(multi_env_ids_int32))
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self._force_tensor[self.base_body_indices, 2] = -9.81 * 16.83
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self._force_tensor),
            gymtorch.unwrap_tensor(self._torque_tensor),
            gymapi.ENV_SPACE
        )

    #
    def post_physics_step(self):
        self.progress_buf += 1
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        self.frame_idx += 1
        self.global_idx += 1
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)
        self.prev_actions = self.actions.clone()

        # if len(env_ids) > 0:
        #     for env_id in env_ids:
        #         self.grasp_pose_factories[env_id].reset_gltree()

        self.write_stats(self.reward_dict)

        # debug viz
        self.debug_vis_draw_grasp_pos()

    def write_stats(self, reward_dict):
        global_idx = self.global_idx
        for name in self.reward_dict:
            accumulate_reward(name, reward_dict[name], self.progress_buf, self.reward_dict_episode)
        # print('self.frame_idx:    {}'.format(self.frame_idx))
        if global_idx % self.max_episode_length == 0:
            # print('self.global_idx:     {}'.format(self.global_idx))
            debug_vis_draw_scalar(self.reward_writer, self.reward_dict_episode, self.global_idx)


    def debug_vis_draw_grasp_pos(self):
        if self.viewer and self.debug_vis_grasp_pos:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            # 绘制每个env中机器人的ee pos和生成的grasp pos vector
            for i in range(self.num_envs):
                list_scene_node = list(self.grasp_pose_factories[i].scene_node)
                if len(list_scene_node) == 0:
                    continue
                all_grasp_group_array = []
                for node in list_scene_node:
                    tmp_grasp_group_array = node.grasp_group_array
                    all_grasp_group_array.append(tmp_grasp_group_array)
                all_grasp_group_array = np.array(all_grasp_group_array)
                for grasp_pos in list_scene_node:
                    pos = grasp_pos.grasp_pos
                    ori = grasp_pos.grasp_ori
                    quat = R.from_matrix(ori).as_quat()
                    dx = (to_torch(pos, device=self.device) + quat_apply(to_torch(quat, device=self.device), to_torch([0.05, 0, 0], device=self.device))).cpu().numpy()
                    dy = (to_torch(pos, device=self.device) + quat_apply(to_torch(quat, device=self.device), to_torch([0, 0.05, 0], device=self.device))).cpu().numpy()
                    dz = (to_torch(pos, device=self.device) + quat_apply(to_torch(quat, device=self.device), to_torch([0, 0, 0.05], device=self.device))).cpu().numpy()

                    self.gym.add_lines(self.viewer, self.envs[i], 1, [pos[0], pos[1], pos[2], dx[0], dx[1], dx[2]], [0.85, 0.1, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [pos[0], pos[1], pos[2], dy[0], dy[1], dy[2]], [0.1, 0.85, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [pos[0], pos[1], pos[2], dz[0], dz[1], dz[2]], [0.1, 0.1, 0.85])

            eef_pos = self.states["eef_pos"]
            eef_rot = self.states["eef_quat"]
            bottleA_pos = self.states["bottleA_pos"]
            bottleA_rot = self.states["bottleA_quat"]

            # Plot visualizations
            # 绘制每个env中瓶子的中心位置和ee pos的坐标系
            for i in range(self.num_envs):
                for pos, rot in zip((eef_pos, bottleA_pos), (eef_rot, bottleA_rot)):
                    px = (pos[i] + quat_apply(rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                    py = (pos[i] + quat_apply(rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                    pz = (pos[i] + quat_apply(rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                    p0 = pos[i].cpu().numpy()
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]],
                                       [0.85, 0.1, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]],
                                       [0.1, 0.85, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]],
                                       [0.1, 0.1, 0.85])

    def debug_viz(self):
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            # Grab relevant states to visualize
            eef_pos = self.states["eef_pos"]
            eef_rot = self.states["eef_quat"]
            bottleA_pos = self.states["bottleA_pos"]
            bottleA_rot = self.states["bottleA_quat"]

            # Plot visualizations
            for i in range(self.num_envs):
                for pos, rot in zip((eef_pos, bottleA_pos), (eef_rot, bottleA_rot)):
                    px = (pos[i] + quat_apply(rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                    py = (pos[i] + quat_apply(rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                    pz = (pos[i] + quat_apply(rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                    p0 = pos[i].cpu().numpy()
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]],
                                       [0.85, 0.1, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]],
                                       [0.1, 0.85, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]],
                                       [0.1, 0.1, 0.85])

#####################################################################
###=========================jit functions=========================###
#####################################################################
# @torch.jit.script
def compute_moma_reward(
        reset_buf, progress_buf, actions, states, reward_settings, max_episode_length, flag_collision, wall_contact_force,
        bottle_contact_force
):
    # type: (Tensor, Tensor, Tensor, Dict[str, Tensor], Dict[str, float], float, Tensor, Tensor) -> Tuple[Tensor, Tensor]

    # Compute per-env physical parameters
    init_bottleA_height = states["bottleA_height"]
    # print('init_bottleA_height:     {}'.format(init_bottleA_height))
    # distance from hand to the bottleA
    d = torch.norm(states["bottleA_pos_relative"], dim=-1)
    d_lf = torch.norm(states["bottleA_pos"] - states["eef_lf_pos"], dim=-1)
    d_rf = torch.norm(states["bottleA_pos"] - states["eef_rf_pos"], dim=-1)
    dist_reward = 1 - torch.tanh(10.0 * (d + d_lf + d_rf) / 3)

    # penalty for empty grasping
    d_gripper = torch.norm(states["eef_lf_pos"] - states["eef_rf_pos"], dim=-1)
    penalty_gripper_closing = (d_gripper < 1e-2)

    # penalty for action jitter
    delta_action = (actions - states['prev_actions'])
    penalty_action_jitter = (delta_action ** 2).sum(dim=1)

    # Observe-to-grasp reward for RL training !!
    r_o2g = states["r_o2g"]

    # success for grasp pos final
    d_gripper_grasp_pos = states['d_gripper_grasp_pos']
    flag_success_gripper_grasp = (d_gripper_grasp_pos < 0.05)
    # print('d_gripper_grasp_pos:   {}'.format(d_gripper_grasp_pos))
    # Compose rewards
    rewards = reward_settings["r_dist_scale"] * dist_reward \
              + reward_settings["p_collision_scale"] * (flag_collision | bottle_contact_force) \
              + reward_settings["p_action_jitter_scale"] * penalty_action_jitter \
              + reward_settings["r_o2g_reward_scale"] * r_o2g \
              + reward_settings["r_d_gripper_grasp_pos"] * flag_success_gripper_grasp
    # print('!!!!!!   r_o2g:   {}'.format(r_o2g))

    reward_dict = {
        'reward_dist': dist_reward,
        'reward_dist_scale': reward_settings["r_dist_scale"] * dist_reward,
        'r_gg': states['r_gg'],
        'r_gg_scale': states['r_gg'] * states['sigma'],
        'r_go': states['r_go'],
        'r_go_scale': states['r_go'] * (1 - states['sigma']),
        'r_o2g': r_o2g,
        'r_o2g_scale': reward_settings["r_o2g_reward_scale"] * r_o2g,
        'flag_success_gripper_grasp': flag_success_gripper_grasp,
        'flag_success_gripper_grasp_scale': reward_settings["r_d_gripper_grasp_pos"] * flag_success_gripper_grasp,

    }

    if DEBUG_REWARD:
        print('!!! rewards:   {}'.format(rewards))
        print('$$  dist_reward:    {};  flag_success_gripper_grasp:    {}; r_o2g:   {}'.format(
            reward_settings["r_dist_scale"] * dist_reward,
            reward_settings["r_d_gripper_grasp_pos"] * flag_success_gripper_grasp,
            reward_settings['r_o2g_reward_scale'] * r_o2g
        ))
        # print('&&  penalty_gripper_closing:    {};  flag_collision:    {};'.format(
        #     reward_settings["p_gripper_scale"] * penalty_gripper_closing,
        #     reward_settings["p_collision_scale"] * (flag_collision | bottle_contact_force),
        # ))
    # We either provide the stack reward or the align + dist reward
    # rewards = torch.where(
    #     stack_reward,
    #     reward_settings["r_stack_scale"] * stack_reward,
    #     reward_settings["r_dist_scale"] * dist_reward + reward_settings["r_lift_scale"] * lift_reward + reward_settings[
    #         "r_align_scale"] * align_reward,
    # )

    # Compute resets
    reset_buf = torch.where((progress_buf >= max_episode_length - 1), torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where((flag_collision == True), torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where((flag_success_gripper_grasp == True), torch.ones_like(reset_buf), reset_buf)
    return rewards, reset_buf, reward_dict


