import numpy as np
import os
import copy

from isaacgym import gymtorch
from isaacgym import gymapi
import torch

from isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, tensor_clamp, quat_apply
from isaacgymenvs.tasks.base.vec_task import VecTask

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


class MomaMovePick(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.moma_position_noise = self.cfg["env"]["momaPositionNoise"]
        self.moma_rotation_noise = self.cfg["env"]["momaRotationNoise"]
        self.moma_dof_noise = self.cfg["env"]["momaDofNoise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        # Create dicts to pass to reward function
        self.reward_settings = {
            "r_dist_scale": self.cfg["env"]["distRewardScale"],
            "r_lift_scale": self.cfg["env"]["liftRewardScale"],
            "r_lift_always_scale": self.cfg["env"]["liftRewardScale_always"],
            "r_target_pos_scale": self.cfg["env"]["targetposRewardScale"],
            "p_gripper_scale": self.cfg["env"]["gripperPenaltyScale"],
            "p_collision_scale": self.cfg["env"]["collisionPenaltyScale"]
        }

        # Controller type
        self.control_type = self.cfg["env"]["controlType"]
        assert self.control_type in {"osc", "joint_tor"}, \
            "Invalid control type specified. Must be one of: {osc, joint_tor}"

        # dimensions
        # obs include: cubeA_pose (7) + base_pose(7) + eef_pose (6) + q_gripper (6)
        # obs include: cubeA_pose (7) + eef_pose (7) + q_gripper (6)  20
        # obs include: cubeA_pose (7) + eef_pose (7) + q_gripper (6) + target_cubeA_pose(7) 28
        # obs include: cubeA_pose (7) + eef_pose (7) + q_gripper (6) + cubeA_pos_relative_robot_2d(2) + robot_vel(6)  = 28
        self.cfg["env"]["numObservations"] = 29 if self.control_type == "osc" else 26
        # actions include: delta EEF if OSC (6) or joint torques (7) + bool gripper (1)
        # actions include: base_pose(2) + delta EEF if OSC (6) or joint torques (7) + bool gripper (1)
        self.cfg["env"]["numActions"] = 9 if self.control_type == "osc" else 10

        # Values to be filled in at runtime
        self.states = {}  # will be dict filled with relevant states to use for reward calculation
        self.handles = {}  # will be dict mapping names to relevant sim handles
        self.num_dofs = None  # Total number of DOFs per env
        self.actions = None  # Current actions to be deployed
        self._init_cubeA_state = None  # Initial state of cubeA for the current env
        # self._init_cubeB_state = None           # Initial state of cubeB for the current env
        self._cubeA_state = None  # Current state of cubeA for the current env
        # self._cubeB_state = None                # Current state of cubeB for the current env
        self._cubeA_id = None  # Actor ID corresponding to cubeA for a given env
        # self._cubeB_id = None                   # Actor ID corresponding to cubeB for a given env

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
        # self._flag_self_collision = False  # Self collision detection
        # self._flag_obstacle_collision = False  # Obstacle collision detection

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        self._contact_bodies = ['arm_base_link', 'arm_link1', 'arm_link2', 'arm_link3',
                               'arm_link4', 'arm_link5', 'arm_link6', 'link_7']
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device,
                         graphics_device_id=graphics_device_id, headless=headless,
                         virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        self.moma_default_dof_pos = to_torch(
            [-0.67, 0.0, 0.65, 0.0, 1.36, 1.57, 0.5, -0.5, 0.5, 0.5, 0.5, -0.5], device=self.device
        )
        self.moma_default_state_pos = to_torch(
            [-0.7, -0.7, 0.0, 0.0, 0.0, 0.707, 0.707], device=self.device
            # [-0.2, -0.5, 0.0, 0.0, 0.0, 0.707, 0.707], device=self.device
        )

        # 需要检测碰撞的bodies
        # OSC Gains  eef_pose | joint
        self.kp = to_torch([150.] * 6, device=self.device)
        self.kd = 2 * torch.sqrt(self.kp)
        self.kp_null = to_torch([10.] * 6, device=self.device)
        self.kd_null = 2 * torch.sqrt(self.kp_null)
        # self.cmd_limit = None                   # filled in later
        self.cmd_limit = to_torch([0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device).unsqueeze(0) if \
            self.control_type == "osc" else self._moma_effort_limits[:7].unsqueeze(0)

        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

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
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.static_friction = 0.0  # 静摩擦系数
        plane_params.dynamic_friction = 0.0  # 动摩擦系数
        plane_params.restitution = 0.0  # 弹性（反弹程度）
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        Moma_asset_file = "urdf/IslandMomaRobot/urdf/island_moma_robot.urdf"
        # bottle_asset_file = "urdf/object/meshdatav3_scaled/sem/Bottle-f452c1053f88cd2fc21f7907838a35d1/coacd/coacd_012.urdf"
        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      self.cfg["env"]["asset"].get("assetRoot", asset_root))
            Moma_asset_file = self.cfg["env"]["asset"].get("assetFileNameMoma", Moma_asset_file)
            # bottle_asset_file = self.cfg["env"]["asset"].get("assetFileNamebottle", bottle_asset_file)


        # load moma asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = False
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = False
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.use_mesh_materials = True
        moma_asset = self.gym.load_asset(self.sim, asset_root, Moma_asset_file, asset_options)


        # bottle_asset_options = gymapi.AssetOptions()
        # bottle_asset_options.fix_base_link = False
        # bottle_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX  # or NONE if normals already exist
        # bottle_asset_options.disable_gravity = False
        # bottle_asset_options.use_mesh_materials = True
        # bottle_asset = self.gym.load_asset(self.sim, asset_root, bottle_asset_file, bottle_asset_options)

        moma_dof_stiffness = to_torch([0, 0, 0, 0, 0, 0, 5000, 5000, 5000, 5000, 5000, 5000], dtype=torch.float,
                                      device=self.device)
        moma_dof_damping = to_torch([0, 0, 0, 0, 0, 0, 100, 100, 100, 100, 100, 100], dtype=torch.float,
                                    device=self.device)

        # Create table asset
        table_pos = [0.0, 0.0, 0.4]
        table_thickness = 0.05
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, *[0.8, 0.8, table_thickness], table_opts)

        # # Create table stand asset
        # table_stand_height = 0.1
        # table_stand_pos = [-0.5, 0.0, 1.0 + table_thickness / 2 + table_stand_height / 2]
        # table_stand_opts = gymapi.AssetOptions()
        # table_stand_opts.fix_base_link = True
        # table_stand_asset = self.gym.create_box(self.sim, *[0.2, 0.2, table_stand_height], table_opts)

        self.cubeA_size = 0.05

        # Create cubeA asset
        cubeA_opts = gymapi.AssetOptions()
        cubeA_asset = self.gym.create_box(self.sim, *([self.cubeA_size] * 3), cubeA_opts)
        cubeA_color = gymapi.Vec3(0.6, 0.1, 0.0)

        self.num_moma_bodies = self.gym.get_asset_rigid_body_count(moma_asset)
        self.num_moma_dofs = self.gym.get_asset_dof_count(moma_asset)

        print("num moma bodies: ", self.num_moma_bodies)
        print("num moma dofs: ", self.num_moma_dofs)

        moma_dof_props = self.gym.get_asset_dof_properties(moma_asset)
        self.moma_dof_lower_limits = []
        self.moma_dof_upper_limits = []
        self._moma_effort_limits = []

        for i in range(self.num_moma_dofs):
            if i > 11:
                break
            moma_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS if i > 5 else gymapi.DOF_MODE_EFFORT
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
        self._table_surface_pos = np.array(table_pos) + np.array([0, 0, table_thickness / 2])
        self.reward_settings["table_height"] = self._table_surface_pos[2]

        # Define start pose for cubes (doesn't really matter since they're get overridden during reset() anyways)
        cubeA_start_pose = gymapi.Transform()
        cubeA_start_pose.p = gymapi.Vec3(-0.3, 0.0, 0.8)
        cubeA_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # compute aggregate size
        num_moma_bodies = self.gym.get_asset_rigid_body_count(moma_asset)
        num_moma_shapes = self.gym.get_asset_rigid_shape_count(moma_asset)
        # todo: 增加墙的时候这部分要修改
        max_agg_bodies = num_moma_bodies + 2  # 1 for table, 1 for object, 4 for walls
        max_agg_shapes = num_moma_shapes + 2  # 1 for table, 1 for object, 4 for walls

        self.momas = []
        self.envs = []

        # Create environments
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: moma should ALWAYS be loaded first in sim!
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create moma
            # Potentially randomize start pose
            if self.moma_position_noise > 0:
                rand_xy = self.moma_position_noise * (-1. + np.random.rand(2) * 2.0)
                moma_start_pose.p = gymapi.Vec3(-0.45 + rand_xy[0], 0.0 + rand_xy[1], 0.0)

            if self.moma_rotation_noise > 0:
                rand_rot = torch.zeros(1, 3)
                rand_rot[:, -1] = self.moma_rotation_noise * (-1. + np.random.rand() * 2.0)
                new_quat = axisangle2quat(rand_rot).squeeze().numpy().tolist()
                moma_start_pose.r = gymapi.Quat(*new_quat)

            moma_actor = self.gym.create_actor(env_ptr, moma_asset, moma_start_pose, "moma", i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, moma_actor, moma_dof_props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create table
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0)
            # self.gym.set_rigid_body_color(env_ptr, table_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, cubeA_color)
            # table_stand_actor = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_start_pose, "table_stand",
            #                                           i, 1, 0)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            self._cubeA_id = self.gym.create_actor(env_ptr, cubeA_asset, cubeA_start_pose, "cubeA", i, 2, 0)

            # Set colors
            self.gym.set_rigid_body_color(env_ptr, self._cubeA_id, 0, gymapi.MESH_VISUAL, cubeA_color)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.momas.append(moma_actor)

        # Setup init state buffer
        self._init_cubeA_state = torch.zeros(self.num_envs, 13, device=self.device)

        # Setup data
        self.init_data()

    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        moma_handle = 0
        self.handles = {
            # moma

            # For calculation
            "left_pad": self.gym.find_actor_rigid_body_handle(env_ptr, moma_handle, "left_pad"),
            "right_pad": self.gym.find_actor_rigid_body_handle(env_ptr, moma_handle, "right_pad"),
            "gripper_site": self.gym.find_actor_rigid_body_handle(env_ptr, moma_handle, "gripper_site"),

            # Table
            "Table_handle": self.gym.find_actor_rigid_body_handle(env_ptr, 1, 'box'),
            
            # Cubes
            "cubeA_body_handle": self.gym.find_actor_rigid_body_handle(env_ptr, self._cubeA_id, "box"),
        }

        for body in self._contact_bodies:
            # print('body:   {}'.format(body))
            self.handles.update({body: self.gym.find_actor_rigid_body_handle(env_ptr, moma_handle, body)})

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
        jacobian = gymtorch.wrap_tensor(_jacobian)  # Tensor(num_env, num_bodis, 6, num_dofs)
        gripper_joint_index = self.gym.get_actor_joint_dict(env_ptr, moma_handle)['joint_7']  # 包括所有的joint
        self._j_eef = jacobian[:, gripper_joint_index, :, :6]
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "moma")
        mm = gymtorch.wrap_tensor(_massmatrix)  # Tensor(num_env, num_dof, num_dof)
        self._mm = mm[:, :6, :6]
        self._cubeA_state = self._root_state[:, self._cubeA_id, :]
        self._moma_robot_state = self._root_state[:, self.momas[0], :]
        # Initialize states
        self.states.update({
            "cubeA_size": torch.ones_like(self._eef_state[:, 0]) * self.cubeA_size,
        })

        # Initialize actions
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float,
                                        device=self.device)  # num_dofs是活动关节数量
        self._effort_control = torch.zeros_like(self._pos_control)
        # self._robot_state_control = self._moma_robot_state.clone()

        # Initialize control
        self._arm_control = self._effort_control[:, :6]
        self._gripper_control = self._pos_control[:, 6:12]
        self._robot_pos_control = self._moma_robot_state[:, :7]
        self._robot_vel_control = self._moma_robot_state[:, 7:]

        # Initialize indices
        self._global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32,
                                            device=self.device).view(self.num_envs, -1)  # 1 for robot table cubeA

        # Initialize base vel
        # self._robot_state_vel_control = self._moma_robot_state[:, 7:]
        self._constant_base_vel = torch.tensor(self.cfg["constant_action"]["base_vel"])

    def _update_states(self):

        self.states.update({
            # Robot
            "q": self._q[:, :],
            "q_gripper": self._q[:, 6:12],
            "eef_pos": self._eef_state[:, :3],
            "eef_quat": self._eef_state[:, 3:7],
            "eef_vel": self._eef_state[:, 7:],
            "eef_lf_pos": self._eef_lf_state[:, :3],
            "eef_rf_pos": self._eef_rf_state[:, :3],
            "robot_pos": self._moma_robot_state[:, 0:7],
            "robot_vel": self._moma_robot_state[:, 7:],
            # Cubes
            "cubeA_quat": self._cubeA_state[:, 3:7],
            "cubeA_pos": self._cubeA_state[:, :3],
            "cubeA_pos_relative": self._cubeA_state[:, :3] - self._eef_state[:, :3],
            "target_cubeA_pos": self._init_cubeA_state[:, :3] + torch.tensor([[0, 0, 0.2]], device=self.device),
            "target_cubeA_quat": self._init_cubeA_state[:, 3:7],
            "cubeA_pos_relative_robot_2d": self._cubeA_state[:, :2] - self._moma_robot_state[:, :2],
            # todo orientation
            # state:
            "flag_collision": self.flag_collision.view(-1, 1),
        })

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)


        self.flag_collision = self.collision_detection()
        # print('flag_collision:    {}'.format(self.flag_collision))
        # if self.flag_collision.any():
        #     indices2 = torch.where(self.flag_collision)[0]
        #     print('Existing collision!!!!  indices:   {}'.format(indices2))

        # Refresh states
        self._update_states()


    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_moma_reward(
            self.reset_buf, self.progress_buf, self.actions, self.states, self.reward_settings,
            self.max_episode_length, self.flag_collision
        )

    def compute_observations(self):
        self._refresh()
        obs = ["cubeA_quat", "cubeA_pos", "eef_pos", "eef_quat", "robot_vel", "cubeA_pos_relative_robot_2d", "flag_collision"]  # 4 + 3 + 3 + 4 + 6 + 1
        obs += ["q_gripper"] if self.control_type == "osc" else ["q"]
        # print('compute_observations:  cubeA_pos-{}, cubeA_quat-{}'.format(self.states["cubeA_pos"], self.states["cubeA_quat"]))
        # print('compute_observations:  target_cubeA_pos-{}, target_cubeA_quat-{}'.format(self.states["target_cubeA_pos"], self.states["target_cubeA_quat"]))
        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)

        maxs = {ob: torch.max(self.states[ob]).item() for ob in obs}

        return self.obs_buf

    # # @ brief: cubeA的位置 random
    # def _reset_init_cube_state(self, cube, env_ids, check_valid=True):
    #     """
    #     Init the position of cube(env)
    #     """
    #     if env_ids is None:
    #         env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)
    #
    #     num_resets = len(env_ids)
    #     sampled_cube_state = torch.zeros(num_resets, 13, device=self.device)
    #
    #     if cube.lower() == 'a':
    #         this_cube_state_all = self._init_cubeA_state
    #         cube_heights = self.states["cubeA_size"]
    #
    #     # 让木块能在机械臂的行程之内
    #     offset_centered_cube_xy_state = torch.tensor([-0.2, 0.0], device=self.device, dtype=torch.float32)
    #     centered_cube_xy_state = torch.tensor(self._table_surface_pos[:2], device=self.device, dtype=torch.float32) + offset_centered_cube_xy_state
    #
    #     sampled_cube_state[:, 2] = self._table_surface_pos[2] + cube_heights[env_ids] / 2
    #     # sampled_cube_state[:, :2] = torch.tensor([-0.3, 0.0]).repeat(num_resets, 1)
    #     sampled_cube_state[:, 6] = 1.0
    #
    #     sampled_cube_state[:, :2] = centered_cube_xy_state.unsqueeze(0) + \
    #                                 2.0 * self.start_position_noise * (
    #                                         torch.rand(num_resets, 2, device=self.device) - 0.5)
    #
    #     if self.start_rotation_noise > 0:
    #         aa_rot = torch.zeros(num_resets, 3, device=self.device)
    #         aa_rot[:, 2] = 2.0 * self. start_rotation_noise * (torch.rand(num_resets, device=self.device) - 0.5)
    #         sampled_cube_state[:, 3:7] = quat_mul(axisangle2quat(aa_rot), sampled_cube_state[:, 3:7])
    #
    #     # lastly, set these sampled values as the new init state.
    #     this_cube_state_all[env_ids, :] = sampled_cube_state

    # @ brief: cubeA的位置固定
    def _reset_init_cube_state(self, cube, env_ids, check_valid=True):
        """
        Init the position of cube(env)
        """
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)

        # Initialize buffer to hold sampled values
        num_resets = len(env_ids)
        sampled_cube_state = torch.zeros(num_resets, 13, device=self.device)  # [x y z qx qy qz qw vx vy vz wx wy wz]

        if cube.lower() == 'a':
            this_cube_state_all = self._init_cubeA_state
            # other_cube_state = self._init_cubeB_state[env_ids, :]
            cube_heights = self.states["cubeA_size"]

        sampled_cube_state[:, 2] = self._table_surface_pos[2] + cube_heights[env_ids] / 2
        sampled_cube_state[:, 6] = 1.0
        sampled_cube_state[:, :2] = torch.tensor([-0.3, 0.0]).repeat(num_resets, 1)

        this_cube_state_all[env_ids, :] = sampled_cube_state

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

    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # if not self._i:
        self._reset_init_cube_state(cube='A', env_ids=env_ids, check_valid=True)
        # self._i = True

        # Write these new init states to the sim states
        self._cubeA_state[env_ids] = self._init_cubeA_state[env_ids]

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

        ## todo velocity control reset
        self._robot_vel_control[env_ids, :] = torch.zeros_like(self._robot_vel_control[env_ids, :])
        self._robot_pos_control[env_ids, :] = self.moma_default_state_pos.repeat(len(env_ids), 1)

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
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_state),
                                                     gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                     len(multi_env_ids_int32))

        # Update cube states
        multi_env_ids_cubes_int32 = self._global_indices[env_ids, -1].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_cubes_int32), len(multi_env_ids_cubes_int32))
        # print('reset_idx:     {}'.format(self._root_state))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._init_states = None
        self.flag_collision = False
        # self._flag_self_collision = False
        # self._flag_arm_obs_collision = False

    def collision_detection(self, collision_threshold = 0.1):
        target_indices = [self.handles[link_name] for link_name in self._contact_bodies]
        target_forces = self._contact_force[:, target_indices, :]
        # print('self._contact_force:    {}'.format(self._contact_force))
        force_norms = torch.norm(target_forces, dim=2)
        collision_flags = (force_norms > collision_threshold).any(dim=1)
        return collision_flags

    def move_gripper(self, u_gripper):
        u_fingers = torch.zeros_like(self._gripper_control)
        tmp_u_finger = torch.where(u_gripper >= 0.0, 1.0, 0.0)
        u_fingers = torch.concatenate([tmp_u_finger.unsqueeze(1),
                                       -1 * tmp_u_finger.unsqueeze(1),
                                       tmp_u_finger.unsqueeze(1),
                                       tmp_u_finger.unsqueeze(1),
                                       tmp_u_finger.unsqueeze(1),
                                       -1 * tmp_u_finger.unsqueeze(1)], dim=1)
        return u_fingers

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        u_wheel, u_arm, u_gripper = self.actions[:, :2], self.actions[:, 2:-1], self.actions[:, -1]
        u_arm = u_arm * self.cmd_limit / self.action_scale
        if self.control_type == "osc":
            u_arm = self._compute_osc_torques(dpose=u_arm)
        self._arm_control[:, :] = u_arm

        self._gripper_control[:, :] = self.move_gripper(u_gripper)

        self._robot_pos_control[:, :3] = self._moma_robot_state[:, :3]
        self._robot_pos_control[:, 3:7] = torch.tensor([0, 0, 0.707, 0.707], device=self.device)
        self._robot_vel_control[:, :] = self._constant_base_vel.repeat(u_arm.shape[0], 1)

        # Deploy actions
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))

        env_ids = torch.arange(self.num_envs, device=self.device)
        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_state),
                                                     gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                     len(multi_env_ids_int32))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        # print('self._contact_state:   {}'.format(self._contact_state))
        self.compute_observations()
        self.compute_reward(self.actions)

        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            # Grab relevant states to visualize
            eef_pos = self.states["eef_pos"]
            eef_rot = self.states["eef_quat"]
            cubeA_pos = self.states["cubeA_pos"]
            cubeA_rot = self.states["cubeA_quat"]

            # Plot visualizations
            for i in range(self.num_envs):
                for pos, rot in zip((eef_pos, cubeA_pos), (eef_rot, cubeA_rot)):
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
        reset_buf, progress_buf, actions, states, reward_settings, max_episode_length, flag_collision,
):
    # type: (Tensor, Tensor, Tensor, Dict[str, Tensor], Dict[str, float], float, Tensor) -> Tuple[Tensor, Tensor]

    # Compute per-env physical parameters
    cubeA_size = states["cubeA_size"]

    # distance from hand to the cubeA
    d = torch.norm(states["cubeA_pos_relative"], dim=-1)
    d_lf = torch.norm(states["cubeA_pos"] - states["eef_lf_pos"], dim=-1)
    d_rf = torch.norm(states["cubeA_pos"] - states["eef_rf_pos"], dim=-1)
    dist_reward = 1 - torch.tanh(10.0 * (d + d_lf + d_rf) / 3)

    # reward for lifting cubeA
    cubeA_height = states["cubeA_pos"][:, 2] - reward_settings["table_height"]
    cubeA_lifted = (cubeA_height - cubeA_size) > 0.04
    lift_reward = cubeA_lifted

    # penalty for empty grasping
    d_gripper = torch.norm(states["eef_lf_pos"] - states["eef_rf_pos"], dim=-1)
    penalty_gripper_closing = (d_gripper < 1e-2)

    # lift the cube to 0.2m more
    target_cubeA_pos = states["target_cubeA_pos"]
    d_cubeA_pos = torch.norm(states["cubeA_pos"] - target_cubeA_pos, dim=-1)
    dist_target_pos_reward = 1 - torch.tanh(10.0 * d_cubeA_pos)

    # panelty for collision

    # Compose rewards

    rewards = reward_settings["r_dist_scale"] * dist_reward \
              + reward_settings["r_lift_scale"] * lift_reward \
              + reward_settings["p_gripper_scale"] * penalty_gripper_closing \
              + reward_settings["p_collision_scale"] * flag_collision

    # We either provide the stack reward or the align + dist reward
    # rewards = torch.where(
    #     stack_reward,
    #     reward_settings["r_stack_scale"] * stack_reward,
    #     reward_settings["r_dist_scale"] * dist_reward + reward_settings["r_lift_scale"] * lift_reward + reward_settings[
    #         "r_align_scale"] * align_reward,
    # )

    # Compute resets
    reset_buf = torch.where((progress_buf >= max_episode_length - 1), torch.ones_like(reset_buf), reset_buf)


    return rewards, reset_buf

