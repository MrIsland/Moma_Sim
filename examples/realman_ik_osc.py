import pdb

from isaacgym import gymapi, gymtorch
import torch

import numpy as np

def control_ik(dpose):
    global damping, j_eef, num_envs
    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device='cpu') * (damping ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, 6)
    return u


np.random.seed(42)

torch.set_printoptions(precision=4, sci_mode=False)

# 初始化 gym
gym = gymapi.acquire_gym()

# 创建模拟环境
sim_params = gymapi.SimParams()
sim_params.dt = 1.0 / 60.0
sim_params.use_gpu_pipeline = False
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
damping = 0.05
num_envs = 1
if sim is None:
    print("*** Failed to create sim")
    quit()

# 创建 viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())

# 加载机器人 asset（没有夹爪的 URDF）
asset_root = "../../assets"
asset_file = "urdf/franka_description/robots/realman.urdf"
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.armature = 0.01
robot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)


robot_dof_props = gym.get_asset_dof_properties(robot_asset)
robot_dof_props["driveMode"][:6].fill(gymapi.DOF_MODE_POS)
robot_dof_props["stiffness"][:6].fill(5000.0)
robot_dof_props["damping"][:6].fill(100.0)
# 创建一个环境
env = gym.create_env(sim, gymapi.Vec3(-1.0, 0.0, 0.0), gymapi.Vec3(1.0, 1.0, 1.0), 1)
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0, 0, 0)
robot_handle = gym.create_actor(env, robot_asset, pose, "robot", 0, 1)

gym.set_actor_dof_properties(env, robot_handle, robot_dof_props)

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

# 获取 DOF 状态和刚体状态
_dof_state_tensor = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_state_tensor)
_dof_pos = dof_states[:, 0].view(1, 6, 1)
_dof_vel = dof_states[:, 1].view(1, 6, 1)
_rb_state_tensor = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(_rb_state_tensor)

# 获取末端执行器的索引
ee_idx = gym.find_actor_rigid_body_index(env, robot_handle, "arm_link6", gymapi.DOMAIN_ENV)
_jacobian = gym.acquire_jacobian_tensor(sim, "robot")
jacobian = gymtorch.wrap_tensor(_jacobian)
j_eef = jacobian[:, 5, :, :6]

# 设定两个 EE 目标位置（世界坐标系下）
ee_pos_a = torch.tensor([-0.230687, 0.182896, 0.546390], device='cpu')
ee_pos_b = torch.tensor([0.285074, 0.261468, 0.483137], device='cpu')
target_a = True
# 控制循环
cnt = 0
while not gym.query_viewer_has_closed(viewer):
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    cnt += 1
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)
    gym.refresh_mass_matrix_tensors(sim)

    ee_pos = rb_states[ee_idx, 0:3]

    # 判断目标切换
    target_pos = ee_pos_a if target_a else ee_pos_b
    print('dis:     {}'.format(torch.norm(ee_pos - target_pos)))
    if torch.norm(ee_pos - target_pos) < 0.01:
        target_a = not target_a
        target_pos = ee_pos_a if target_a else ee_pos_b
        print('stitch  !!!!!!!!!!!!!')

    # 简单 IK 控制
    pos_err = target_pos - ee_pos
    orn_err = torch.zeros(3, device='cpu')  # 不控制姿态
    dpose = torch.cat([pos_err, orn_err]).unsqueeze(0).unsqueeze(-1)
    # pdb.set_trace()
    # 调用 IK 控制器（假设你有 control_ik）
    pos_action = torch.zeros((1, 6), device='cpu')
    pos_action[:, :6] = _dof_pos.squeeze(-1)[:, :6] + control_ik(dpose)
    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_action))

    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
