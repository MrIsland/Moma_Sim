import mujoco
import mujoco.viewer
import numpy as np
import time

# 加载模型和数据
model = mujoco.MjModel.from_xml_path("./assets/island_moma_robot/xml/rm.xml")
data = mujoco.MjData(model)
model.opt.gravity[:] = [0, 0, 0]
# 控制参数
kp = 1
kd = 0.01

# 初始和目标关节角度
q_start = np.array([-0.5, 0.3, -0.1, 0.5, -0.5, 0.3])  # 同上
q_goal = np.array([0.5, -0.3, 0.1, -0.5, 0.5, -0.3])  # 同上
# q_start = np.array([0.0, -0.0, 0.1, -0.5, 0.5, 0.3])  # 同上
# q_goal = np.array([0.0, -0.0, 0.1, -0.5, 0.5, -0.3])  # 同上

def reset_joint_state(q):
    data.qpos[:6] = q
    data.qvel[:6] = 0
    mujoco.mj_forward(model, data)

def position_control_step(q_desired):
    # 简单的 PD 控制器（MuJoCo 直接支持位置控制）
    # q = data.qpos[:6]
    # qdot = data.qvel[:6]
    # torque = kp * (q_desired - q) - kd * qdot
    data.ctrl[:6] = q_desired
    # print('torque:     {}'.format(torque))

def torque_control_step(q_desired):
    # 模拟力矩控制器（不依赖 model.actuator_gainprm）
    q = data.qpos[:6]
    qdot = data.qvel[:6]
    torque = kp * (q_desired - q) - kd * qdot
    data.qfrc_applied[:6] = torque
    print('torque:     {}'.format(torque))

def run_controller(mode="position", q_target=None, steps=500):
    for _ in range(steps):
        if mode == "position":
            position_control_step(q_target)
        elif mode == "torque":
            torque_control_step(q_target)
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)

# 初始化 viewer
with mujoco.viewer.launch_passive(model, data) as viewer:

    # 设置初始状态
    reset_joint_state(q_start)
    # for _ in range(100):
    #     mujoco.mj_step(model, data)
    viewer.sync()

    # print("开始位置控制：向目标移动")
    # run_controller(mode="position", q_target=q_goal, steps=50000)

    # print("返回原位")
    # run_controller(mode="position", q_target=q_start, steps=50000)

    print("使用力矩控制：向目标移动")
    # reset_joint_state(q_start)
    run_controller(mode="torque", q_target=q_goal, steps=5000)
    #
    print("返回原位（力矩控制）")
    run_controller(mode="torque", q_target=q_start, steps=5000)

    print("完成")
    time.sleep(2)
