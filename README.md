## Moma_Sim_ROS程序

### 1. moma_bringup

​	主要程序包括两个`moma_controller.py`和`moma_state_publisher.py`

**I. `moma_controller.py`**

该程序实现了一个mujoco中的controller部分。其从外部不断地读取command，然后在仿真中完成机器人的运动。​     

**II. `moma_state_publisher.py`**

该程序实现了将当前仿真中的所有观察以rostopic的方式publish出来，其中信息包括

1. 移动机器人的state状态
2. 底盘的速度
3. rgbd图像
4. 手臂的joint state
5. 当前的里程计 odometry
6. lidar坐标系下的ee pos



### 2. moma_perception

​	主要程序包括两个`model_inference.py`和`moma_perception_main.py`
**I. `model_inference.py`**

顾名思义，input observation; output action

**II. `moma_perception_main.py`**

多线程程序:

    1. 数据接受线程
    2. 数据处理线程 - 图像，grasp pos
    3. 将data传到policy网络，输出action



###################################

### Version 0923:

	以闭环控制的方式完成所有state从pybullet中出，不受到mujoco中的state的影响。 模拟的是真机中所有的state中出。
	isaacgym   ->    取消了重力
	pybullet   ->    没有重力
	mujoco     ->     有重力影响
	真机        ->       没有重力，较准确的执行。