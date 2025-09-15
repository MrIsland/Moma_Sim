## Moma_Sim_ROS程序

### 1. moma_bringup

​	主要程序包括两个`moma_controller.py`和`moma_state_publisher.py`
​	**I. `moma_controller.py`**
    该程序实现了一个mujoco中的controller部分。其从外部不断地读取command，然后在仿真中完成机器人的运动。​     
​	**II. `moma_state_publisher.py`**
    该程序实现了将当前仿真中的所有观察以rostopic的方式publish出来，其中信息包括
    1. 移动机器人的state状态
    2. 底盘的速度
    3. rgbd图像
    4. 手臂的joint state
    5. 当前的里程计 odometry
    6. lidar坐标系下的ee pos

    TODO: I. `moma_controller.py`中还没有完成self._get_observation()函数
          II. 需要一个程序，将两个程序同步启动起来，可以完成实时的state publish和mujoco controller.

### 2. moma_perception

​	主要程序包括两个`model_inference.py`和`moma_perception_main.py`
​	**I. `model_inference.py`**
    顾名思义，input observation; output action
   **II. `moma_perception_main.py`**
    多线程程序:
    1. 数据接受线程
    2. 数据处理线程 - 图像，grasp pos
    3. 将data传到policy网络，输出action