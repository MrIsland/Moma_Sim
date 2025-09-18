#!/home/island/.conda/envs/rlgpu/bin/python3
import rospy
import rospkg
import numpy as np
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, Pose, Twist
import threading
from sensor_msgs.msg import Image, CompressedImage
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import String, Bool
from moma_perception.msg import moma_cmd
from moma_bringup.msg import moma_state
from message_filters import Subscriber, TimeSynchronizer, ApproximateTimeSynchronizer
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R
import sys
import os
import pdb
# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..')
sys.path.insert(0, project_root)

from utils.graspnet_wrapper import run_graspnet_demo
from utils.openseed_new.OpenSeeD.detection import detection
from utils.common import *
from queue import Queue

import cv2
import pdb
import time
import random
from model_Inference import ModelInference

class MomaPerception:
    def __init__(self, cfg_path):
        rospy.init_node('MomaPerception')
        rospy.loginfo("[MomaPerception] Starting")

        # 先初始化所有成员变量
        self.moma_cmd_pub = rospy.Publisher('/moma_cmd', moma_cmd, queue_size=10)
        self.rgb_msg = None
        self.depth_msg = None
        self.odom_msg = None
        self.joint_state_msg = None
        self.twist_msg = None
        self.epos_msg = None
        self.gripper_state_msg = None
        self.message_lock = threading.Lock()
        self.data_queue = Queue(maxsize=1)  # 只保存最新的数据
        self.processing_event = threading.Event()  # 用于通知处理线程
        self.message_count = 0
        self.processing_count = 0
        self.cfg = load_config(cfg_path)
        random.seed()
        self.cnt = 0
        self.bridge = CvBridge()
        self.camera_info = self.cfg['camera_info']
        self.ModelInference = ModelInference()

        # 再注册回调
        # rospy.Subscriber('/moma_state/rgb_image', CompressedImage, self.rgb_callback)
        # rospy.Subscriber('/moma_state/depth_image', Image, self.depth_callback)
        rospy.Subscriber('/moma_state/odometry', Odometry, self.odom_callback)
        rospy.Subscriber('/moma_state/joint_state', JointState, self.joint_state_callback)
        rospy.Subscriber('/moma_state/chassis_twist', Twist, self.twist_callback)
        rospy.Subscriber('/moma_state/epos_in_lidar', Pose, self.epos_callback)
        rospy.Subscriber('/moma_state/gripper_state', Bool, self.gripper_state_callback)
        self.epos_cnt, self.odom_cnt, self.joint_state_cnt, self.twist_cnt, self.gripper_cnt = 0, 0, 0, 0, 0
        # 启动两个线程
        self.data_receive_thread = threading.Thread(target=self.data_receive_loop, name="DataReceiveThread")
        # self.data_process_thread = threading.Thread(target=self.data_process_loop, name="DataProcessThread")
        self.data_receive_thread.daemon = True
        # self.data_process_thread.daemon = True
        self.data_receive_thread.start()
        # self.data_process_thread.start()

    def rgb_callback(self, msg):
        with self.message_lock:
            self.rgb_msg = msg
            
    def depth_callback(self, msg):
        with self.message_lock:
            self.depth_msg = msg
            
    def odom_callback(self, msg):
        with self.message_lock:
            self.odom_msg = msg
            self.odom_cnt += 1
            print('!!!!  self.odom_cnt:       {}'.format(self.odom_cnt))
            
    def joint_state_callback(self, msg):
        with self.message_lock:
            self.joint_state_msg = msg
            self.joint_state_cnt += 1
            print('!!!!  self.joint_state_cnt:       {}'.format(self.joint_state_cnt))
            
    def twist_callback(self, msg):      
        with self.message_lock:
            self.twist_msg = msg

    def epos_callback(self, msg):
        with self.message_lock:
            self.epos_msg = msg
            self.epos_cnt += 1
            print('!!!!  self.epos_cnt:       {}'.format(self.epos_cnt))

    def gripper_state_callback(self, msg): 
        with self.message_lock:
            self.gripper_state_msg = msg

    def moma_cmd_publish_from_list(self, cnt):
        """发布Moma命令"""
        cmd_arm_limit = np.array([0.05, 0.05, 0.05, 0.1, 0.1, 0.1])
        cmd_base_limit = np.array([0.3, 0.6])

        # action_file_path = "/home/igrape/Moma_Sim_ROS/src/moma_perception/test_action.txt"
        package_path = rospack.get_path('moma_perception')
        action_file_path = os.path.join(package_path, 'test_action.txt')
        with open(action_file_path, 'r') as f:
            action_data = [line.strip() for line in f if line.strip()]
        action_data_line = action_data[cnt]
        values = np.array(action_data_line.split(), dtype=float)
        # """读取eepos"""
        # curr_delta_epos_pos = values[:3]  # 修改为3个位置元素
        # curr_delta_epos_quat = values[3:]  # 对应调整四元数索引
        # curr_delta_epos_orn = R.from_quat(curr_delta_epos_quat).as_euler('xyz')
        # curr_delta_epos = np.concatenate([curr_delta_epos_pos, curr_delta_epos_orn])
        # gripper_state = 0.0
        """读取delta eepos"""
        base_data = values[:2]
        # base_data = base_data  #频率关系
        delta_epos_data = values[2:8]
        gripper_state = values[-1]
        curr_base =  base_data * cmd_base_limit
        curr_delta_epos = delta_epos_data * cmd_arm_limit / 25
        # curr_delta_epos = delta_epos_data
        # curr_base = base_data
        # curr_delta_epos = [0, 0, 0, 0, 0, 0]
        # curr_base = [0.0, 0.0]
    

        msg = moma_cmd()
        msg.header.stamp = rospy.Time.now()
        msg.data = np.array([
            # curr_base[0], curr_base[1] * 0.6553938 + 0.0148119,    ?????????????//
            curr_base[0], curr_base[1],
            curr_delta_epos[0], curr_delta_epos[1], curr_delta_epos[2],
            curr_delta_epos[3], curr_delta_epos[4], curr_delta_epos[5],
            gripper_state
        ], dtype=float)
        self.moma_cmd_pub.publish(msg)
        # print(msg.data)

    def moma_cmd_publish_from_action(self, action):
        cmd_arm_limit = np.array([0.05, 0.05, 0.05, 0.1, 0.1, 0.1])
        cmd_base_limit = np.array([0.3, 0.6])
        
        action = action.flatten()
        print('!!!!!!!!!action:     {}'.format(action))
        chassis_vel = action[:2]
        # delta_ee_pos = [0,0,0,0,0,0]
        delta_ee_pos = action[2:8]
        gripper_state = action[-1]
        curr_base = chassis_vel * cmd_base_limit
        curr_delta_epos = delta_ee_pos * cmd_arm_limit / 25
        # curr_delta_epos = delta_ee_pos * cmd_arm_limit
        curr_epos = curr_delta_epos

        # curr_base = [0,0]
        msg = moma_cmd()
        msg.header.stamp = rospy.Time.now()
        msg.data = np.array([curr_base[0],curr_base[1],curr_epos[0],curr_epos[1],curr_epos[2],curr_epos[3],curr_epos[4],curr_epos[5],gripper_state],dtype=float)
        print('moma_cmd_publish_from_action:    {}'.format(msg.data))
        self.moma_cmd_pub.publish(msg)
    
    def data_receive_loop(self):
        """数据接收线程：负责检查消息完整性并计数"""
        print("[DEBUG] Starting data receive loop...")
        rate = rospy.Rate(60)  # 高频率检查数据
        
        while not rospy.is_shutdown():
            with self.message_lock:
                # 检查是否所有必要的消息都已接收
                if all([
                    # self.rgb_msg,
                    # self.depth_msg,
                    self.odom_msg,
                    self.joint_state_msg,
                    self.twist_msg,
                    self.epos_msg,
                    self.joint_state_msg
                ]):
                    data_package = {
                            # 'rgb_msg': self.rgb_msg,
                            # 'depth_msg': self.depth_msg,
                            'odom_msg': self.odom_msg,
                            'joint_state_msg': self.joint_state_msg,
                            'twist_msg': self.twist_msg,
                            'epos_msg': self.epos_msg,
                            'gripper_state_msg': self.gripper_state_msg,
                            'count': self.message_count
                        }
                    action = self.ModelInference.get_action(data_package, None, 'state', is_deterministic=True)
                    # print('!!! action      {}'.format(action))
                    self.moma_cmd_publish_from_action(action)    # TODO:  这里发出来的action动作都特别的小。。。
                    # self.moma_cmd_publish_from_list(self.message_count)
                    self.message_count += 1
                    print("message_count:        {}".format(self.message_count))
                    # print(f"[DEBUG] Data receive thread: message count = {self.message_count}")
                    
                    # 每接收到20次完整消息时，传递最新数据给处理线程
                    # if self.message_count % 20 == 0:
                    #     if self.data_queue.full():
                    #         try:
                    #             old_data = self.data_queue.get_nowait()
                    #         except Exception as e:
                    #             print(f"[WARNING] Failed to remove old data from queue: {e}")
                    #     try:
                    #         self.data_queue.put_nowait(data_package)
                    #         self.processing_event.set()  # 通知处理线程
                    #     except Exception as e:
                    #         print(f"[WARNING] Failed to put data into queue: {e}")
            
            rate.sleep()

    def data_process_loop(self):
        """数据处理线程：负责运行OpenSeeD和GraspNet"""
        print("[DEBUG] Starting data process loop...")
        
        while not rospy.is_shutdown():
            # 等待处理事件
            self.processing_event.wait(timeout=1.0)
            
            if not self.data_queue.empty():
                try:
                    # 获取数据包
                    data_package = self.data_queue.get_nowait()
                    self.processing_event.clear()  # 清除事件标志
                    
                    self.processing_count += 1
                    print(f"[DEBUG] Processing thread: starting processing #{self.processing_count} (from message count: {data_package['count']})")
                    
                    # 处理数据
                    start_time = time.time()
                    result = self.process_image(data_package['rgb_msg'], data_package['depth_msg'])
                    
                    if result is not None:
                        depth_img, goal_mask, color_img = result
                        self.graspnet_process(color_img, depth_img, goal_mask)
                        
                        processing_time = time.time() - start_time
                        print(f"[DEBUG] Processing #{self.processing_count} completed in {processing_time:.2f}s")
                    
                except Exception as e:
                    print(f"[ERROR] Error in data processing thread: {e}")
                    import traceback
                    traceback.print_exc()

    def _get_semantic_2D_mask_openseed(self, color_img, idx):
        goal_mask = np.zeros((color_img.shape[0], color_img.shape[1]), dtype=bool)
        mask = detection(color_img, idx)
        mask = mask.astype(bool)
        if mask is not None:
            goal_mask |= mask
        return goal_mask

    def process_image(self, rgb_msg, depth_msg):
        # 处理深度图像
        depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        depth_img = cv2.resize(depth_img, (self.camera_info['width'], self.camera_info['height']))
        
        # 处理RGB图像
        color_img = self.bridge.compressed_imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        color_img = cv2.resize(color_img, (self.camera_info['width'], self.camera_info['height']))
        
        # 获取语义掩码
        goal_mask = self._get_semantic_2D_mask_openseed(color_img, self.cnt)
        
        return depth_img, goal_mask, color_img

    def graspnet_process(self, color_img, depth_img, goal_mask):
        try:
            # GraspNet期望的图像尺寸 (1280x720)
            graspnet_width, graspnet_height = 1280, 720
            
            # 调整图像尺寸以匹配GraspNet的要求
            color_img = cv2.resize(color_img, (graspnet_width, graspnet_height))
            depth_img = cv2.resize(depth_img, (graspnet_width, graspnet_height))
            goal_mask = cv2.resize(goal_mask.astype(np.uint8), (graspnet_width, graspnet_height))
            
            # 转换为RGB格式（如果需要）
            if len(color_img.shape) == 3 and color_img.shape[2] == 3:
                # 检查是否是BGR格式，如果是则转换为RGB
                color_img_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            else:
                color_img_rgb = color_img
                
            # 转换掩码格式
            mask = goal_mask.astype(np.uint8) * 255 
            
            # 调用graspnet demo通过wrapper
            result = run_graspnet_demo(color_img_rgb, depth_img, mask)
            return result
        
        except ImportError as ie:
            print(f"Import error in graspnet_process: {ie}")
            print("[DEBUG] Skipping graspnet processing due to import issues")
        except Exception as e:
            print(f"Error in graspnet_process: {e}")
            import traceback
            traceback.print_exc()
            
    def spin(self):
        rospy.spin()


if __name__ == '__main__':
    config_path = './configs/moma.yaml'
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('moma_perception')
    moma_perception = MomaPerception(os.path.join(package_path, config_path))
    moma_perception.spin()
