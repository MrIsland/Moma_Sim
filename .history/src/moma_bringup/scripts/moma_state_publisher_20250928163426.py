#!/home/island/.conda/envs/rlgpu/bin/python3
import os, sys
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)
import rospy
import numpy as np
import pdb
import cv2
import rospkg
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R
from moma_bringup.msg import moma_state
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Header,Bool
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry  
from geometry_msgs.msg import Pose
from tf.transformations import quaternion_from_matrix
# from moma_bringup.scripts.moma_controller import MomaController
from utils.common import *

cur_position = 14
gripper_state = 1

class SyncAndPublish:
    def __init__(self, moma_controller):
        self.pub_state = rospy.Publisher("/moma_state", moma_state, queue_size=10)
        self.pub_odometry = rospy.Publisher("/moma_state/odometry", Odometry, queue_size=10)
        self.pub_rgb = rospy.Publisher("/moma_state/rgb_image", CompressedImage, queue_size=10)
        self.pub_depth = rospy.Publisher("/moma_state/depth_image", Image, queue_size=10)
        self.pub_joint_state = rospy.Publisher("/moma_state/joint_state", JointState, queue_size=10)
        self.pub_gripper_state = rospy.Publisher("/moma_state/gripper_state", Bool, queue_size=10)
        self.pub_epos_in_lidar = rospy.Publisher("/moma_state/epos_in_lidar", Pose, queue_size=10)
        self.pub_chassis_twist = rospy.Publisher("/moma_state/chassis_twist", Twist, queue_size=10)

        self.timer = rospy.Timer(rospy.Duration(0.01), self.publish_synced_msg)
        self.joint_names = []
        cfg_path = rospkg.RosPack().get_path('moma_bringup')
        cfg_path = os.path.join(cfg_path, '../../')

        self.moma_controller = moma_controller

        self.bridge = CvBridge()
        self.syn_cnt = 0
        self.joint_state_data, self.rgb_data, self.depth_data, self.chassis_vel, self.odometry_data, self.ee_pos\
            = None, None, None, None, None, None

    def publish_synced_msg(self, event):
        try:
            self.syn_cnt += 1
            rospy.loginfo('self.syn_cnt:    {}'.format(self.syn_cnt))
            self.joint_state_data, self.rgb_data, self.depth_data, self.chassis_vel, self.odometry_data, self.ee_pos\
                = self.moma_controller._get_state()
            
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('self.joint_state_data:      {}'.format(self.joint_state_data))
            print('self.chassis_vel:           {}'.format(self.chassis_vel))
            print('self.odometry_data:         {}'.format(self.odometry_data))
            print('self.ee_pos:                {}'.format(self.ee_pos))
            

            #@msg 1.底盘速度 
            chassis_twist = Twist()
            if self.chassis_vel is not None and len(self.chassis_vel) >= 2:
                chassis_twist.linear.x = float(self.chassis_vel[0])
                chassis_twist.angular.z = float(self.chassis_vel[1])

            #@msg 2.odometry, 底盘所在世界坐标系下的坐标
            odometry_msg = Odometry()
            if self.odometry_data is not None and len(self.odometry_data) >= 7:
                odometry_msg.header.stamp = rospy.Time.now()
                odometry_msg.header.frame_id = "odom"
                odometry_msg.pose.pose.position.x = float(self.odometry_data[0])
                odometry_msg.pose.pose.position.y = float(self.odometry_data[1]) 
                odometry_msg.pose.pose.position.z = float(self.odometry_data[2]) 
                odometry_msg.pose.pose.orientation.w = float(self.odometry_data[3])
                odometry_msg.pose.pose.orientation.x = float(self.odometry_data[4])
                odometry_msg.pose.pose.orientation.y = float(self.odometry_data[5])
                odometry_msg.pose.pose.orientation.z = float(self.odometry_data[6])

            #@msg 3.visual result, rgb, depth img.
            rgb_msg = CompressedImage()
            rgb_msg.header.stamp = rospy.Time.now()
            rgb_msg.header.frame_id = "camera"


        except Exception as e:
            rospy.logwarn("Failed to publish synced MomaState: {}".format(str(e)))