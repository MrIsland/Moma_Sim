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