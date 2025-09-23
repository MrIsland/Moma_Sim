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

class ComputeEposeInLidarFrame:
    def __init__(self):
        self.T_lidar2base = np.array([
            [-0.964, 0.0037,  -0.2658, -0.2503],
            [-0.0047,   -1, 0.0031, -0.1156],
            [-0.2657, -0.0042, 0.964, 0.3096],
            [ 0.0,      0.0,      0.0,     1.0 ]
            ])

    def pose_to_matrix(self, ee_pos):
        x, y, z, rx, ry, rz = ee_pos
        rot = R.from_euler('xyz', [rx, ry, rz])
        T = np.eye(4)
        T[:3, :3] = rot.as_matrix()
        T[:3, 3] = [x, y, z]
        return T

    def invert_transform(self, T):
        R_inv = T[:3, :3].T
        t_inv = -R_inv @ T[:3, 3]
        T_inv = np.eye(4)
        T_inv[:3, :3] = R_inv
        T_inv[:3, 3] = t_inv
        return T_inv

    def compute(self, ee_pose):
        T_base2ee = self.pose_to_matrix(ee_pose)
        T_lidar2base = self.T_lidar2base          
        T_epos_in_lidar = T_lidar2base @ T_base2ee
        return T_epos_in_lidar

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
        self.pose_computer = ComputeEposeInLidarFrame()
        self.timer = rospy.Timer(rospy.Duration(0.005), self.publish_synced_msg)  
        self.joint_names = []
        # cfg_path = '/home/igrape/Moma_Sim_ROS/assets/config'
        cfg_path = rospkg.RosPack().get_path('moma_bringup')
        cfg_path = os.path.join(cfg_path, '../../')
        # cfg_yaml = 'assets/config/config.yaml'
        # config = load_config(os.path.join(cfg_path, cfg_yaml))
        self.moma_controller = moma_controller
        self.bridge = CvBridge()

        self.joint_state_data, self.rgb_data, self.depth_data, self.chassis_vel, self.odometry_data, self.ee_pos\
            = None, None, None, None, None, None

    def publish_synced_msg(self, event):
        try:
            self.joint_state_data, self.rgb_data, self.depth_data, self.chassis_vel, self.odometry_data, self.ee_pos\
                  = self.moma_controller._get_state()
            # print('joint_state_data:    {}'.format(joint_state_data))
            chassis_twist = Twist()
            if chassis_vel is not None and len(chassis_vel) >= 2:
                chassis_twist.linear.x = float(chassis_vel[0])
                chassis_twist.angular.z = float(chassis_vel[1])

            odometry_msg = Odometry()
            if odometry_data is not None and len(odometry_data) >= 7:
                odometry_msg.header.stamp = rospy.Time.now()
                odometry_msg.header.frame_id = "odom"
                odometry_msg.pose.pose.position.x = float(odometry_data[0])
                odometry_msg.pose.pose.position.y = float(odometry_data[1])
                odometry_msg.pose.pose.position.z = float(odometry_data[2])
                odometry_msg.pose.pose.orientation.w = float(odometry_data[3])
                odometry_msg.pose.pose.orientation.x = float(odometry_data[4])
                odometry_msg.pose.pose.orientation.y = float(odometry_data[5])
                odometry_msg.pose.pose.orientation.z = float(odometry_data[6])

            rgb_msg = CompressedImage()
            rgb_msg.header.stamp = rospy.Time.now()
            rgb_msg.header.frame_id = "camera"
            if rgb_data is not None:
                if isinstance(rgb_data, np.ndarray):
                    if len(rgb_data.shape) == 3 and rgb_data.shape[2] == 3:
                        # 转换RGB到BGR用于OpenCV
                        bgr_data = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR)
                        _, compressed_data = cv2.imencode('.jpg', bgr_data)
                        rgb_msg.format = "jpeg"
                        rgb_msg.data = compressed_data.tobytes()
                    else:
                        rgb_msg.format = "jpeg"
                        rgb_msg.data = b""
                else:
                    rgb_msg.format = "jpeg"
                    rgb_msg.data = b""
            else:
                rgb_msg.format = "jpeg"
                rgb_msg.data = b""
                
            depth_msg = Image()
            depth_msg.header.stamp = rospy.Time.now()
            depth_msg.header.frame_id = "camera"
            if depth_data is not None and isinstance(depth_data, np.ndarray):
                depth_msg.height = depth_data.shape[0]
                depth_msg.width = depth_data.shape[1]
                depth_msg.encoding = "32FC1"  
                depth_msg.is_bigendian = 0
                depth_msg.step = depth_data.shape[1] * 4  
                depth_msg.data = depth_data.astype(np.float32).tobytes()
            else:
                depth_msg.height = 480
                depth_msg.width = 640
                depth_msg.encoding = "32FC1"
                depth_msg.is_bigendian = 0
                depth_msg.step = 640 * 4
                depth_msg.data = b""

            # 创建关节状态消息
            joint_state = JointState()
            joint_state.header.stamp = rospy.Time.now()
            joint_state.header.frame_id = "base_link"
            self.joint_names = [
                'joint_1',
                'joint_2', 
                'joint_3',
                'joint_4',
                'joint_5',
                'joint_6'
            ]
            joint_state.name = self.joint_names
            if joint_state_data is not None and len(joint_state_data) >= 6:
                joint_state.position = [float(x) for x in joint_state_data]
            else:
                joint_state.position = [0.0] * 6

            # 处理末端执行器位姿
            if ee_pos is not None:
                epos_in_lidar = self.pose_computer.compute(ee_pos)

                theta = -np.pi / 2
                Rz = np.array([
                    [np.cos(theta), -np.sin(theta), 0, 0],
                    [np.sin(theta),  np.cos(theta), 0, 0],
                    [0,              0,             1, 0],
                    [0,              0,             0, 1]
                ])
                epos_in_lidar = Rz @ epos_in_lidar
                
                pose_msg = Pose()
                pose_msg.position.x = float(epos_in_lidar[0, 3])
                pose_msg.position.y = float(epos_in_lidar[1, 3])
                pose_msg.position.z = float(epos_in_lidar[2, 3])
                q = quaternion_from_matrix(epos_in_lidar)
                pose_msg.orientation.x = float(q[0])
                pose_msg.orientation.y = float(q[1])
                pose_msg.orientation.z = float(q[2])
                pose_msg.orientation.w = float(q[3])
            else:
                pose_msg = Pose()

            # 设置夹爪状态
            gripper_state_value = True
            
            # 创建主状态消息
            state_msg = moma_state()
            state_msg.header.stamp = rospy.Time.now()
            state_msg.header.frame_id = "base_link"

            state_msg.joint_state = joint_state
            state_msg.epos_in_lidar = pose_msg
            state_msg.chassis_twist = chassis_twist
            state_msg.gripper_state = Bool(gripper_state_value)
            state_msg.odometry = odometry_msg
            state_msg.rgb_image = rgb_msg
            state_msg.depth_image = depth_msg

            # 发布所有消息
            self.pub_state.publish(state_msg)
            self.pub_chassis_twist.publish(chassis_twist)
            self.pub_gripper_state.publish(Bool(gripper_state_value))
            self.pub_rgb.publish(rgb_msg)
            self.pub_depth.publish(depth_msg)
            self.pub_joint_state.publish(joint_state)
            self.pub_odometry.publish(odometry_msg)
            self.pub_epos_in_lidar.publish(pose_msg)
                                 
        except Exception as e:
            rospy.logwarn("Failed to publish synced MomaState: {}".format(str(e)))


if __name__ == '__main__':
    rospy.init_node('MomaStatePublisher', anonymous=True)

    sync_pub = SyncAndPublish()

    rospy.loginfo("[Master] Ready.")
    rospy.spin()
