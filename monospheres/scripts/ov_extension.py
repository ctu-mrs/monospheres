"exec" "`dirname $0`/../python-env/bin/python" "$0" "$@"

# #{ imports

import copy
import rospy
from std_srvs.srv import Empty as EmptySrv
from std_srvs.srv import EmptyResponse as EmptySrvResponse
import threading

import heapq

import rospkg

from monospheres.common_spatial import *
from monospheres.fire_slam_module import *
from monospheres.submap_builder_module import *
from monospheres.local_navigator_module import *
from monospheres.global_navigator_module import *

from sensor_msgs.msg import Image, CompressedImage, PointCloud2, CameraInfo
import sensor_msgs.point_cloud2 as pc2
import mrs_msgs.msg
import std_msgs.msg
from scipy.spatial.transform import Rotation
import scipy
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull

import trimesh

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
from scipy.spatial import Delaunay, delaunay_plot_2d
import matplotlib.pyplot as plt
import io
import copy

from scipy.spatial.transform import Rotation
import tf
import tf2_ros
import tf2_geometry_msgs  # for transforming geometry_msgs
import tf.transformations as tfs
from geometry_msgs.msg import TransformStamped


import matplotlib.pyplot as plt
import numpy as np

import sys

# #}

# #{ global variables
STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2
kMinNumFeature = 200
kMaxNumFeature = 2000
# #}

# #{ structs and util functions
# LKPARAMS
lk_params = dict(winSize  = (31, 31),
                 maxLevel = 3,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

def featureTracking(image_ref, image_cur, px_ref, tracking_stats):
    '''
    Performs tracking and returns correspodning well tracked features (kp1 and kp2 have same size and correspond to each other)
    '''
    kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params)  #shape: [k,2] [k,1] [k,1]

    st = st.reshape(st.shape[0])
    kp1 = px_ref[st == 1]
    kp2 = kp2[st == 1]

    return kp1, kp2, tracking_stats[st == 1]

class TrackingStat:
    def __init__(self):
        self.age = 0
        self.invdepth_measurements = 0
        self.invdepth_mean = 0
        self.invdepth_sigma2 = 1
        self.prev_points = []
        self.invdepth_buffer = []

class KeyFrame:
    def __init__(self, odom_msg, img_timestamp, T_odom):
        self.triangulated_points = []
        self.odom_msg = odom_msg
        self.img_timestamp = img_timestamp
        self.T_odom = T_odom

# #}

# #{ class NavNode:
class NavNode:
    def __init__(self):# # #{
        print("INITIALIZING")
        self.node_initialized = False
        self.node_offline = False
        self.bridge = CvBridge()
        self.prev_image = None
        self.prev_time = None
        self.proper_triang = False

        self.spheremap = None
        # self.mchunk.submaps = []
        self.mchunk = CoherentSpatialMemoryChunk()

        self.keyframes = []
        self.noprior_triangulation_points = None
        self.odomprior_triangulation_points = None
        self.predicted_traj_mutex = threading.Lock()

        # SRV
        self.save_episode_full = rospy.Service("save_episode_full", EmptySrv, self.saveEpisodeFull)
        self.return_home_srv = rospy.Service("home", EmptySrv, self.return_home)

        # TIMERS
        self.submap_builder_rate = 10
        self.submap_builder_timer = rospy.Timer(rospy.Duration(1.0 / self.submap_builder_rate), self.submap_builder_update_iter)

        self.planning_frequency = 5
        self.planning_timer = rospy.Timer(rospy.Duration(1.0 / self.planning_frequency), self.planning_loop_iter)

        self.global_nav_frequency = 2
        self.global_nav_timer = rospy.Timer(rospy.Duration(1.0 / self.global_nav_frequency), self.global_nav_loop_iter)

        # VIS PUB
        self.slam_points = None

        self.path_planning_vis_pub = rospy.Publisher('path_planning_vis', MarkerArray, queue_size=10)
        self.visual_similarity_vis_pub = rospy.Publisher('visual_similarity_vis', MarkerArray, queue_size=10)
        self.unsorted_vis_pub = rospy.Publisher('unsorted_markers', MarkerArray, queue_size=10)

        self.marker_pub = rospy.Publisher('/vo_odom', Marker, queue_size=10)

        self.tf_listener = tf.TransformListener()

        # --Load calib

        self.robot_platform = rospy.get_param("platform")
        self.marker_scale = rospy.get_param("marker_scale")
        self.using_external_slam_pts = rospy.get_param("local_mapping/using_external_slam_pts")

        self.is_cam_info_undistorted = False
        if self.robot_platform == "hardnav_underwater":
            # UNITY
            self.K = np.array([642.8495341420769, 0, 400, 0, 644.5958939934509, 300, 0, 0, 1]).reshape((3,3))
            self.T_imu_to_cam = np.eye(4)
            self.T_fcu_to_imu = np.eye(4)
            self.width = 800
            self.height = 600
            self.ov_slampoints_topic = '/ov_msckf/points_slam'
            self.img_topic = '/robot1/camera1/image'
            self.odom_topic = '/ov_msckf/odomimu'
            self.imu_frame = 'imu'
            # self.fcu_frame = 'uav1/fcu'
            self.fcu_frame = 'imu'
            self.camera_frame = 'cam0'
            self.odom_frame = 'global'
        elif self.robot_platform == "bluefox_uav_gazebo":
            # BLUEFOX UAV
            self.K = np.array([227.4, 0, 376, 0, 227.4, 240, 0, 0, 1]).reshape((3,3))
            self.P = np.zeros((3,4))
            self.P[:3, :3] = self.K

            # self.distortion_coeffs: [0.019265981371039506, 0.0011428473998276235, -0.0003811659324868097, 6.340084698783884e-05]
            self.T_imu_to_cam = np.eye(4)
            self.T_fcu_to_imu = np.eye(4)
            self.width = 752
            self.height = 480
            self.ov_slampoints_topic = rospy.get_param("local_mapping/slampoints_topic")
            self.img_topic = rospy.get_param("local_mapping/img_topic")
            self.odom_topic = '/ov_msckf/odomimu'
            self.imu_frame = 'imu'
            self.fcu_frame = rospy.get_param("local_mapping/fcu_frame")
            self.camera_frame = rospy.get_param("local_mapping/camera_frame")
            self.odom_frame = rospy.get_param("local_mapping/odom_frame")
            self.is_cam_info_undistorted = True
            # self.marker_scale = 0.5

            # # Get the transform
            # self.tf_listener.waitForTransform(self.fcu_frame, self.imu_frame, rospy.Time(), rospy.Duration(4.0))
            # (trans, rotation) = self.tf_listener.lookupTransform(self.fcu_frame, self.imu_frame, rospy.Time(0))
            # rotation_matrix = tfs.quaternion_matrix(rotation)
            # print(rotation_matrix)
            # self.T_fcu_to_imu[:3, :3] = rotation_matrix[:3,:3]
            # self.T_fcu_to_imu[:3, 3] = trans
            # print("T_fcu_to_imu")
            # print(self.T_fcu_to_imu)

            # self.tf_listener.waitForTransform(self.imu_frame, self.camera_frame, rospy.Time(), rospy.Duration(4.0))
            # (trans, rotation) = self.tf_listener.lookupTransform(self.imu_frame, self.camera_frame, rospy.Time(0))
            # rotation_matrix = tfs.quaternion_matrix(rotation)
            # print(rotation_matrix)
            # self.T_imu_to_cam[:3, :3] = rotation_matrix[:3,:3]
            # self.T_imu_to_cam[:3, 3] = trans
            # print("T_imu_to_cam")
            # print(self.T_imu_to_cam)

        elif self.robot_platform == "tello":
        # TELLo (imu = fcu)
            self.K = np.array([933.5640667549508, 0.0, 500.5657553739987, 0.0, 931.5001605952165, 379.0130687255228, 0.0, 0.0, 1.0]).reshape((3,3))
            self.T_imu_to_cam = np.eye(4)
            self.T_fcu_to_imu = np.eye(4)
            self.width = 960
            self.height = 720
            self.img_topic = '/uav1/tellopy_wrapper/rgb/image_raw'
            self.odom_topic = '/uav1/estimation_manager/odom_main'

            self.imu_frame = 'imu'
            self.fcu_frame = 'uav1/fcu'
            self.odom_frame = 'uav1/passthrough_origin'
            self.camera_frame = "uav1/rgb"

            self.marker_scale = 0.15
            self.slam_kf_dist_thr = 0.5
            self.smap_fragmentation_dist = 10


        # GET IMU TO CAM TRANSFORM
        # TODO - maybe not for tello?
        print("WAITING FOR TRANSFORM BETWEEN FCU AND CAMERA")
        print(self.fcu_frame)
        print(self.camera_frame)
        self.tf_listener.waitForTransform(self.fcu_frame, self.camera_frame, rospy.Time(), rospy.Duration(4.0))
        (trans, rotation) = self.tf_listener.lookupTransform(self.fcu_frame, self.camera_frame, rospy.Time(0))
        rotation_matrix = tfs.quaternion_matrix(rotation)
        self.T_imu_to_cam[:3, :3] = rotation_matrix[:3,:3]
        self.T_imu_to_cam[:3, 3] = trans
        # self.T_imu_to_cam = np.linalg.inv(self.T_imu_to_cam)
        print("T_imu(fcu)_to_cam")
        print(self.T_imu_to_cam)

        # WAIT FOR CAMERA INFO
        self.camera_info = None
        self.camera_info_topic = rospy.get_param("local_mapping/camera_info_topic")
        self.sub_camera_info = rospy.Subscriber(self.camera_info_topic, CameraInfo, self.camera_info_callback, queue_size=10)
        print("WAITING FOR CAMERA INFO")
        while self.camera_info is None:
            print("WAITING FOR CAMERA INFO AT TOPIC " + self.camera_info_topic)
            time.sleep(0.1)

        # --INITIALIZE MODULES
        # FIRESLAM
        self.fire_slam_module = FireSLAMModule(self.width, self.height, self.K, self.camera_frame, self.odom_frame, self.tf_listener, self.camera_info)

        # SUBMAP BUILDER
        self.smap_copy = None
        self.submap_builder_input_mutex = threading.Lock()
        self.submap_builder_input_pcl = None
        self.submap_builder_input_point_ids = None

        self.submap_builder_module = SubmapBuilderModule(self.width, self.height, self.K, self.camera_frame, self.odom_frame,self.fcu_frame, self.tf_listener, self.T_imu_to_cam, self.T_fcu_to_imu, self.camera_info, self.is_cam_info_undistorted)

        # LOCAL NAVIGATOR
        ptraj_topic = rospy.get_param("local_nav/predicted_trajectory_topic")
        output_path_topic = rospy.get_param("local_nav/output_path_topic")
        self.local_navigator_module = LocalNavigatorModule(self.submap_builder_module, ptraj_topic, output_path_topic)

        # GLOBAL NAVIGATOR
        self.global_navigator_module = GlobalNavigatorModule(self.submap_builder_module, self.local_navigator_module)

        # --INIT SUBSCRIBERS
        # CAMERA
        # self.sub_cam = rospy.Subscriber(self.img_topic, Image, self.image_callback, queue_size=10)
        self.sub_cam = rospy.Subscriber(self.img_topic, CompressedImage, self.image_callback, queue_size=10)
        if self.using_external_slam_pts:
            self.sub_slam_pts = rospy.Subscriber(self.ov_slampoints_topic, PointCloud2, self.slam_points_callback, queue_size=10)

        self.odom_buffer = []
        self.odom_buffer_maxlen = 1000
        self.sub_odom = rospy.Subscriber(self.odom_topic, Odometry, self.odometry_callback, queue_size=10000)
        
        self.n_plan_iters_without_smap_update = 0
        self.n_smap_iters_since_last_planning = 0
        self.node_initialized = True
        # # #}

    # --SERVICE CALLBACKS
    def return_home(self, req):# # #{
        print("RETURN HOME SRV")
        self.state = 'home'
        return EmptySrvResponse()# # #}

    # -- MEMORY MANAGING
    def saveEpisodeFull(self, req):# # #{
        self.submap_builder_module.saveEpisodeFull(None)
        return EmptySrvResponse()# # #}

    # --SUBSCRIBE CALLBACKS

    def camera_info_callback(self, msg):
        if not self.camera_info is None:
            return
        print("OBTAINED CAMERA INFO")
        self.camera_info = msg

    def planning_loop_iter(self, event):# # #{
        print("PLANNING ITER")
        if not self.node_initialized:
            return

        # HIGH RATE OBSTACLE AVOIDANCE
        self.local_navigator_module.quick_replanning_iter(self.smap_copy)

        # SLOW LONG PLANNING
        if not self.local_navigator_module.random_explo:
            if self.n_plan_iters_without_smap_update > 1:
                print("NOT LOCAL NAVING, GIVING SMAP BUILDER CHANCE TO UPDATE!")
                return
            n_upd = 2
            if self.n_smap_iters_since_last_planning < n_upd :
                print("NOT LOCAL NAVING, GIVING SMAP BUILDER CHANCE TO DO " + str(n_upd) + "UPDATES!")
                return

        # TEST COPY
        # interm_time2 = time.time()
        # spheremap_copy = copy.deepcopy(self.submap_builder_module.spheremap)
        # copy_dt = time.time() - interm_time2
        # print("SMAP COPY DT: " + str((copy_dt) * 1000) +  " ms")

        # run with no argument to use spheremap locking, run with argument for variant using copied spheremap
        self.local_navigator_module.planning_loop_iter()
        # self.local_navigator_module.planning_loop_iter(self.smap_copy)
        self.n_plan_iters_without_smap_update += 1
        self.n_smap_iters_since_last_planning = 0
    # # #}

    def global_nav_loop_iter(self, event):
        print("GLOBAL PLANNING ITER")
        if not self.node_initialized:
            return
        self.global_navigator_module.main_iter()

    def lookupTransformAsMatrix(self, frame1, frame2):# # #{
        return lookupTransformAsMatrix(frame1, frame2, self.tf_listener)
    # # #}

    def odometry_callback(self, msg):# # #{
        if not self.node_initialized:
            return

        self.odom_buffer.append(msg)
        if len(self.odom_buffer) > self.odom_buffer_maxlen:
            self.odom_buffer.pop(0)# # #}

    def image_callback(self, msg):# # #{
        if not self.node_initialized:
            return

        self.verbose_visualization = True
        if self.verbose_visualization:
            self.submap_builder_module.save_camera_img_for_visualization(msg)

        # UPDATE VISUAL SLAM MODULE, PASS INPUT TO SUBMAP BUILDER IF NEW INPUT
        if self.using_external_slam_pts:
            return

        self.fire_slam_module.image_callback(msg)

        if self.fire_slam_module.has_new_pcl_data:
            pcl_msg, point_ids = self.fire_slam_module.get_visible_pointcloud_metric_estimate(visualize=True)
            # print("PCL MSG TYPE:")
            # print(pcl_msg)
            with ScopedLock(self.submap_builder_input_mutex):
                self.submap_builder_input_pcl = pcl_msg
                self.submap_builder_input_point_ids = point_ids
        # # #}

    def submap_builder_update_iter(self, event=None):# # #{
        if not self.node_initialized:
            return

        # copy deepcopy the input data, its not that big! then can leave mutex!! (so rest of img callback is not held up)
        pcl_msg = None
        points_info = None

        if self.submap_builder_input_pcl is None:
            print("NO PCL FOR UPDATE YET!")
            return

        with ScopedLock(self.submap_builder_input_mutex):
            pcl_msg = copy.deepcopy(self.submap_builder_input_pcl)
            # points_info = copy.deepcopy(self.submap_builder_input_point_ids)
            points_info = None

        # print("SMAP ITER")
        self.submap_builder_module.camera_update_iter(pcl_msg, points_info) 
        rospy.loginfo("copying spheremap")
        interm_time2 = time.time()
        self.smap_copy = copy.deepcopy(self.submap_builder_module.spheremap)
        copy_dt = time.time() - interm_time2
        print("SMAP COPY DT: " + str((copy_dt) * 1000) +  " ms")
        rospy.loginfo("done copying spheremap")

        # print("SMAP ITER DONE")
        self.n_plan_iters_without_smap_update = 0
        self.n_smap_iters_since_last_planning += 1
    # # #}

    def slam_points_callback(self, msg):# # #{
        if not self.node_initialized:
            return
        if self.using_external_slam_pts:
            with ScopedLock(self.submap_builder_input_mutex):
                self.submap_builder_input_pcl = msg
                self.submap_builder_input_point_ids = None
# # #}

    # --VISUALIZATIONS
    def visualize_keyframe_scores(self, scores, new_kf):# # #{
        marker_array = MarkerArray()

        new_kf_global_pos = transformPoints(new_kf.position.reshape((1,3)), self.spheremap.T_global_to_own_origin)
        s_min = np.min(scores[:(len(scores)-1)])
        s_max = np.max(scores[:(len(scores)-1)])
        scores = (scores - s_min) / (s_max-s_min)

        marker_id = 0
        for i in range(self.test_db_n_kframes - 1):
            # assuming new_kf is the latest in the scores

            smap_idx, kf_idx = self.test_db_indexing[i]
            kf_pos_in_its_map = None
            T_vis = None
            if smap_idx >= len(self.mchunk.submaps):
                kf_pos_in_its_map = self.spheremap.visual_keyframes[kf_idx].pos
                T_vis = self.spheremap.T_global_to_own_origin
            else:
                # print("IDXS")
                # print(smap_idx)
                # print(kf_idx)
                kf_pos_in_its_map = self.mchunk.submaps[smap_idx].visual_keyframes[kf_idx].pos
                T_vis = self.mchunk.submaps[smap_idx].T_global_to_own_origin
            second_kf_global_pos = transformPoints(kf_pos_in_its_map.reshape((1,3)), T_vis)

            line_marker = Marker()
            line_marker.header.frame_id = "global"  # Set your desired frame_id
            line_marker.type = Marker.LINE_LIST
            line_marker.action = Marker.ADD

            line_marker.scale.x = 1 * scores[i]
            line_marker.color.a = 1.0
            line_marker.color.r = 1.0

            line_marker.id = marker_id
            marker_id += 1

            point1 = Point() 
            point2 = Point()

            p1 = new_kf_global_pos
            p2 = second_kf_global_pos 
            
            point1.x = p1[0, 0]
            point1.y = p1[0, 1]
            point1.z = p1[0, 2]
            point2.x = p2[0, 0]
            point2.y = p2[0, 1]
            point2.z = p2[0, 2]

            line_marker.points.append(point1)
            line_marker.points.append(point2)
            marker_array.markers.append(line_marker)

        self.visual_similarity_vis_pub.publish(marker_array)
        return
    # # #}

    # --UTILS

    def get_closest_time_odom_msg(self, stamp):# # #{
        bestmsg = None
        besttimedif = 0
        for msg in self.odom_buffer:
            # print(msg.header.stamp.to_sec())
            tdif2 = np.abs((msg.header.stamp - stamp).to_nsec())
            if bestmsg == None or tdif2 < besttimedif:
                bestmsg = msg
                besttimedif = tdif2
        final_tdif = (msg.header.stamp - stamp).to_sec()
        # print("TARGET")
        # print(stamp)
        # print("FOUND")
        # print(msg.header.stamp)
        # if not bestmsg is None:
            # print("found msg with time" + str(msg.header.stamp.to_sec()) + " for time " + str(stamp.to_sec()) +" tdif: " + str(final_tdif))
        return bestmsg# # #}

# #}

if __name__ == '__main__':
    rospy.init_node('ultra_navigation_node')
    optical_flow_node = NavNode()
    rospy.spin()
    cv2.destroyAllWindows()
