#!/usr/bin/env python

# imports# # #{
import rospy
# from sensor_msgs.msg import Image
from sensor_msgs.msg import Image, CompressedImage, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from scipy.spatial.transform import Rotation
import scipy
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
# import pcl

import tf.transformations as tfs

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
from geometry_msgs.msg import TransformStamped

import matplotlib.pyplot as plt
import numpy as np

import sys

from monospheres.common_spatial import *
# # #}


class Tracked2DPoint:# # #{
    def __init__(self, pos, keyframe_id):
        self.keyframe_observations = {keyframe_id: pos}
        self.current_pos = pos
        self.age = 1
        self.last_measurement_kf_id = keyframe_id

        self.invdepth_mean = None
        self.depth = None
        self.invdist = None
        self.invdist_last_meas = None
        self.invdist_cov = None

        self.last_observed_keyframe_id = keyframe_id
        self.body_index = None
        self.active = True

        self.has_reproj = False
        self.reproj = None


    def addObservation(self, pt, keyframe_id):
        # self.last_observed_keyframe_id = np.max(keyframe_id, self.last_observed_keyframe_id)
        # self.last_observed_keyframe_id = np.max(keyframe_id, self.last_observed_keyframe_id)
        if self.last_observed_keyframe_id < keyframe_id:
            self.last_observed_keyframe_id = keyframe_id
        # self.keyframe_observations[keyframe_id] = [u,v]
        self.keyframe_observations[keyframe_id] = pt

    def getAge(self):
        return len(self.keyframe_observations)
# # #}

class KeyFrame:# # #{
    def __init__(self, img_timestamp, T_odom ):
        self.triangulated_points = []
        self.img_timestamp = img_timestamp
        self.T_odom = T_odom 
# # #}


class FireSLAMModule:
    # CORE

    def __init__(self, w, h, K, camera_frame_id, odom_orig_frame_id, tf_listener, camera_info = None, image_sub_topic=None, standalone_mode = False ):# # #{

        if not camera_info is None:
            self.width = camera_info.width
            self.height = camera_info.height
            self.cam_info_K = np.reshape(camera_info.K, (3,3))
            self.distortion_coeffs =  np.array(camera_info.D)
            # self.distortion_coeffs[0] *= 1
            # self.distortion_coeffs[1] *= 1
            # self.distortion_coeffs[2] *= 3
            # self.distortion_coeffs[3] *= 2

            # FUNKE, EQUIDISTANT
            # intrinsics: [221.01292402 220.90655859 381.87430745 227.72045365] #fu, fv, cu, cv
            # self.distortion_coeffs = np.array([-0.24006028,  0.03866184, -0.00146516,  0.00067817])
            # self.K = np.reshape(np.array([221.01292402, 0, 381.87430745,0, 220.90655859, 227.72045365, 0, 0, 1]), (3,3)) #fu, fv, cu, cv

            # CISARAK
            # K: [217.09030731481513, 0.0, 391.34523065933536, 0.0, 217.09030731481513, 220.41892147142084, 0.0, 0.0, 1.0]
            # self.distortion_coeffs =  np.array([[0.007849599795381143, -0.0428506202709645, 0.04404368320767509, -0.020526488829380334, 0.0]])
            #data: [-0.020526488829380334, 0.04404368320767509, -0.0428506202709645, 0.007849599795381143, 0.0]

            # CUSTOM
            # self.distortion_coeffs =  np.array([[-0.5, 0, 0, 0, 0.0]])
        else:
            self.width = w
            self.height = h
            self.K = K
            self.distortion_coeffs  = None


        self.camera_frame_id = camera_frame_id
        self.odom_orig_frame_id = odom_orig_frame_id
        self.image_sub_topic = image_sub_topic
        self.standalone_mode = standalone_mode
        self.has_new_pcl_data = False

        self.toonear_vis_dist = 1
        self.invdist_meas_cov = 0.002

        self.coherent_visual_odom_len = 0

        # LKPARAMS
        self.lk_params = dict(winSize  = (31, 31),
                         maxLevel = 3,
                         criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.01))

        self.bridge = CvBridge()
        self.prev_image = None
        self.prev_time = None
        self.proper_triang = False
        self.last_img_T_odom = None

        self.spheremap = None
        self.keyframes = []
        self.keyframe_idx = 0

        self.tracked_2d_points = {}
        self.next_2d_point_id = 0


        self.noprior_triangulation_points = None
        self.odomprior_triangulation_points = None

        self.active_2d_points_ids = []

        self.slam_points = None
        self.slam_pcl_pub = rospy.Publisher('extended_slam_points', PointCloud2, queue_size=10)

        self.slam_traj_vis_pub = rospy.Publisher('fire_slam_trajectory_vis', MarkerArray, queue_size=1)
        self.triang_vis_pub = rospy.Publisher('triang_features_img', Image, queue_size=1)
        self.track_vis_pub = rospy.Publisher('tracked_features_img', Image, queue_size=1)
        self.depth_pub = rospy.Publisher('estim_depth_img', Image, queue_size=1)
        self.marker_pub = rospy.Publisher('/vo_odom', Marker, queue_size=10)
        self.kp_pcl_pub = rospy.Publisher('tracked_features_space', PointCloud, queue_size=10)
        self.kp_pcl_pub_invdepth = rospy.Publisher('tracked_features_space_invdepth', Marker, queue_size=10)

        self.keyframes_marker_pub = rospy.Publisher('/keyframe_vis', MarkerArray, queue_size=10)


        self.tf_broadcaster = tf.TransformBroadcaster()

        self.orb = cv2.ORB_create(nfeatures=3000)


        # SUBSCRIBE CAM
        if self.standalone_mode:
            self.sub_cam = rospy.Subscriber(self.image_sub_topic, Image, self.image_callback, queue_size=10)

        self.imu_to_cam_T = np.array( [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0.0, 0.0, 0.0, 1.0]])
        print("IMUTOCAM", self.imu_to_cam_T)

        self.triang_vis_px1 = []
        self.triang_vis_px2 = []
        self.triang_vis_d = []
        self.triang_vis_reproj = []


        # self.camera_frame_id = "uav1/rgb"
        # self.odom_orig_frame_id = "uav1/fixed_origin"

        if self.standalone_mode:
            self.tf_listener = tf.TransformListener()
        else:
            self.tf_listener = tf_listener

        # LOAD PARAMS
        self.kf_dist_thr = rospy.get_param("local_mapping/keyframe_dist_thr")
        self.marker_scale = rospy.get_param("marker_scale")
        self.slam_filtering_enabled = rospy.get_param("local_mapping/slam_filtering_enabled")

        # NEW
        self.new_frame = None
        self.last_frame = None
        self.last_img_stamp = None
        self.new_img_stamp = None
        self.cur_R = None
        self.cur_t = None
        self.px_ref = None
        self.px_cur = None

        self.tracking_bin_width = 100
        self.min_features_per_bin = 1
        self.max_features_per_bin = 5
        # self.max_features_per_bin = 1
        self.tracking_history_len = 4
        self.node_offline = False
        self.last_tried_landmarks_pxs = None

        self.trueX, self.trueY, self.trueZ = 0, 0, 0
        self.detector = cv2.FastFeatureDetector_create(threshold=60, nonmaxSuppression=True)

        self.tracking_colors = np.random.randint(0, 255, (100, 3)) 

        self.n_frames = 0

        # # #}

    def control_features_population(self):# # #{
        wbins = self.width // self.tracking_bin_width
        hbins = self.height // self.tracking_bin_width
        found_total = 0

        # FIND PTS IN PT DICT THAT WERE SEEN IN PREVIOUS FRAME (NOT LOST SINCE PREV KEYFRAME)
        active_ids  = [pt_id for pt_id, pt in self.tracked_2d_points.items() if pt.active]

        active_pix = None

        if len(active_ids ) == 0:
            new_px = self.detector.detect(self.new_frame)
            if new_px is None or len(new_px) == 0:
                return
            n_new_px = len(new_px)
            maxtotal = wbins*hbins*self.max_features_per_bin
            keep_idxs = np.arange(n_new_px)
            if n_new_px > maxtotal:
                shuf = np.arange(n_new_px)
                np.random.shuffle(shuf)
                # new_px = new_px[shuf[:maxtotal], :]
                keep_idxs = keep_idxs[shuf[:maxtotal]]
                n_new_px = maxtotal

            print("N FOUND IN BEGINNING: " + str(n_new_px))

            active_pix = np.array([x.pt for x in new_px], dtype=np.float32)
            active_pix = active_pix[keep_idxs, :]

            # ADD NEWLY DETECTED POINTS TO THE DICT FOR TRACKING
            new_ids = range(self.next_2d_point_id, self.next_2d_point_id + n_new_px)
            self.next_2d_point_id += n_new_px
            active_ids = new_ids
            # for i in range(n_new_px):
            for i in keep_idxs:
                pt_object = Tracked2DPoint(new_px[i].pt, self.keyframe_idx)
                # self.tracked_2d_points.insert(new_ids[i], pt_object) 
                self.tracked_2d_points[new_ids[i]] = pt_object
        else:
            active_pix  = np.array([self.tracked_2d_points[pt_id].current_pos for pt_id in active_ids], dtype=np.float32)

        active_ids = np.array(active_ids)

        new_ids = []
        new_px = []
        deletion_ids = []

        n_culled = 0
        # center_deltas = active_pix - np.array([center_x, center_y])
        # active_mags = np.linalg.norm(center_deltas, axis = 1)

        # FIND THE IDXS OF THE ACTIV IDS WHICH TO DELETE, AND ACCUMULATE NEW POINT POSITIONS TO INIT
        for xx in range(wbins):
            for yy in range(hbins):
                # count how many we have there and get the points in there:
                ul = np.array([xx * self.tracking_bin_width , yy * self.tracking_bin_width])  
                lr = np.array([ul[0] + self.tracking_bin_width , ul[1] + self.tracking_bin_width]) 

                inside_bin = np.logical_and(ul <= active_pix, active_pix <= lr)
                inidx = np.all(inside_bin, axis=1)

                # print(inidx)
                inside_points = []
                inside_ids = []

                n_existing_in_bin = 0
                if np.any(inidx):
                    inside_points = active_pix[inidx]
                    inside_ids = active_ids[inidx]
                    n_existing_in_bin = inside_points.shape[0]

                if n_existing_in_bin > self.max_features_per_bin:
                    # CUTOFF POINTS ABOVE MAXIMUM, SORTED BY AGE
                    ages = np.array([-self.tracked_2d_points[pt_id].age for pt_id in inside_ids])

                    idxs = np.argsort(ages)
                    surviving_idxs = idxs[:self.max_features_per_bin]
                    n_culled_this_bin = n_existing_in_bin - self.max_features_per_bin

                    # self.px_cur = np.concatenate((self.px_cur, inside_points[surviving_idxs , :]))
                    # self.tracking_stats = np.concatenate((self.tracking_stats, inside_stats[surviving_idxs ]))

                    deletion_ids = deletion_ids + [inside_ids[i] for i in idxs[(self.max_features_per_bin-1):]]

                    # TODO - LET THE ONES WITH MANY DEPTH MEASUREMENTS LIVE

                elif n_existing_in_bin < self.min_features_per_bin:
                    # ADD THE EXISTING
                    # if n_existing_in_bin > 0:
                    #     self.px_cur = np.concatenate((self.px_cur, inside_points))
                    #     self.tracking_stats = np.concatenate((self.tracking_stats, inside_stats))

                    # FIND NEW ONES
                    locally_found = self.detector.detect(self.new_frame[ul[1] : lr[1], ul[0] : lr[0]])
                    n_found_in_bin = len(locally_found)
                    if n_found_in_bin == 0:
                        continue
                    locally_found = np.array([x.pt for x in locally_found], dtype=np.float32)

                    # be sure to not add too many!
                    n_to_add = n_found_in_bin 
                    if n_existing_in_bin + n_found_in_bin > self.max_features_per_bin:
                        n_to_add = int(self.max_features_per_bin - n_existing_in_bin)

                    shuf = np.arange(n_found_in_bin)
                    np.random.shuffle(shuf)
                    locally_found = locally_found[shuf[:n_to_add], :]

                    found_total += len(locally_found)

                    # ADD THE NEW ONES
                    locally_found[:, 0] += ul[0]
                    locally_found[:, 1] += ul[1]
                    # self.px_cur = np.concatenate((self.px_cur, locally_found))
                    new_px = new_px + [locally_found]
                else:
                    # JUST COPY THEM
                    # self.px_cur = np.concatenate((self.px_cur, inside_points))
                    # self.tracking_stats = np.concatenate((self.tracking_stats, inside_stats))
                    continue

        # NONDEACTIVATED POINTS ... ARENT CHANGED AT ALL! OH JUST INCREASED AGE (NUMBER OF IMGS LIVED)
        for px_id in active_ids:
            self.tracked_2d_points[px_id].age += 1
        for del_id in deletion_ids:
            # self.tracked_2d_points[active_ids[del_idx]].age -= 1
            # self.tracked_2d_points[active_ids[del_idx]].active = False
            self.tracked_2d_points[del_id].age -= 1
            self.tracked_2d_points[del_id].active = False


        print("DEACTIVATED: " + str(len(deletion_ids)))

        n_added = 0
        for batch in new_px:
            n_pts = batch.shape[0]
            n_added += n_pts
            # print(batch)

            new_ids = range(self.next_2d_point_id, self.next_2d_point_id + n_pts)
            self.next_2d_point_id += n_pts
            for i in range(n_pts):
                pt_object = Tracked2DPoint(batch[i, :], self.keyframe_idx)
                self.tracked_2d_points[new_ids[i]] = pt_object

        print("ADDED: " + str(n_added))

        # TODO - add the found points to dict!!!

        # print("CURRENT FEATURES: " + str(self.px_cur.shape[0]))

        # # FIND FEATS IF ZERO!
        # if(self.px_cur.shape[0] == 0):
        #     print("ZERO FEATURES! FINDING FEATURES")
        #     self.px_cur = self.detector.detect(self.new_frame)
        #     self.px_cur = np.array([x.pt for x in self.px_cur], dtype=np.float32)
    # # #}

    def image_callback(self, msg):# # #{
        if self.node_offline:
            return
        self.has_new_pcl_data  = False

        # GET CURRENT ODOM POSE
        (trans, rotation) = self.tf_listener.lookupTransform(self.odom_orig_frame_id , self.camera_frame_id, rospy.Time(0))
        rotation_matrix = tfs.quaternion_matrix(rotation)
        T_odom = np.eye(4)
        T_odom[:3, :3] = rotation_matrix[:3,:3]
        T_odom[:3, 3] = trans
        # print("CUR ODOM:")
        # print(trans)

        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")


        # new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (1280, 720), np.eye(3), balance=1, new_size=(3400, 1912), fov_scale=1)
        # map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (3400, 1912), cv2.CV_16SC2)
        # dst = cv2.remap(img, map1[575:1295, 23:3119, :], map2[575:1295, 23:3119], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        D = self.distortion_coeffs[:4]
        print(D.shape)
        wh = (self.width,self.height)
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(self.cam_info_K, D, wh, np.eye(3), balance=1, new_size=wh, fov_scale=1)
        self.K = new_K
        print("NEW K")
        print(new_K)
        self.P = np.zeros((3,4))
        self.P[:3, :3] = self.K

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(self.cam_info_K, D, np.eye(3), new_K, wh, cv2.CV_16SC2)
        dst = cv2.remap(img, map1[:, :, :], map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        img = dst


        # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.K, self.distortion_coeffs, (self.width,self.height), 1, (self.width,self.height))
        # dst = cv2.undistort(img, self.K, self.distortion_coeffs, None, newcameramtx)
        # print("DISTORTION")
        # print(img.shape)
        # print(roi)
        # # crop the image
        # x, y, w, h = roi
        # print(dst.shape)
        # # dst = dst[y:y+h, x:x+w]
        # # print(dst.shape)
        # # cv.imwrite('calibresult.png', dst)

        # self.track_vis_pub.publish(self.bridge.cv2_to_imgmsg(dst, "bgr8"))

        comp_start_time = rospy.get_rostime()

        # SHIFT LAST AND NEW
        self.last_img_stamp = self.new_img_stamp 
        self.new_img_stamp  = msg.header.stamp

        self.last_frame = self.new_frame
        self.new_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # self.new_frame = cv2.normalize(img, None, 0, 1.0,
# cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        self.px_ref = self.px_cur

        comp_start_time = time.time()

        # RETURN IF FIRST FRAME
        if self.n_frames == 0:
            # FIND FIRST FEATURES
            self.n_frames = 1
            # self.last_img_stamp = stamp
            return
        self.n_frames += 1


        # IF YOU CAN - TRACK
        self.active_2d_points_ids = [pt_id for pt_id, pt in self.tracked_2d_points.items() if pt.active]
        # print("N_ACTIV 2D POINTS: " + str(len(self.active_2d_points_ids)))
        if len(self.active_2d_points_ids) > 0:
            # print("BEFORE TRACKING: " + str(self.px_ref.shape[0]))

            # DO TRACKING
            px_ref = np.array([self.tracked_2d_points[p].current_pos for p in self.active_2d_points_ids], dtype=np.float32)
            # print("PX REF SHAPE")
            # print(px_ref.shape)
            # print(px_ref)
            kp2, st, err = cv2.calcOpticalFlowPyrLK(self.last_frame, self.new_frame, px_ref, None, **self.lk_params)  #shape: [k,2] [k,1] [k,1]
            st = st.reshape(st.shape[0])

            # FILTER TOO FAR AWAY FROM CENTER (FOR FISHEYE)
            center_x = self.width/2
            center_y = self.height * 0.7
            center_deltas = kp2 - np.array([center_x, center_y])
            center_dists = np.linalg.norm(center_deltas, axis=1)
            max_center_dist = self.width * 0.23
            st[center_dists > max_center_dist] = 0
            # kp1 = px_ref[st == 1]
            # kp2 = kp2[st == 1]

            # ADD OBSERVATION POSITIONS AND THIS_FRAME INDEX FOR ALL 2D POINTS IN 2D POINT DICT
            for i in range(px_ref.shape[0]):
                pt_id = self.active_2d_points_ids[i]
                if st[i] == 1:
                    self.tracked_2d_points[pt_id].addObservation(kp2[i], self.keyframe_idx)
                    self.tracked_2d_points[pt_id].current_pos = kp2[i]
                else:
                    self.tracked_2d_points[pt_id].active = False

        # TODO - REMOVE 2D POINTS THAT HAVE NOT BEEN OBSERVED FOR N FRAMES

        keyframe_time_threshold = 0.1
        # keyframe_distance_threshold = 4
        keyframe_distance_threshold = self.kf_dist_thr

        time_since_last_keyframe = None
        dist_since_last_keyframe = None

        # CHECK IF SHOULD ADD KEYFRAME FOR TRIANGULATION OF POINTS
        # TODO - and prev keyframe parallax condition
        timedist_to_img = (rospy.get_rostime() - self.new_img_stamp).to_sec()
        # print("TIMEDIST IMG: " + str(timedist_to_img))
        if len(self.keyframes) > 0:
            time_since_last_keyframe = (self.new_img_stamp - self.keyframes[-1].img_timestamp).to_sec()
        # print("TIME SINCE LAST KF:" + str(time_since_last_keyframe ))



        if not time_since_last_keyframe is None :
            dist_since_last_keyframe = np.linalg.norm((np.linalg.inv(self.keyframes[self.keyframe_idx-1].T_odom) @ T_odom)[:3, 3])
        if time_since_last_keyframe is None or (time_since_last_keyframe > keyframe_time_threshold and dist_since_last_keyframe > keyframe_distance_threshold):
            print("ATTEMPTING TO ADD NEW KEYFRAME! " + str(len(self.keyframes)))

            new_kf = KeyFrame(self.new_img_stamp, T_odom)

            # FOR ALL PTS THAT ARE BEING TRACKED:
            #     IF CAN TRIANGULATE (parallax + motion since last depth meas time) - TRY TRIANGULATION IF OK, ADD MEAS! 
            self.active_2d_points_ids = [pt_id for pt_id, pt in self.tracked_2d_points.items() if pt.active]
            # min_parallax = 30

            self.triang_vis_px1 = []
            self.triang_vis_px2 = []
            self.triang_vis_d = []
            self.triang_vis_reproj = []

            n_ok_meas = 0

            ransacable_ids = [pt_id for pt_id, pt in self.tracked_2d_points.items() if pt.active and pt.age > 1]
            n_ransac = len(ransacable_ids)
            # print("N RANSACABLE: " + str(n_ransac))
            if n_ransac > 4 and self.keyframe_idx > 0:
                kfi = self.keyframe_idx-1
                dst_pts = np.array([[self.tracked_2d_points[i].current_pos[0], self.tracked_2d_points[i].current_pos[1]] for i in ransacable_ids])
                src_pts = np.array([[self.tracked_2d_points[i].keyframe_observations[kfi][0], self.tracked_2d_points[i].keyframe_observations[kfi][1]] for i in ransacable_ids])
                # M, mask = cv2.findEssentialMat(src_pts, dst_pts, self.K, threshold=1, prob=0.9)
                M, mask = cv2.findEssentialMat(src_pts, dst_pts, self.K, threshold=1, prob=0.99)
                mask = mask.flatten() == 1
                inlier_pt_ids = np.array(ransacable_ids)[mask]

                decomp_res = self.decomp_essential_mat(M, src_pts[mask, :], dst_pts[mask, :])
                T_ransac = np.linalg.inv(self._form_transf(decomp_res[0], decomp_res[1]))

                T_odom_prev = self.keyframes[self.keyframe_idx - 1].T_odom
                T_delta_odom = (np.linalg.inv(T_odom_prev) @  T_odom)
                scaling_factor_simple = np.linalg.norm(T_delta_odom[:3,3]) / np.linalg.norm(T_ransac[:3,3])


                tracked_3d_pts_mask = np.array([not self.tracked_2d_points[i].invdist is None for i in inlier_pt_ids])
                if not np.any(tracked_3d_pts_mask):
                    self.coherent_visual_odom_len = 0
                    print("LOST TRACKING 3D!!")


                mean_inv_depth = np.mean(np.reciprocal(np.linalg.norm(self.triangulated_points1, axis=1)))

                # T_ransac[:3, 3] = T_ransac[:3, 3] * mean_inv_depth 
                # self.triangulated_points1 = self.triangulated_points1 * mean_inv_depth 
                # self.triangulated_points2 = transformPoints(self.triangulated_points1, T_ransac)
                # mean_inv_depth = np.mean(np.reciprocal(np.linalg.norm(self.triangulated_points1, axis=1)))

                print("MEAN INV DEPTH:")
                print(mean_inv_depth)

                # SCALING FACTOR IS BETWEEN UNSCALED VISUAL ODOM AND SCALED METRIC ODOM ESTIMATE
                if self.coherent_visual_odom_len > 0:
                    # which_of_newly_triangulated_have_previous_measurements

                    cur_kf_pts_mask = np.array([(i in self.keyframes[self.keyframe_idx - 1].inlier_pt_ids) for i in inlier_pt_ids])
                    tracked_3d_pts_cur_kf = self.triangulated_points1[cur_kf_pts_mask, :]

                    prev_kf_pts_mask = np.array([(i in inlier_pt_ids) for i in self.keyframes[self.keyframe_idx - 1].inlier_pt_ids])
                    tracked_3d_pts_prev_kf = self.keyframes[self.keyframe_idx - 1].triangulated_points2[prev_kf_pts_mask, :]

                    print("SHAPEZ")
                    print(tracked_3d_pts_prev_kf.shape)
                    print(tracked_3d_pts_cur_kf.shape)

                    # sum_dists_prev_kf = np.sum(np.linalg.norm(tracked_3d_pts_prev_kf, axis=1))
                    # sum_dists_cur_kf = np.sum(np.linalg.norm(tracked_3d_pts_cur_kf, axis=1))

                    # sum_dists_prev_kf = np.mean(np.linalg.norm(tracked_3d_pts_prev_kf, axis=1))
                    # sum_dists_cur_kf = np.mean(np.linalg.norm(tracked_3d_pts_cur_kf, axis=1))

                    # sum_dists_prev_kf = np.min((np.linalg.norm(tracked_3d_pts_prev_kf, axis=1)))
                    # sum_dists_cur_kf = np.min((np.linalg.norm(tracked_3d_pts_cur_kf, axis=1)))

                    sum_dists_prev_kf = np.sum(np.reciprocal(np.linalg.norm(tracked_3d_pts_prev_kf, axis=1)))
                    sum_dists_cur_kf = np.sum(np.reciprocal(np.linalg.norm(tracked_3d_pts_cur_kf, axis=1)))

                    # sum_dists_prev_kf = np.mean(np.reciprocal(np.linalg.norm(tracked_3d_pts_prev_kf, axis=1)))
                    # sum_dists_cur_kf = np.mean(np.reciprocal(np.linalg.norm(tracked_3d_pts_cur_kf, axis=1)))

#                     print("SUMDIST_PREV:")
#                     print(sum_dists_prev_kf)
#                     print("SUMDIST_CUR:")
#                     print(sum_dists_cur_kf)

#                     # scaling_factor_cur_kf = sum_dists_prev_kf / sum_dists_cur_kf  
#                     scaling_factor_cur_kf = sum_dists_cur_kf/ sum_dists_prev_kf   
#                     print("FACTOR:")
#                     print(scaling_factor_cur_kf)
#                     T_ransac[:3, 3] = T_ransac[:3, 3] * scaling_factor_cur_kf
#                     self.triangulated_points1 = self.triangulated_points1 * scaling_factor_cur_kf
#                     self.triangulated_points2 = transformPoints(self.triangulated_points1, T_ransac)

                new_kf.T_visual_odom = T_ransac
                new_kf.triangulated_points2 = self.triangulated_points2
                new_kf.inlier_pt_ids = inlier_pt_ids
                self.coherent_visual_odom_len += 1

                if self.coherent_visual_odom_len > 1:
                    self.coherent_visual_odom_len = 1

                # SCALING FACTOR OF CURRENT VISUAL ODOM TO SCALED ODOM
                reversed_metric_traj_rel_to_current_odom = []
                reversed_unscaled_traj_rel_to_current_odom = []

                T_delta_vis = T_ransac

                poses_metric = [np.eye(4)]
                poses_unscaled = [np.eye(4)]
                len_metric = np.linalg.norm(T_delta_odom[:3,3])
                len_unscaled = np.linalg.norm(T_delta_vis[:3,3])
                T_odom_next_pt = T_odom_prev

                poses_metric.append(poses_metric[-1] @ np.linalg.inv(T_delta_odom))
                poses_unscaled.append(poses_unscaled[-1] @ np.linalg.inv(T_delta_vis ))


                for i in range(self.coherent_visual_odom_len-1):
                    # TODO fix indexing going back, maybe go forward? - DRAW!!!
                    kfi = self.keyframe_idx - i - 1


                    T_odom_cur_pt = self.keyframes[kfi].T_odom
                    T_delta_odom = np.linalg.inv(T_odom_cur_pt) @ T_odom_next_pt
                    T_delta_vis = self.keyframes[kfi].T_visual_odom
                    T_odom_next_pt = T_odom_cur_pt

                    # poses_metric.append(np.linalg.inv(T_delta_odom) @ poses_metric[-1])
                    # poses_unscaled.append(np.linalg.inv(T_delta_ransac) @ poses_unscaled[-1])

                    poses_metric.append(poses_metric[-1] @ np.linalg.inv(T_delta_odom))
                    poses_unscaled.append(poses_unscaled[-1] @ np.linalg.inv(T_delta_vis ))

                    len_metric += np.linalg.norm(T_delta_odom[:3,3])
                    len_unscaled += np.linalg.norm(T_delta_vis [:3,3])


                print("LEN METRIC: " + str(len_metric) + ", UNSCALED: " + str(len_unscaled))


                scaling_factor = (len_metric / len_unscaled)
                # scaling_factor = scaling_factor_simple

                print("COHERENCE LEN: " + str(self.coherent_visual_odom_len) + ", FACTOR: " + str(scaling_factor) + ", SIMPLE FACTOR: " + str(scaling_factor_simple))
                # self.triangulated_points = copy.deepcopy(self.triangulated_points2) * scaling_factor

                # TODO visualize the scaled visually tracked odom alongside the metric odom, HERE!
                marker_array = MarkerArray()
                self.get_pose_array_markers(poses_metric, self.odom_orig_frame_id, marker_array)
                self.get_pose_array_markers(poses_unscaled, self.odom_orig_frame_id, marker_array, scaling_factor, r=0, g=0, b=1)
                self.slam_traj_vis_pub.publish(marker_array)


                reproj_pts = transformPoints(copy.deepcopy(src_pts), M)
                mask_idxs = np.where(mask)[0]
                inlier_idx = 0 
                for i in range(n_ransac):
                    if mask[i]:
                        dist_f1 = np.linalg.norm(self.triangulated_points1[inlier_idx, :] * scaling_factor)
                        dist_f2 = np.linalg.norm(self.triangulated_points2[inlier_idx, :] * scaling_factor)

                        self.tracked_2d_points[ransacable_ids[i]].last_measurement_kf_id = self.keyframe_idx
                        self.tracked_2d_points[ransacable_ids[i]].depth = dist_f2
                        invdist_meas = 1.0 / dist_f2
                        invdist_estimate = self.tracked_2d_points[ransacable_ids[i]].invdist_last_meas
                        # self.tracked_2d_points[ransacable_ids[i]].invdist_last_meas  = meas

                        # DO BAYESIAN FUSION OF MEASUREMENTS!!!
                        if invdist_estimate is None or not self.slam_filtering_enabled:
                        # if True:
                            self.tracked_2d_points[ransacable_ids[i]].invdist_last_meas  = invdist_meas 
                            self.tracked_2d_points[ransacable_ids[i]].invdist_cov = self.invdist_meas_cov * 3
                        else:
                            last_cov = self.tracked_2d_points[ransacable_ids[i]].invdist_cov

                            # PROPAGATE
                            dist_estimate_f1 = 1 / invdist_estimate 
                            dist_estimate_f2 = dist_estimate_f1 + (dist_f2 - dist_f1)
                            invdist_estimate_propagated = 1 / dist_estimate_f2

                            fused_meas = (invdist_estimate_propagated * self.invdist_meas_cov + invdist_meas * last_cov) / (last_cov + self.invdist_meas_cov)
                            fused_cov = (last_cov * self.invdist_meas_cov) / (last_cov + self.invdist_meas_cov)

                            self.tracked_2d_points[ransacable_ids[i]].invdist_last_meas = fused_meas
                            self.tracked_2d_points[ransacable_ids[i]].invdist_cov = fused_cov

                        self.tracked_2d_points[ransacable_ids[i]].invdist = self.tracked_2d_points[ransacable_ids[i]].invdist_last_meas


                        inlier_idx += 1 

                    else:
                        self.tracked_2d_points[ransacable_ids[i]].has_reproj = False

            print("N OK MEAS: " + str(n_ok_meas) + "/" + str(len(self.active_2d_points_ids)))

            # VISUALIZE BY PROJECTING PTS THRU CAMERA WITH ESTIMATED DEPTH!!!

            # CONTROL FEATURE POPULATION - ADDING AND PRUNING
            self.control_features_population()

            # RETURN IF STILL CANT FIND ANY, NOT ADDING KEYFRAME
            # if(self.px_cur is None):
            if len(self.active_2d_points_ids) < 4:
                print("--WARNING! NOT ENOUGH FEATURES FOUND EVEN AFTER POPULATION CONTROL! NOT ADDING KEYFRAME! NUM TRACKED PTS: " + str(len(self.active_2d_points_ids)))
                return

            # HAVE ENOUGH POINTS, ADD KEYFRAME
            print("ADDED NEW KEYFRAME! KF: " + str(len(self.keyframes)))


            # ADD SLAM POINTS INTO THIS KEYFRAME (submap)
            new_kf.slam_points = self.slam_points
            self.slam_points = None

            self.keyframes.append(new_kf)
            self.keyframe_idx += 1

            # STORE THE PIXELPOSITIONS OF ALL CURRENT POINTS FOR THIS GIVEN KEYFRAME 
            # for i in range(self.px_cur.shape[0]):
            #     self.tracking_stats[i].prev_keyframe_pixel_pos = self.px_cur[i, :]

            if self.standalone_mode:
                self.get_visible_pointcloud_metric_estimate(visualize=True)

            if self.coherent_visual_odom_len > 0:
                self.has_new_pcl_data = True

            comp_time = time.time() - comp_start_time
            print("computation time: " + str((comp_time) * 1000) +  " ms")
            self.track_vis_pub.publish(self.bridge.cv2_to_imgmsg(self.visualize_tracking(), "bgr8"))

        # VISUALIZE FEATURES

        # self.triang_vis_pub.publish(self.bridge.cv2_to_imgmsg(self.visualize_triang(), "bgr8"))
        self.keyframes_marker_pub.publish(self.visualize_keyframes())

# # #}

    def get_visible_pointcloud_metric_estimate(self, visualize = True):# # #{
        point_cloud = PointCloud2()
        # point_cloud.header.frame_id = 'uav1/fcu'  # Set the frame ID according to your robot's configuration
        # point_cloud.header.frame_id = self.camera_frame_id  # Set the frame ID according to your robot's configuration

        point_cloud.header.stamp = rospy.Time.now()
        point_cloud.header.frame_id = self.odom_orig_frame_id  # Set the frame ID according to your robot's configuration

        (trans, rotation) = self.tf_listener.lookupTransform(self.odom_orig_frame_id, self.camera_frame_id, rospy.Time(0))
        rotation_matrix = tfs.quaternion_matrix(rotation)
        T_odom = np.eye(4)
        T_odom[:3, :3] = rotation_matrix[:3,:3]
        T_odom[:3, 3] = trans

        pts = []
        line_pts = []

        line_marker = Marker()
        line_marker.header.frame_id = self.odom_orig_frame_id  # Set your desired frame_id
        line_marker.type = Marker.LINE_LIST
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.08 * self.marker_scale  # Line width
        line_marker.color.a = 1.0  # Alpha
        line_marker.color.r = 1.0  
        line_marker.color.g = 1.0  

        cloud_pts = []

        self.active_2d_points_ids = [pt_id for pt_id, pt in self.tracked_2d_points.items() if pt.active]
        sent_pt_ids = []

        for pt_id in self.active_2d_points_ids:
            cur_pos = self.tracked_2d_points[pt_id].current_pos
            if not self.tracked_2d_points[pt_id].invdist is None:

                kfi = self.tracked_2d_points[pt_id].last_measurement_kf_id  
                if kfi != self.keyframe_idx - 1:
                    continue

                sent_pt_ids.append(pt_id)
                T_odom_pt = self.keyframes[kfi ].T_odom
                obsv_pos = self.tracked_2d_points[pt_id].keyframe_observations[kfi]

                # T_odom_pt = T_odom
                # obsv_pos = cur_pos

                invdist = self.tracked_2d_points[pt_id].invdist
                d = 1.0 / invdist
                dir1 = np.array([obsv_pos[0], obsv_pos[1], 1]).reshape(3,1)

                invK = np.linalg.inv(self.K)
                dir1 = invK @ dir1
                dir1 = (dir1 / np.linalg.norm(dir1)).flatten() 


                # TRANSFORM POINT BY FRAME OF CAM AT LAST OBSERVED POSE
                # dir1 = transformPoints(dir1.reshape((1,3)), self.keyframes[last_observed_keyframe_id].T_odom).flatten()

                pt_mean = d * copy.deepcopy(dir1)
                t_pt_mean = transformPoints(pt_mean.reshape((1,3)), T_odom_pt).flatten()
                # pts.append([t_pt_mean[0], t_pt_mean[1], t_pt_mean[2]])
                cloud_pts.append([t_pt_mean[0], t_pt_mean[1], t_pt_mean[2]])
                
                # COVARIANCE VISUALIZING
                if visualize:
                    cov = self.tracked_2d_points[pt_id].invdist_cov
                    conf_interval = 1
                    idist_min = invdist - conf_interval * np.sqrt(cov)
                    idist_max = invdist + conf_interval * np.sqrt(cov)
                    if idist_min <= 0:
                        idist_min = invdist * 0.5

                    dist_est_max = (1.0/idist_min)
                    dist_est_min = (1.0/idist_max)
                    if dist_est_min < d*0.5:
                        dist_est_min = d*0.5
                    if dist_est_max > d*1.5:
                        dist_est_max = d*1.5

                    maxpt = dist_est_min * dir1
                    minpt = dist_est_max * dir1

                    t_minpt = transformPoints(minpt.reshape((1,3)), T_odom_pt).flatten()
                    line_pts.append([t_minpt[0], t_minpt[1], t_minpt[2]])

                    t_maxpt = transformPoints(maxpt.reshape((1,3)), T_odom_pt).flatten()
                    line_pts.append([t_maxpt[0], t_maxpt[1], t_maxpt[2]])


        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.odom_orig_frame_id  # Set the frame ID according to your robot's configuration

        fields = [PointField('x', 0, PointField.FLOAT32, 1),
          PointField('y', 4, PointField.FLOAT32, 1),
          PointField('z', 8, PointField.FLOAT32, 1),
          # PointField('rgb', 12, PointField.UINT32, 1),
          # PointField('rgba', 12, PointField.UINT32, 1),
          ]
        cloud = pc2.create_cloud(header, fields, cloud_pts)


        for i in range(len(cloud_pts)):
            # LINE
            point1 = Point()
            point2 = Point()

            p1 = line_pts[i*2]
            p2 = line_pts[i*2 + 1]
            
            point1.x = p1[0]
            point1.y = p1[1]
            point1.z = p1[2]
            point2.x = p2[0]
            point2.y = p2[1]
            point2.z = p2[2]
            line_marker.points.append(point1)
            line_marker.points.append(point2)

        if visualize:
            self.slam_pcl_pub.publish(cloud)
            self.kp_pcl_pub_invdepth.publish(line_marker)

        return cloud, np.array(sent_pt_ids)
    # # #}

    # VISUALIZATION

    def visualize_triang(self):# # #{
        # rgb = np.zeros((self.new_frame.shape[0], self.new_frame.shape[1], 3), dtype=np.uint8)
        # print(self.new_frame.shape)
        rgb = np.repeat(copy.deepcopy(self.new_frame)[:, :, np.newaxis], 3, axis=2)
        # rgb = np.repeat((self.new_frame)[:, :, np.newaxis], 3, axis=2)

        px_cur = np.array([self.tracked_2d_points[p].current_pos for p in self.active_2d_points_ids])

        n_pts = len(self.triang_vis_px1)
        if n_pts  > 0:
            depths = np.array(self.triang_vis_d)

            # min_d = np.min(depths)
            # max_d = np.max(depths)
            min_d = 0.2
            max_d = 5

            minsize = 3
            maxsize = 12

            for i in range(n_pts):
                # size = self.tracking_stats[inidx][i].age / growsize
                # if size > growsize:
                #     size = growsize
                # size += minsize

                # size = minsize
                rel_d = (self.triang_vis_d[i] - min_d) / (max_d - min_d)
                if rel_d < 0:
                    rel_d = 0
                elif rel_d > 1:
                    rel_d = 1

                size = minsize + (maxsize - minsize) * (1 - rel_d)


                # rgb = cv2.line(rgb, (int(self.triang_vis_px1[i][0]), int(self.triang_vis_px1[i][1])), (int(self.triang_vis_reproj[i][0]), int(self.triang_vis_reproj[i][1])), (0, 0, 0), 3)
                # rgb = cv2.line(rgb, (int(self.triang_vis_px2[i][0]), int(self.triang_vis_px2[i][1])), (int(self.triang_vis_reproj[i][0]), int(self.triang_vis_reproj[i][1])), (0, 0, 0), 3)
                rgb = cv2.line(rgb, (int(self.triang_vis_px2[i][0]), int(self.triang_vis_px2[i][1])), (int(self.triang_vis_px1[i][0]), int(self.triang_vis_px1[i][1])), (0, 0, 0), 3)

                rgb = cv2.circle(rgb, (int(self.triang_vis_px1[i][0]), int(self.triang_vis_px1[i][1])), int(size), 
                               (255, 0, 0), -1) 

                rgb = cv2.circle(rgb, (int(self.triang_vis_px2[i][0]), int(self.triang_vis_px2[i][1])), int(size), 
                               (0, 255, 0), -1) 

                # rgb = cv2.circle(rgb, (int(self.triang_vis_reproj[i][0]), int(self.triang_vis_reproj[i][1])), int(size), 
                #                (255, 0, 255), -1) 


        res = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return res
# # #}
    
    def visualize_ransac_reproj(self):# # #{
        rgb = np.repeat(copy.deepcopy(self.new_frame)[:, :, np.newaxis], 3, axis=2)
        # ransacd = [pt_id for pt_id, pt in self.tracked_2d_points.items() if pt.has_reproj]

        px_cur = np.array([self.tracked_2d_points[p].current_pos for p in self.active_2d_points_ids])
        have_reproj = np.array([(self.tracked_2d_points[p].has_reproj ) for p in self.active_2d_points_ids])
        reprojs = np.array([(self.tracked_2d_points[p].reproj ) for p in self.active_2d_points_ids])
        n_active = len(self.active_2d_points_ids)

        growsize = 7
        minsize = 7

        for i in range(n_active):
            # size = self.tracking_stats[inidx][i].age / growsize
            # if size > growsize:
            #     size = growsize
            # size += minsize
            size = minsize
            color = (200,0,0)
            px = int(px_cur[i][0])
            py = int(px_cur[i][1])
            if have_reproj[i]:
                color = (255,0,255)
                size = minsize + 5
                rpx = int(reprojs[i][0])
                rpy = int(reprojs[i][1])


                rgb = cv2.circle(rgb, (rpx, rpy), int(size), 
                           (0,255,0), -1) 
                rgb = cv2.line(rgb, (rpx, rpy), (px, py), (0, 0, 0), 3)

            rgb = cv2.circle(rgb, (px, py), int(size), 
                       color, -1) 


        res = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return res
# # #}

    def visualize_tracking(self):# # #{
        # rgb = np.zeros((self.new_frame.shape[0], self.new_frame.shape[1], 3), dtype=np.uint8)
        # print(self.new_frame.shape)
        rgb = np.repeat(copy.deepcopy(self.new_frame)[:, :, np.newaxis], 3, axis=2)
        # rgb = np.repeat((self.new_frame)[:, :, np.newaxis], 3, axis=2)

        px_cur = np.array([self.tracked_2d_points[p].current_pos for p in self.active_2d_points_ids])
        # have_depth = np.array([(not self.tracked_2d_points[p].depth is None) for p in self.active_2d_points_ids])
        have_depth = np.array([(self.tracked_2d_points[p].last_measurement_kf_id == self.keyframe_idx - 1) for p in self.active_2d_points_ids])
        depths = np.array([self.tracked_2d_points[p].depth for p in self.active_2d_points_ids])

        if not px_cur is None and px_cur.size > 0:

            ll = np.array([0, 0])  # lower-left
            ur = np.array([self.width, self.height])  # upper-right
            inidx = np.all(np.logical_and(ll <= px_cur, px_cur <= ur), axis=1)
            inside_pix_idxs = px_cur[inidx].astype(int)

            growsize = 7
            minsize = 7

            for i in range(inside_pix_idxs.shape[0]):
                # size = self.tracking_stats[inidx][i].age / growsize
                # if size > growsize:
                #     size = growsize
                # size += minsize
                size = minsize
                color = (0,0,0)
                if have_depth[i]:
                    color = (255,0,255)
                    # if depths[i] < self.toonear_vis_dist:
                    #     color = (255,0,0)
                    #     size += 3

                rgb = cv2.circle(rgb, (inside_pix_idxs[i,0], inside_pix_idxs[i,1]), int(size), 
                               color, -1) 

        res = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return res
    # # #}

    def publish_pose_msg(self):# # #{
        tf_msg = TransformStamped()
        tf_msg.header.stamp = rospy.Time.now()
        tf_msg.header.frame_id = "mission_origin"
        tf_msg.child_frame_id = "cam_odom"

        # Set the translation 
        tf_msg.transform.translation.x = self.cur_t[0]
        tf_msg.transform.translation.y = self.cur_t[1]
        tf_msg.transform.translation.z = self.cur_t[2]

        # Set the rotation 
        r = Rotation.from_matrix(self.cur_R)
        quat = r.as_quat()
        # quat = pose.rotation().toQuaternion()
        tf_msg.transform.rotation.x = quat[0]
        tf_msg.transform.rotation.y = quat[1]
        tf_msg.transform.rotation.z = quat[2]
        tf_msg.transform.rotation.w = quat[3]


        # Broadcast the TF transform
        self.tf_broadcaster.sendTransformMessage(tf_msg)
    # # #}

    def visualize_keypoints(self, img, kp):# # #{
        rgb = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        for k in kp:
            rgb[int(k.pt[1]), int(k.pt[0]), 0] = 255
        flow_vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return flow_vis
    # # #}

    def get_pose_array_markers(self, poses, frame_id, marker_array, scaling_factor = 1,r=1, g=0, b=0):# # #{
        marker_id = 0
        if len(marker_array.markers) > 0:
            marker_id = len(marker_array.markers)
        ms = self.marker_scale * 8

        T_vis = lookupTransformAsMatrix(frame_id, self.camera_frame_id, self.tf_listener)

        for i in range(len(poses)):
            marker = Marker()
            marker.header.frame_id = frame_id  # Change this frame_id if necessary
            marker.header.stamp = rospy.Time.now()
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.id = marker_id
            marker_id += 1

            # Set the scale
            marker.scale.x = ms *0.1
            marker.scale.y = ms *0.2
            marker.scale.z = ms *0.2

            marker.color.a = 1
            marker.color.r = r
            marker.color.g = g
            marker.color.b = b

            arrowlen = 0.5 * ms

            endpose = copy.deepcopy(poses[i])
            endpose[:3,3] = endpose[:3,3] * scaling_factor
            endpose = T_vis @ endpose

            pt1 = endpose[:3, 3] 
            pt2 = pt1 + endpose[:3, 2] * arrowlen

            points_msg = [Point(x=pt1[0], y=pt1[1], z=pt1[2]), Point(x=pt2[0], y=pt2[1], z=pt2[2])]
            marker.points = points_msg

            marker_array.markers.append(marker)
    # # #}

    def visualize_keyframes(self):# # #{
        marker_array = MarkerArray()
        marker_id = 0
        ms = 1

        for i in range(len(self.keyframes)):
            marker = Marker()
            marker.header.frame_id = self.odom_orig_frame_id  # Change this frame_id if necessary
            marker.header.stamp = rospy.Time.now()
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.id = marker_id
            marker_id += 1

            # Set the scale
            marker.scale.x = ms *0.1
            marker.scale.y = ms *0.2
            marker.scale.z = ms *0.2

            marker.color.a = 1
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0

            arrowlen = 1 * ms
            # xbonus = arrowlen * np.cos(smap.visual_keyframes[i].heading + map_heading)
            # ybonus = arrowlen * np.sin(smap.visual_keyframes[i].heading + map_heading)
            pt1 = self.keyframes[i].T_odom[:3, 3]
            pt2 = pt1 + self.keyframes[i].T_odom[:3, 2]

            points_msg = [Point(x=pt1[0], y=pt1[1], z=pt1[2]), Point(x=pt2[0], y=pt2[1], z=pt2[2])]
            marker.points = points_msg

            marker_array.markers.append(marker)

        return marker_array
    # # #}

    def visualize_keyframes(self):# # #{
        marker_array = MarkerArray()
        marker_id = 0
        ms = 1

        for i in range(len(self.keyframes)):
            marker = Marker()
            marker.header.frame_id = self.odom_orig_frame_id  # Change this frame_id if necessary
            marker.header.stamp = rospy.Time.now()
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.id = marker_id
            marker_id += 1

            # Set the scale
            marker.scale.x = ms *0.1
            marker.scale.y = ms *0.2
            marker.scale.z = ms *0.2

            marker.color.a = 1
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0

            arrowlen = 1 * ms
            # xbonus = arrowlen * np.cos(smap.visual_keyframes[i].heading + map_heading)
            # ybonus = arrowlen * np.sin(smap.visual_keyframes[i].heading + map_heading)
            pt1 = self.keyframes[i].T_odom[:3, 3]
            pt2 = pt1 + self.keyframes[i].T_odom[:3, 2]

            points_msg = [Point(x=pt1[0], y=pt1[1], z=pt1[2]), Point(x=pt2[0], y=pt2[1], z=pt2[2])]
            marker.points = points_msg

            marker_array.markers.append(marker)

        return marker_array
    # # #}

    # UTILS

    def triangulateDepthLineNearestPoint(self, T_cam, pos1, pos2, visualize=False):# # #{
        # TRANSFORM TO RAY DIRECTIONS IN 3D

        if visualize:
            self.triang_vis_px1.append(pos1)
            self.triang_vis_px2.append(pos2)

        # pos1 = copy.deepcopy(pos1)
        # pos2 = copy.deepcopy(pos2)
        # pos1[0] = self.width - pos1[0]
        # pos1[1] = self.height - pos1[1]
        # pos2[0] = self.width - pos2[0]
        # pos2[1] = self.height - pos2[1]

        dir1 = np.array([pos1[0], pos1[1], 1]).reshape(3,1)
        dir2 = np.array([pos2[0], pos2[1], 1]).reshape(3,1)

        invK = np.linalg.inv(self.K)
        # invK = self.K
        dir1 = invK @ dir1
        dir2 = invK @ dir2


        dir1 = (dir1 / np.linalg.norm(dir1)).flatten() 
        dir2 = (dir2 / np.linalg.norm(dir2)).flatten()

        linepos2 = T_cam[:3,3]

        # ROTATE DIR2 BY THE T_CAM
        # dir2 = np.linalg.inv(T_cam)[:3,:3] @ dir2
        dir2 = T_cam[:3,:3] @ dir2

        angle_thresh = 0.05
        dirdot = (dir1).dot(dir2)
        angle = np.arccos(dirdot)
        print("ANGLE:")
        print(angle)
        if angle <= angle_thresh:
            print("DIRVECS TOO PARALLEL! " + str(dirdot) + "/" + str(angle_thresh))
            self.triang_vis_d.append(-1)
            self.triang_vis_reproj.append(pos2)
            return None
        # TODO perpendicular motion threshold

        crossprod = np.cross(dir1, dir2)
        crossprod_normalized = crossprod / np.linalg.norm(crossprod)
        # print("CROSSPROD_N:")
        # print(crossprod_normalized)

        dist_lines = np.abs(linepos2.dot(crossprod_normalized))
        print("DIST LINES:")
        print(dist_lines)

        # vec_along_line1 = (dir2-dir1) - pricka_vec
        # dist_pt_on_line2 = ((dist_lines - linepos2.dot(dir1)) / (dir2.dot(dir1)))

        dist_pt_on_line2 = ((np.cross(dir1, crossprod)).dot(linepos2)) / (crossprod.dot(crossprod))
        print("DEPTH: ")
        print(dist_pt_on_line2 )

        # TEST:
        pt_on_line_2 = linepos2 + dir2 * dist_pt_on_line2
        pt_on_line_1 = pt_on_line_2  - crossprod_normalized * dist_lines

        # TRY REPROJECTING PT1 TO CAM1
        reproj1 = self.K @ pt_on_line_1
        reproj1 = reproj1[:2] / reproj1[2]

        res_pt = np.ones((4))
        res_pt[:3] = (pt_on_line_2 + pt_on_line_1) / 2
        # res_pt[:3] = pt_on_line_2
        res_pt = (np.linalg.inv(T_cam) @ res_pt)[:3]

        reproj2 = self.K @ (res_pt)
        reproj2 = reproj2[:2] / reproj2[2]

        # reproj2 = reproj2[:2] 
        reproj_error = np.linalg.norm(reproj2 - pos2)

        # depth = np.linalg.norm(res_pt)
        depth = dist_pt_on_line2
        if visualize:
            self.triang_vis_d.append(depth)
            self.triang_vis_reproj.append(reproj2)

        return depth# # #}

    def decomp_essential_mat(self, E, q1, q2):# # #{
        """
        Decompose the Essential matrix

        Parameters
        ----------
        E (ndarray): Essential matrix
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image

        Returns
        -------
        right_pair (list): Contains the rotation matrix and translation vector
        """
        def sum_z_cal_relative_scale(R, t, store=False):
            # Get the transformation matrix
            T = self._form_transf(R, t)
            # Make the projection matrix
            P = np.matmul(np.concatenate((self.K, np.zeros((3, 1))), axis=1), T)

            # Triangulate the 3D points
            hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
            # Also seen from cam 2
            hom_Q2 = np.matmul(T, hom_Q1)

            # Un-homogenize
            uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

            # if store:
            #     relative_scale = np.mean(np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1)/
            #                             np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1))
            #     print("REL SCALE:")
            #     print(relative_scale)
            if store:
                return uhom_Q1.T, uhom_Q2.T

            # Find the number of points there has positive z coordinate in both cameras
            sum_of_pos_z_Q1 = sum(uhom_Q1[2, :] > 0)
            sum_of_pos_z_Q2 = sum(uhom_Q2[2, :] > 0)

            # Form point pairs and calculate the relative scale
            relative_scale = np.mean(np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1)/
                                     np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1))
            # print("R, t:")
            # print(R)
            # print(t)
            # print(sum_of_pos_z_Q1)
            # print(sum_of_pos_z_Q2)
            return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale

        # Decompose the essential matrix
        R1, R2, t = cv2.decomposeEssentialMat(E)
        t = np.squeeze(t)

        # Make a list of the different possible pairs
        pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]

        # Check which solution there is the right one
        z_sums = []
        relative_scales = []
        for R, t in pairs:
            z_sum, scale = sum_z_cal_relative_scale(R, t)
            z_sums.append(z_sum)
            relative_scales.append(scale)

        # Select the pair there has the most points with positive z coordinate
        right_pair_idx = np.argmax(z_sums)
        right_pair = pairs[right_pair_idx]
        relative_scale = relative_scales[right_pair_idx]
        R1, t = right_pair

        # store the unhomogenized keypoints for the correct pair
        self.triangulated_points1, self.triangulated_points2 = sum_z_cal_relative_scale(R1, t, True)

        t = t * relative_scale

        return [R1, t]
    # # #}
    
    @staticmethod# # #{
    def _form_transf(R, t):
        """
        Makes a transformation matrix from the given rotation matrix and translation vector

        Parameters
        ----------
        R (ndarray): The rotation matrix
        t (list): The translation vector

        Returns
        -------
        T (ndarray): The transformation matrix
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
    # # #}

