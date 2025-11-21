import copy# # #{
import rospy
from std_srvs.srv import Empty as EmptySrv
from std_srvs.srv import EmptyResponse as EmptySrvResponse
import threading

import heapq

import rospkg

from monospheres.common_spatial import *

from sensor_msgs.msg import Image, CompressedImage, PointCloud2
import sensor_msgs.point_cloud2 as pc2
import mrs_msgs.msg
import std_msgs.msg
from scipy.spatial.transform import Rotation
import scipy
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
from shapely import geometry

import pyembree
import trimesh

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
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

from numba import njit, jit


import matplotlib.pyplot as plt
import numpy as np

import sys
# # #}

class SubmapBuilderModule:
    def __init__(self, w, h, K, camera_frame_id, odom_orig_frame_id, fcu_frame, tf_listener, T_imu_to_cam, T_fcu_to_imu, camera_info = None, is_cam_info_undistorted = False):# # #{
        self.latest_camera_image = None

        self.do_distbased_fusion = True

        # default values for getting from master (TODO - put into parameters or launch files)
        self.width = w
        self.height = h
        self.K = K
        self.new_K = K

        if not camera_info is None:
            if not is_cam_info_undistorted:
                self.width = camera_info.width
                self.height = camera_info.height
                self.cam_info_K = np.reshape(camera_info.K, (3,3))
                self.distortion_coeffs =  np.array(camera_info.D)[:4]

                self.new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(self.cam_info_K, self.distortion_coeffs, (w, h), np.eye(3), balance=1, new_size=(w, h), fov_scale=1)

        # TODO - something like " if self.using_disorted_camera and not self.should_undistort"
        self.distortion_coeffs = np.array([0.019265981371039506, 0.0011428473998276235, -0.0003811659324868097, 6.340084698783884e-05])
        # self.distortion_coeffs = np.array([0.015,-0.020,0.017,-0.003])
        # self.distortion_coeffs = np.array([0.037,-0.029,0.019,-0.004])

        # self.dist_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(self.K, self.distortion_coeffs, (w, h), np.eye(3), balance=1, new_size=(w, h), fov_scale=1)

        self.camera_frame = camera_frame_id
        self.odom_frame = odom_orig_frame_id
        self.fcu_frame = fcu_frame
        self.tf_listener = tf_listener
        self.T_imu_to_cam = T_imu_to_cam
        self.T_fcu_to_imu = T_fcu_to_imu 

        self.do_frontiers = True

        self.bridge = CvBridge()
        self.prev_image = None
        self.prev_time = None
        self.proper_triang = False

        self.spheremap = None
        self.potential_fr_sample_pts = None
        # self.mchunk.submaps = []

        self.mchunk = CoherentSpatialMemoryChunk()

        # self.testing_mchunk_filename = rospy.get_param("global_nav/testing_mchunk_filename")
        # mchunk_filepath = rospkg.RosPack().get_path('monospheres') + "/memories/" + self.testing_mchunk_filename
        # self.mchunk = CoherentSpatialMemoryChunk.load(mchunk_filepath)

        self.keyframes = []
        self.noprior_triangulation_points = None
        self.odomprior_triangulation_points = None
        self.spheremap_mutex = threading.Lock()
        self.predicted_traj_mutex = threading.Lock()

        # SRV
        # self.vocab_srv = rospy.Service("save_vocabulary", EmptySrv, self.saveCurrentVisualDatabaseToVocabFile)
        # self.save_episode_full = rospy.Service("save_episode_full", EmptySrv, self.saveEpisodeFull)
        # self.return_home_srv = rospy.Service("home", EmptySrv, self.return_home)

        # DATA PUB
        # vis_prefix = "AAA/"
        vis_prefix = ""
        self.runtimes_pub = rospy.Publisher(vis_prefix + 'spheremap_runtimes', mrs_msgs.msg.Float64ArrayStamped, queue_size=10)

        # VIS PUB
        self.assumed_freespace_vis_pub = rospy.Publisher(vis_prefix + 'assumed_freespace_points', MarkerArray, queue_size=10)
        self.spheremap_outline_pub = rospy.Publisher(vis_prefix + 'spheres', MarkerArray, queue_size=10)
        self.spheremap_freespace_pub = rospy.Publisher(vis_prefix + 'spheremap_freespace', MarkerArray, queue_size=10)
        self.freespace_polyhedron_pub = rospy.Publisher(vis_prefix + 'visible_freespace_poly', MarkerArray, queue_size=10)

        self.recent_submaps_vis_pub = rospy.Publisher(vis_prefix + 'recent_submaps_vis', MarkerArray, queue_size=10)
        self.path_planning_vis_pub = rospy.Publisher(vis_prefix + 'path_planning_vis', MarkerArray, queue_size=10)
        self.visual_similarity_vis_pub = rospy.Publisher(vis_prefix + 'visual_similarity_vis', MarkerArray, queue_size=10)
        self.unsorted_vis_pub = rospy.Publisher(vis_prefix + 'unsorted_markers', MarkerArray, queue_size=10)

        self.frontier_samples_pub = rospy.Publisher(vis_prefix + 'frontier_samples', PointCloud2, queue_size=10)

        # self.kp_pub = rospy.Publisher('tracked_features_img', Image, queue_size=1)
        self.polygon_debug_pub = rospy.Publisher('polygon_debug', Image, queue_size=1)
        self.polygon_enhanced_debug_pub = rospy.Publisher('polygon_enhanced_debug', Image, queue_size=1)
        self.marker_pub = rospy.Publisher('/vo_odom', Marker, queue_size=10)


        print("T_imu(fcu)_to_cam")
        print(self.T_imu_to_cam)


        # LOAD PARAMS
        self.min_sphere_rad = rospy.get_param("local_mapping/min_sphere_rad")
        self.surfel_resolution = rospy.get_param("local_mapping/surfel_resolution")
        self.frontier_resolution = rospy.get_param("local_mapping/frontier_resolution")
        self.protection_constant = rospy.get_param("local_mapping/protection_constant")

        self.verbose_submap_construction = rospy.get_param("local_mapping/verbose_submap_construction")
        self.carryover_dist = rospy.get_param("local_mapping/carryover_dist")
        self.uav_radius = rospy.get_param("local_nav/uav_radius")
        self.min_planning_odist = rospy.get_param("local_nav/min_planning_odist")

        self.marker_scale = rospy.get_param("marker_scale")
        self.n_sphere_samples_per_update = rospy.get_param("local_mapping/n_sphere_samples_per_update")
        self.fragmenting_travel_dist = rospy.get_param("local_mapping/smap_fragmentation_dist")
        self.max_sphere_update_dist = rospy.get_param("local_mapping/max_sphere_sampling_z")
        self.visual_kf_addition_heading = 3.14159 /2
        self.visual_kf_addition_dist = 2

        self.ofs_freespace_pts_enabled = rospy.get_param("local_mapping/ofs_pts_enabled")
        self.ff_points_always_active = rospy.get_param("local_mapping/ofs_pts_always_active")
        self.ff_points_ignore_tracking = rospy.get_param("local_mapping/ofs_ignore_tracking")


        # LOAD ofs FREESPACE POINTS
        self.ff_dist = rospy.get_param("local_mapping/ofs_pts_dist")
        self.ff_min_parallax = rospy.get_param("local_mapping/ofs_min_parallax")
        self.ff_pts_W = 5
        self.ff_pts_H = 5
        self.n_frontier_samples = 300
        # self.n_frontier_samples = 3000
        self.init_ofs_freespace_pts();
        self.ff_pose_buffer = []
        self.max_ff_buffer_len = rospy.get_param("local_mapping/ofs_tracking_buffer_len")
        
        # # #}

    # CORE

    # @jit
    def camera_update_iter(self, msg, slam_ids = None):# # #{
        runtime_data = []

        runtime_polyhedron_update = 0
        runtime_old_spheres_update = 0
        runtime_new_spheres_update = 0
        runtime_obstacles_and_frontiers_update = 0
        runtime_connectivity_update = 0

        self.ofs_freespace_update()

        # Add new visual keyframe if enough distance has been traveled# # #{
        with ScopedLock(self.spheremap_mutex):
            if self.verbose_submap_construction:
                print("PCL MSG")

            if self.verbose_submap_construction:
                print("PCL MSG PROCESSING NOW")
            update_start_time = time.time()
            interm_time = time.time()

            # CHECK TRAVELED DIST
            if not self.spheremap is None:
                T_global_to_fcu = lookupTransformAsMatrix(self.odom_frame, self.fcu_frame, self.tf_listener)
                T_fcu_relative_to_smap_start  = np.linalg.inv(self.spheremap.T_global_to_own_origin) @ T_global_to_fcu

                new_kf = SubmapKeyframe(T_fcu_relative_to_smap_start)


                # CHECK IF NEW ENOUGH
                # TODO - check in near certainly connected submaps
                tooclose = False
                for kf in self.spheremap.visual_keyframes:
                    # TODO - scaling
                    if kf.euclid_dist(new_kf) < self.visual_kf_addition_dist and kf.heading_dif(new_kf) < self.visual_kf_addition_heading:
                        tooclose = True
                        break

                if len(self.spheremap.visual_keyframes) > 0:
                    dist_bonus = new_kf.euclid_dist(self.spheremap.visual_keyframes[-1])
                    heading_bonus = new_kf.heading_dif(self.spheremap.visual_keyframes[-1]) * 0.2

                    self.spheremap.traveled_context_distance += dist_bonus + heading_bonus

                # print("KFS: adding new visual keyframe!")
                self.spheremap.visual_keyframes.append(new_kf)
        # # #}

            # Read Points from PCL msg# #{
            pc_data = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            pc_list = []
            for point in pc_data:
                x, y, z = point
                pc_list.append([x, y, z])
            if len(pc_list) == 0:
                return
            point_cloud_array = np.array(pc_list, dtype=np.float32)

            # OpenVINS outputs always some point at origin, and it is not really useful for us, breaks hull stuff
            #TODO - solve what to do with this in future
            nonzero_pts = np.array(np.linalg.norm(point_cloud_array, axis=1) > 0)
            if not np.any(nonzero_pts):
                if self.verbose_submap_construction:
                    print("NO POINTS?")
                return
            point_cloud_array = point_cloud_array[nonzero_pts, :]

            # TODO - remove
            # noise_intensity = 1
            # point_cloud_array = point_cloud_array + np.random.uniform(-noise_intensity, noise_intensity , size=point_cloud_array.shape)

            # print("PCL ARRAY SHAPE:")
            # print(point_cloud_array.shape)
            # # #}

            # DECIDE WHETHER TO UPDATE SPHEREMAP OR INIT NEW ONE# #{
            T_global_to_fcu = lookupTransformAsMatrix(self.odom_frame, self.fcu_frame, self.tf_listener)

            self.submap_unpredictability_signal = 0
            init_new_spheremap = False
            memorized_transform_to_prev_map = None

            init_rad = self.uav_radius
            min_rad = init_rad

            if self.spheremap is None:
                init_new_spheremap = True
            else:
                if self.submap_unpredictability_signal > 1:
                    pass
                else:
                    max_split_score = 100
                    split_score = 0
                    secs_since_creation = (rospy.get_rostime() - self.spheremap.creation_time).to_sec()

                    # TIME BASED
                    # if secs_since_creation > 70:
                    #     split_score = max_split_score

                    # CONTEXT DIST BASED
                    if self.spheremap.traveled_context_distance > self.fragmenting_travel_dist:
                        split_score = max_split_score
                    else:
                        if self.verbose_submap_construction:
                            print("TRAVELED CONTEXT DISTS: " + str(self.spheremap.traveled_context_distance))

                    if split_score >= max_split_score:
                        # SAVE OLD SPHEREMAP
                        print("SPLITTING SUBMAP!")
                        # TODO give it the transform to the next ones origin from previous origin
                        memorized_transform_to_prev_map = np.linalg.inv(self.spheremap.T_global_to_own_origin) @ T_global_to_fcu

                        # self.mchunk.submaps.append(self.spheremap)
                        self.mchunk.addSubmap(self.spheremap)
                        init_new_spheremap = True
                        self.visualize_episode_submaps()
            if init_new_spheremap:
                # INIT NEW SPHEREMAP
                print("INFO: initing new spheremap")

                pts_to_transfer = None
                radii_to_transfer = None

                # CREATE CONNECTION FROM AND TO THE PREV SPHEREMAP IF PREV MAP EXISTS!
                connection_to_prev_map = None
                if (not self.spheremap is None) and (not memorized_transform_to_prev_map is None):
                    print("CREATING CONNECTION TO PREV MAP")
                    pos_in_old_frame = memorized_transform_to_prev_map[:3,3].reshape((1,3))
                    pos_in_new_frame = np.zeros((1,3))
                    odist = self.spheremap.getMaxDistToFreespaceEdge(pos_in_old_frame)
                    print("ODIST: " + str(odist))

                    if odist < self.uav_radius:
                        odist = self.uav_radius
                        print("WARN! ODIST SMALLER THAN UAV RADIUS!")

                    self.spheremap.map2map_conns.append(MapToMapConnection(pos_in_old_frame, len(self.mchunk.submaps), odist))
                    connection_to_prev_map = MapToMapConnection(pos_in_new_frame, len(self.mchunk.submaps)-1, odist)
                    # else:
                    #     # print("ODIST NOT SAFE!!! NOT ADDING CONNECTION!")
                    #     TODO - 
                    # old_map_connection_pt = 

                    # GET SOME CLOSE SPHERES
                    if not self.spheremap.spheres_kdtree is None:
                        qres = self.spheremap.spheres_kdtree.query(pos_in_old_frame, k=10, distance_upper_bound=self.carryover_dist)
                        print("CARRYOVER:")
                        idxs = qres[1][0]
                        existing_mask = idxs != self.spheremap.points.shape[0]
                        print(np.sum(existing_mask ))
                        existing_mask = np.where(existing_mask)[0]
                        if np.any(existing_mask):
                            pts_to_transfer = self.spheremap.points[existing_mask, :]
                            radii_to_transfer = self.spheremap.radii[existing_mask]
                            pts_to_transfer = transformPoints(pts_to_transfer, np.linalg.inv(memorized_transform_to_prev_map))


                self.spheremap = SphereMap(init_rad, self.min_sphere_rad)
                self.spheremap.surfels_filtering_radius = 0.2


                if not connection_to_prev_map is None:
                    self.spheremap.map2map_conns.append(connection_to_prev_map)

                self.spheremap.memorized_transform_to_prev_map = memorized_transform_to_prev_map 
                print("PESSS")
                self.spheremap.T_global_to_own_origin = T_global_to_fcu
                self.spheremap.creation_time = rospy.get_rostime()

                if not pts_to_transfer is None:
                    self.spheremap.points = copy.deepcopy(pts_to_transfer)
                    self.spheremap.radii = copy.deepcopy(radii_to_transfer)
                    self.spheremap.connections = np.array([None for c in range(pts_to_transfer.shape[0])], dtype=object)
                    self.spheremap.updateConnections(np.arange(0, pts_to_transfer.shape[0]))
                    self.spheremap.labelSpheresByConnectivity()



                if len(self.mchunk.submaps) > 2:
                    self.saveEpisodeFull(None)
            # # #}
            
            start_comps_dt = (time.time() - interm_time)
            interm_time = time.time()
            if self.verbose_submap_construction:
                print("START COMPS TIME: " + str(start_comps_dt))

            # COMPUTE VISIBLE FREESPACE MESH # #{
            # TRANSFORM SLAM PTS TO -----CAMERA FRAME---- AND COMPUTE THEIR PIXPOSITIONS
            T_global_to_cam = T_global_to_fcu @ self.T_fcu_to_imu @ self.T_imu_to_cam

            transformed = transformPoints(point_cloud_array, np.linalg.inv(T_global_to_cam)).T

            positive_z_mask = transformed[2, :] > 0
            positive_z_points = transformed[:, positive_z_mask]

            if not np.any(positive_z_mask):
                print("WARN - no points coming from SLAM have positive Z in camera frame. ENding update")
                return

            pts_for_surfel_addition_tmp = positive_z_points  #SAVE JUST THE MEASURED ONES FOR SURFEL ADDITION

            positive_z_slam_ids = None
            if not slam_ids is None:
                positive_z_slam_ids = slam_ids[positive_z_mask]





            visible_obstacle_pts = positive_z_points
            print("zpoints shape: ")
            print(positive_z_points.shape)
            pixpos = getPixelPositions(visible_obstacle_pts, self.K, self.distortion_coeffs)
            # print("OBSTACLE MESH PROJECTED PIXPOS:")
            # print(pixpos)

            # COMPUTE DELAUNAY TRIANG OF VISIBLE SLAM POINTS
            if visible_obstacle_pts.shape[1] < 4:
                if self.verbose_submap_construction:
                    print("NOT ENAUGH PTS FOR DELAUNAY!")
                return
            if self.verbose_submap_construction:
                print("HAVE DELAUNAY:")
            tri = Delaunay(pixpos)
            self.polygon_debug_pub.publish(self.bridge.cv2_to_imgmsg(self.visualize_polygon(pixpos, tri), "bgr8"))


            # CONSTRUCT POLYGON OF PIXPOSs OF VISIBLE SLAM PTS
            hull2d = ConvexHull(pixpos)

            # print("OBSTACLE POLYGON HULL:")
            # print(hull2d.points)

            if self.verbose_submap_construction:
                print("POLY")
            visible_obstacle_pts_polygon = geometry.Polygon(pixpos[hull2d.vertices, :]) # MUST BE ORDERED PROPERLY!!!
            extended_pts_polygon = visible_obstacle_pts_polygon 
            hull2d_idxs = hull2d.vertices

            # CONSTRUCT OBSTACLE MESH
            extended_obstacle_mesh = trimesh.Trimesh(vertices=positive_z_points.T, faces = tri.simplices)

            # CONSTRUCT FOV MESH (of visible pts and current cam origin)
            # BY DEFAULT THE MESH PTS ARE JUS TTHE VISIBLE PTS (and origin added later)
            fullmesh_pts = positive_z_points

            # GET ACTIVATED AND USABLE OFS FREESPACE PTS
            added_ff_pts = None
            if np.any(self.ff_points_active_mask) and self.ofs_freespace_pts_enabled:
                # positive_z_points  = np.concatenate((positive_z_points, self.egocentric_ff_pts[self.ff_points_active_mask, :].T), axis = 1)
                pixpos_ff_pts = getPixelPositions(self.egocentric_ff_pts.T, self.K, self.distortion_coeffs)
                inhull = np.array([visible_obstacle_pts_polygon.contains(geometry.Point(pixpos_ff_pts[i, 0], pixpos_ff_pts[i, 1])) for i in range(pixpos_ff_pts.shape[0])])

                outside_hull_mask = np.logical_not(inhull)

                addition_mask = np.logical_and(self.ff_points_active_mask, outside_hull_mask)
                added_ff_pts = self.egocentric_ff_pts[addition_mask, :].T

                # print("PIXPOS ACTIVE AND OUTSIDE HULL:")
                # print(pixpos_ff_pts[addition_mask, :])

                self.ofs_pixpos_lastframe = pixpos_ff_pts[addition_mask, :]

            # ADD THEM IF SOME CAN BE ADDED
            if not added_ff_pts is None:
                # print("ADDING " + str(added_ff_pts.shape[1]) + " OFS FREESPACE PTS THAT DO NOT LIE IN CURRENT VISIBLE PT POLYGON")
                fullmesh_pts = np.concatenate((fullmesh_pts, added_ff_pts), axis = 1)
                # RECOMPUTE HULL 
                pixpos = getPixelPositions(fullmesh_pts, self.K, self.distortion_coeffs)
                tri = Delaunay(pixpos)
                # CONSTRUCT POLYGON OF PIXPOSs OF VISIBLE SLAM PTS
                hull2d = ConvexHull(pixpos)
                hull2d_idxs = hull2d.vertices
                extended_pts_polygon = geometry.Polygon(pixpos[hull2d.vertices, :]) # MUST BE ORDERED PROPERLY!!!

                # CONSTRUCT OBSTACLE MESH
                extended_obstacle_mesh = trimesh.Trimesh(vertices=fullmesh_pts.T, faces = tri.simplices)


                # self.polygon_enhanced_debug_pub.publish(self.bridge.cv2_to_imgmsg(self.visualize_polygon(pixpos, tri, True), "bgr8"))


            # ADD ORIGIN PT
            orig_pts = np.zeros((3,1))
            fullmesh_pts = np.concatenate((fullmesh_pts, orig_pts), axis=1)
            fullmesh_zero_pt_index = fullmesh_pts.shape[1] - 1
            new_simplices = [[hull2d_idxs[i], hull2d_idxs[i+1], fullmesh_zero_pt_index] for i in range(len(hull2d_idxs) - 1)]
            new_simplices.append([hull2d_idxs[len(hull2d_idxs)-1], hull2d_idxs[0], fullmesh_zero_pt_index])
            fullmesh_simplices = np.concatenate((tri.simplices, new_simplices), axis=0)

            fov_mesh = trimesh.Trimesh(vertices=fullmesh_pts.T, faces = fullmesh_simplices, use_embree = True)
            fov_mesh_query = trimesh.proximity.ProximityQuery(fov_mesh)

            # # #}

            meshing_dt = time.time() - interm_time
            interm_time = time.time()
            if self.verbose_submap_construction:
                print("POLYHEDRON CONSTRUCTION time: " + str((meshing_dt) * 1000) +  " ms")
            runtime_data.append(meshing_dt)
            runtime_polyhedron_update = meshing_dt

            # SAMPLE NEW FRONTIERS ALONG THE VISIBLE FREESPACE MESH# #{
            fr_samples = None
            if self.do_frontiers:
                # print("FRONTIERS SAMPLING")
                fr_samples, fr_face_indices = trimesh.sample.sample_surface_even(fov_mesh, int(self.n_frontier_samples / 2))
                # ADD NOISE
                if self.spheremap.frontier_deleting_inside_dist > 0.1:
                    sigma = self.spheremap.frontier_deleting_inside_dist * 2
                    fr_samples = fr_samples  + np.random.normal(0, sigma, size=fr_samples.shape)

                # fr_samples2, fr_face_indices2 = trimesh.sample.sample_surface_even(fov_mesh, 3)
                # fr_samples = np.concatenate((fr_samples, fr_samples2))


                # Sample some also around nearby spheres

                # fr_samples2, fr_face_indices2 = trimesh.sample.sample_surface_even(obstacle_mesh, 50)
                # print(fr_samples.shape)
                # fr_samples = fr_samples[fr_sample_odists > -0.1, :]
                # print(fr_samples.shape)

            # VISUALIZE VISIBLE FREESPACE POLYHDRON
            polyhedron_markers = self.get_freespace_polyhedron_markers(self.spheremap.T_global_to_own_origin, T_global_to_fcu, fullmesh_pts.T, fullmesh_simplices)
            self.freespace_polyhedron_pub.publish(polyhedron_markers) 

            frontiers_dt = time.time() - interm_time
            interm_time = time.time()
            interm_time2 = time.time()
            if self.verbose_submap_construction:
                print("FRONTIER SAMPLING time: " + str((frontiers_dt) * 1000) +  " ms")
            runtime_data.append(frontiers_dt)
            runtime_obstacles_and_frontiers_update += frontiers_dt
            # # #}

            # UPDATE OLD SPHERES# #{
            T_orig_to_current_cam = np.eye(4)
            T_delta_odom  = np.eye(4)
            old_pts = None
            old_radii = None
            if not self.spheremap is None: # TODO remove
                # transform existing sphere points to current camera frame
                n_spheres_old = self.spheremap.points.shape[0]

                T_delta_odom = np.linalg.inv(self.spheremap.T_global_to_own_origin) @ T_global_to_fcu
                T_orig_to_current_cam = ( T_delta_odom @ self.T_fcu_to_imu @ self.T_imu_to_cam)

                # project sphere points to current camera frame
                transformed_old_points  = transformPoints(self.spheremap.points, np.linalg.inv(T_orig_to_current_cam))

                # GET UPDATABLE SPHERES 
                # CONSTRUCT BBX
                bbx = BoundingBox3D()
                bbx.pos = np.zeros((1,3))
                bbx.axes = np.eye(3)
                # GET MIN AND MAX BY MIN AND MAX POLYHEDRON PTS + MAX RADIUS
                bbx_min = np.min(fullmesh_pts.T, axis=0).reshape((1,3))
                bbx_max = np.max(fullmesh_pts.T, axis=0).reshape((1,3))
                bbx.minmaxvals = np.concatenate((bbx_min, bbx_max), axis = 0)
                bbx_for_surfel_deletion = copy.deepcopy(bbx)
                bbx.expand(self.spheremap.max_radius)

                # USE BBX TO GET SPHERES THAT COULD BE UPDATED 
                # max_vis_z = np.max(positive_z_points[2, :])
                # z_ok_mask = np.logical_and(transformed_old_points[:, 2] > 0, transformed_old_points[:, 2] <= max_vis_z)
                z_ok_mask = bbx.pts_in_mask(transformed_old_points)

                # EXPAND AGAIN TO GET ALL SPHERES THAT COULD CONNECT TO THE MODIFIED SPHERES
                bbx.expand(self.spheremap.max_radius)
                sphere_idxs_for_conn_checking = bbx.pts_in_mask(transformed_old_points)


                z_ok_points = transformed_old_points[z_ok_mask , :] # remove spheres with negative z
                old_pts = z_ok_points
                old_radii = copy.deepcopy(self.spheremap.radii[z_ok_mask])
                worked_sphere_idxs = np.arange(n_spheres_old)[z_ok_mask ]


                # check the ones that are projected into the 2D hull
                old_pixpos = getPixelPositions(z_ok_points.T, self.K)

                # CHECK OLD SURFEL PTS IF THEY FALL INTO CURRENT FULL MESH
                # test_pts = np.array([[0, 0, 1 * i] for i in range(20)])
                # # testdists = fov_mesh_query.signed_distance(test_pts)
                # print("TESTDISTS:")
                # # print(testdists)
                # print(fov_mesh.contains(test_pts))
                nondeleted_visible_surfels = None

                # old_update_dt = time.time() - interm_time2
                # interm_time2 = time.time()
                # if self.verbose_submap_construction:
                #     print("PART -1: " + str((old_update_dt ) * 1000) +  " ms")

                if not self.spheremap.surfel_points is None:
                    # TODO - here, filter out the ones outside of bbx!
                    n_surfels = self.spheremap.surfel_points.shape[0]
                    # print(n_surfels)

                    surfel_points_in_camframe = transformPoints(self.spheremap.surfel_points, np.linalg.inv(T_orig_to_current_cam))

                    surfels_in_bbx_mask = bbx_for_surfel_deletion.pts_in_mask(surfel_points_in_camframe) 
                    surfels_in_bbx_idxs = np.where(surfels_in_bbx_mask)[0] 

                    #FURTHER FILTER OUT

                    # - by distance from camera and max sphere size
                    toofar = surfel_points_in_camframe[surfels_in_bbx_mask, 2] > self.max_sphere_update_dist + self.spheremap.max_radius

                    print("n too far: " + str(np.sum(toofar)))
                    surfels_in_bbx_mask[surfels_in_bbx_mask] = np.logical_not(toofar)
                    surfels_in_bbx_idxs = np.where(surfels_in_bbx_mask)[0] 


                    # - by being in 2D delaunay hull or not
                    pixpos = getPixelPositions(surfel_points_in_camframe[surfels_in_bbx_idxs, :].T, self.K)
                    inhull = tri.find_simplex(pixpos)>=0
                    surfels_in_bbx_mask[surfels_in_bbx_mask] = inhull
                    print("n not inhull: " + str(np.sum(np.logical_not(inhull))))
                    surfels_in_bbx_idxs = np.where(surfels_in_bbx_mask)[0] 

                    # surfel_points_in_camframe = transformPoints(self.spheremap.surfel_points[surfels_in_bbx_mask, :], np.linalg.inv(T_orig_to_current_cam))
                    contained_surfels_mask = np.full(n_surfels, False)
                    # print(surfel_points_in_camframe .shape)
                    # print(contained_surfels_mask .shape)
                    # contained_surfels_mask[surfels_in_bbx_idxs] = contained_surfels_mask
                    
                    # max_z = np.max(z_ok_points[2, :])
                    # print(z_ok_points.shape)
                    # print("maxz")
                    # print(np.sum(surfel_points_in_camframe[surfels_in_bbx_idxs, 2] <= max_z))

                    
                    print("AAA")
                    print(surfels_in_bbx_idxs.shape)
                    tick = time.time()
                    contained_surfels_mask[surfels_in_bbx_idxs] = fov_mesh.contains(surfel_points_in_camframe[surfels_in_bbx_idxs, :])
                    print(time.time() - tick)

                    tick = time.time()
                    # CHECK IF THE SURFELS PROJECT TO HULL FORMED BY ONLY OBSTACLE PTS (dont delete pts with OFS freespace!)
                    pixpos = getPixelPositions(surfel_points_in_camframe[surfels_in_bbx_idxs, :].T, self.K)
                    # print("AAA")
                    # inhull = np.array([visible_obstacle_pts_polygon.contains(geometry.Point(pixpos[i, 0], pixpos[i, 1])) for i in range(pixpos.shape[0])])
                    # inhull = hull2d.find_simplex(pixpos)>=0
                    inhull = tri.find_simplex(pixpos)>=0

                    print(np.sum(inhull.flatten()))
                    # print(inhull)
                    # inhull = np.array([visible_obstacle_pts_polygon.contains(pixpos)])
                    contained_surfels_mask[surfels_in_bbx_idxs] = np.logical_and(contained_surfels_mask[surfels_in_bbx_idxs], inhull)

                    contained_surfels_idxs = np.where(contained_surfels_mask)[0]
                    contained_surfels_dists = np.linalg.norm(surfel_points_in_camframe[contained_surfels_mask, :], axis = 1)
                    print(time.time() - tick)


                    old_update_dt = time.time() - interm_time2
                    interm_time2 = time.time()
                    if self.verbose_submap_construction:
                        print("PART CONTAINS OBSTACLES: " + str((old_update_dt ) * 1000) +  " ms")
                    
                    # DETERMINE PTS THAT HAVE BEEN MEASURED FROM UP CLOSE AND NOW FALL INTO FOV
                    contained_surfels_minmeas_dists = self.spheremap.surfel_minmeas_dists[contained_surfels_mask]
                    protected_contained_surfels_mask = contained_surfels_dists > self.protection_constant * contained_surfels_minmeas_dists 
                    if not self.do_distbased_fusion:
                        protected_contained_surfels_mask = np.zeros(protected_contained_surfels_mask.shape)
                    if np.any(protected_contained_surfels_mask):
                        nondeleted_visible_surfels = surfel_points_in_camframe[contained_surfels_mask, :][protected_contained_surfels_mask, :]

                    deleted_surfels_idxs = contained_surfels_idxs[np.logical_not(protected_contained_surfels_mask)]
                    # TODO - delete only if SOMEHOW DEEP in the MESH!

                    surfel_deletion_mask = np.full(self.spheremap.surfel_points.shape[0], False)
                    surfel_deletion_mask[deleted_surfels_idxs] = True

                    keep_mask = np.logical_not(surfel_deletion_mask)
                    self.spheremap.surfel_points = self.spheremap.surfel_points[keep_mask]
                    self.spheremap.surfel_minmeas_dists = self.spheremap.surfel_minmeas_dists[keep_mask]

                old_update_dt = time.time() - interm_time2
                interm_time2 = time.time()
                if self.verbose_submap_construction:
                    print("PART 0: " + str((old_update_dt ) * 1000) +  " ms")


                pt_distbased_dt = time.time() - interm_time
                interm_time = time.time()
                if self.verbose_submap_construction:
                    print("POINTS DISTBASED CHECKING time: " + str((meshing_dt) * 1000) +  " ms")
                runtime_data.append(pt_distbased_dt)
                runtime_obstacles_and_frontiers_update += pt_distbased_dt


                # GET THE SPHERES THAT COULD BE UPDATED BY CURRENTLY SEEN DATA
                if np.any(z_ok_mask):
                    interm_time2 = time.time()

                    visible_old_points = z_ok_points
                    if not nondeleted_visible_surfels is None:
                        # print("ADDING NONDELTED PTS: " + str(nondeleted_visible_surfels.shape[0]))
                        # TODO fix
                        # visible_points = np.append(visible_old_points, nondeleted_visible_surfels, axis=0)
                        visible_obstacle_pts = np.concatenate((visible_obstacle_pts .T, nondeleted_visible_surfels), axis=0).T
                    worked_sphere_idxs = worked_sphere_idxs
                    # print("N WORKED SPHERE IDXS:")
                    # print(worked_sphere_idxs.size)

                    # GET DISTANCES OF OLD SPHERE POINTS TO THE VISIBLE MESH AND TO CONSIDERED OBSTACLE POINTS (visible and old ones that are far, but measured from close)
                    interm_time2 = time.time()

                    spheres_in_visible_mesh = fov_mesh.contains(visible_old_points)

                    old_update_dt = time.time() - interm_time2
                    interm_time2 = time.time()
                    if self.verbose_submap_construction:
                        print("-- contains sphere centers: " + str((old_update_dt ) * 1000) +  " ms")

                    old_spheres_fov_dists_signed = fov_mesh_query.signed_distance(visible_old_points)
                    # old_spheres_fov_dists_signed = np.full(self.spheremap.radii.shape, 0)
                    # if np.any(spheres_in_visible_mesh):
                    #     old_spheres_fov_dists_signed[spheres_in_visible_mesh] = fov_mesh_query.signed_distance(visible_old_points[spheres_in_visible_mesh, :])
                    old_spheres_fov_dists = np.abs(old_spheres_fov_dists_signed)

                    old_update_dt = time.time() - interm_time2
                    interm_time2 = time.time()
                    if self.verbose_submap_construction:
                        print("-- signed distance: " + str((old_update_dt ) * 1000) +  " ms")

                    pt_distmatrix = scipy.spatial.distance_matrix(visible_old_points, visible_obstacle_pts .T)
                    old_spheres_obs_dists = np.min(pt_distmatrix, axis = 1)
                    upperbound_combined = np.minimum( np.minimum(old_spheres_fov_dists, old_spheres_obs_dists), self.spheremap.max_allowed_radius)

                    old_update_dt = time.time() - interm_time2
                    interm_time2 = time.time()
                    if self.verbose_submap_construction:
                        print("-- distmatrix: " + str((old_update_dt ) * 1000) +  " ms")

                    should_decrease_radius = old_spheres_obs_dists < self.spheremap.radii[worked_sphere_idxs]
                    could_increase_radius = np.logical_and(upperbound_combined > self.spheremap.radii[worked_sphere_idxs], spheres_in_visible_mesh)

                    for i in range(worked_sphere_idxs.size):
                        if should_decrease_radius[i]:
                            self.spheremap.radii[worked_sphere_idxs[i]] = old_spheres_obs_dists[i]
                        elif could_increase_radius[i]:
                            self.spheremap.radii[worked_sphere_idxs[i]] = upperbound_combined[i]

                    old_update_dt = time.time() - interm_time2
                    interm_time2 = time.time()
                    if self.verbose_submap_construction:
                        print("-- radii: " + str((old_update_dt ) * 1000) +  " ms")

                    meas_time = time.time() - interm_time
                    interm_time = time.time()
                    if self.verbose_submap_construction:
                        print("OLD SPHERE UPDATE (without reconnecting) time: " + str((meas_time) * 1000) +  " ms")
                    runtime_data.append(meas_time)
                    runtime_old_spheres_update += meas_time

                    # FIND WHICH SMALL SPHERES TO PRUNE AND STOP WORKING WITH THEM, BUT REMEMBER INDICES TO KILL THEM IN THE END
                    idx_picker = self.spheremap.radii[worked_sphere_idxs[i]] < self.spheremap.min_radius
                    toosmall_idxs = worked_sphere_idxs[idx_picker]
                    shouldkeep = np.full((n_spheres_old , 1), True)
                    shouldkeep[toosmall_idxs] = False
                    shouldkeep = shouldkeep .flatten()

                    worked_sphere_idxs = worked_sphere_idxs[np.logical_not(idx_picker)].flatten()

                    # self.spheremap.consistencyCheck()
                    # RE-CHECK CONNECTIONS
                    self.spheremap.updateConnections(worked_sphere_idxs)

                    old_update_dt = time.time() - interm_time2
                    interm_time2 = time.time()
                    if self.verbose_submap_construction:
                        print("-- connections: " + str((old_update_dt ) * 1000) +  " ms")

                    # AT THE END, PRUNE THE EXISTING SPHERES THAT BECAME TOO SMALL (delete their pos, radius and conns)
                    # self.spheremap.consistencyCheck()

                    self.spheremap.removeSpheresIfRedundant(worked_sphere_idxs)

                    old_update_dt = time.time() - interm_time2
                    interm_time2 = time.time()
                    if self.verbose_submap_construction:
                        print("-- redundancy: " + str((old_update_dt ) * 1000) +  " ms")

                    # self.spheremap.removeNodes(np.where(idx_picker)[0])
            # # #}
            
            meas_time = time.time() - interm_time
            interm_time = time.time()
            if self.verbose_submap_construction:
                print("GRAPH UPKEEP (old spheres) time: " + str((meas_time ) * 1000) +  " ms")
            runtime_data.append(meas_time)
            runtime_connectivity_update += meas_time


            # TRY ADDING NEW SPHERES# #{
            # TODO fix - by raycasting!!!
            # TODO - better sampling to sample near UAV prioritized
            max_sphere_update_dist = self.max_sphere_update_dist

            n_sampled = self.n_sphere_samples_per_update
            sampling_pts = np.random.rand(n_sampled, 2)  # Random points in [0, 1] range for x and y
            sampling_pts = sampling_pts * [self.width, self.height]

            # CHECK THE SAMPLING DIRS ARE INSIDE THE 2D CONVEX HULL OF 3D POINTS
            # inhull = np.array([visible_obstacle_pts_polygon.contains(geometry.Point(p[0], p[1])) for p in sampling_pts])
            inhull = np.array([extended_pts_polygon.contains(geometry.Point(p[0], p[1])) for p in sampling_pts])
            if not np.any(inhull):
                if self.verbose_submap_construction:
                    print("NONE IN HULL")
                self.spheremap.labelSpheresByConnectivity()
                return
            sampling_pts = sampling_pts[inhull, :]

            n_sampled = sampling_pts.shape[0]

            # NOW PROJECT THEM TO 3D SPACE
            sampling_pts = np.concatenate((sampling_pts.T, np.full((1, n_sampled), 1)))

            invK = np.linalg.inv(self.K)
            sampling_pts = invK @ sampling_pts

            # rand_z = np.random.rand(1, n_sampled) * max_sphere_update_dist
            # rand_z[rand_z > max_sphere_update_dist] = max_sphere_update_dist

            # FILTER PTS - CHECK THAT THE MAX DIST IS NOT BEHIND THE OBSTACLE MESH BY RAYCASTING
            ray_hit_pts, index_ray, index_tri = extended_obstacle_mesh.ray.intersects_location(
            ray_origins=np.zeros(sampling_pts.shape).T, ray_directions=sampling_pts.T)
            # TODO - figure out why some rays maybe dont hit?
            n_sampled = ray_hit_pts.shape[0]

            # MAKE THEM BE IN RAND POSITION BETWEEN CAM AND MESH HIT POSITION, UP TO MAX Z (TODO-rename to dist from Z)
            # ray_hit_dists = np.linalg.norm(ray_hit_pts, axis=1)
            # ray_hit_pts = ray_hit_pts / ray_hit_dists
            # ray_hit_dists[ray_hit_dists > max_sphere_update_dist] = max_sphere_update_dist

            # scaling = ray_hit_pts[:,2]

            n_spheres_to_add = 0
            n_spheres_before_adding = self.spheremap.points.shape[0]
            if ray_hit_pts.size == 0:
                print("WARN: no sphere sampling rays hit obstacle mesh")
            else:
                scaling = np.ones(ray_hit_pts.shape[0])
                hit_dists = np.linalg.norm(ray_hit_pts, axis=1)
                # scaling[ray_hit_pts[:,2] > max_sphere_update_dist] = max_sphere_update_dist
                farmask = hit_dists > max_sphere_update_dist
                scaling[farmask] = max_sphere_update_dist * np.reciprocal(hit_dists[farmask])
                # for i in range(scaling.shape[0]):
                #     if np.random.rand() < 0.1 and hit_dists[i] > 2:
                #         scaling[i] = 2.0 / hit_dists[i]
                ray_hit_pts = ray_hit_pts * scaling.reshape((ray_hit_pts.shape[0], 1))

                # scaling = scaling.flatten() / ray_hit_pts[:,2].flatten()
                # ray_hit_pts =  ray_hit_pts * scaling.reshape((ray_hit_pts.shape[0], 1))

                sampling_pts =  (np.random.rand(n_sampled, 1) * ray_hit_pts).T
                # TODO - add max sampling dist


                orig_3dpt_indices_in_hull = np.unique(hull2d.simplices)

                # TRY ADDING NEW SPHERES AT SAMPLED POSITIONS
                new_spheres_fov_dists = np.abs(fov_mesh_query.signed_distance(sampling_pts.T))
                # print(sampling_pts.T.shape)
                # print(self.spheremap.surfel_points.shape)
                # pt_distmatrix = scipy.spatial.distance_matrix(sampling_pts.T, self.spheremap.surfel_points)
                # new_spheres_obs_dists = np.min(pt_distmatrix, axis = 1)
                # mindists = np.minimum(new_spheres_obs_dists, new_spheres_fov_dists)

                mindists = new_spheres_fov_dists 
                new_sphere_idxs = mindists > min_rad

                n_spheres_to_add = np.sum(new_sphere_idxs)
                if self.verbose_submap_construction:
                    print("PUTATIVE SPHERES THAT PASSED FIRST RADIUS CHECKS: " + str(n_spheres_to_add))
                if n_spheres_to_add == 0:
                    if self.verbose_submap_construction:
                        print("NO NEW SPHERES TO ADD")
                    self.spheremap.labelSpheresByConnectivity()
                    return
                else:
                    # TRANSFORM POINTS FROM CAM ORIGIN TO SPHEREMAP ORIGIN! - DRAW OUT!
                    new_sphere_locations_in_spheremap_origin = transformPoints(sampling_pts[:, new_sphere_idxs].T, T_orig_to_current_cam)

                    self.spheremap.points = np.concatenate((self.spheremap.points, new_sphere_locations_in_spheremap_origin))
                    self.spheremap.radii = np.concatenate((self.spheremap.radii.flatten(), mindists[new_sphere_idxs].flatten()))
                    self.spheremap.connections = np.concatenate((self.spheremap.connections.flatten(), np.array([None for i in range(n_spheres_to_add)], dtype=object).flatten()))
            # # #}

            # ADD SPHERE AT CURRENT POSITION!# #{
            pos_in_smap_frame = T_delta_odom[:3, 3].reshape((1,3)) 
            self.spheremap.points = np.concatenate((self.spheremap.points, pos_in_smap_frame))
            self.spheremap.radii = np.concatenate((self.spheremap.radii.flatten(), np.array([self.uav_radius]) ))
            self.spheremap.connections = np.concatenate((self.spheremap.connections.flatten(), np.array([None], dtype=object).flatten()))
            n_spheres_to_add += 1
            # # #}

            meas_time = time.time() - interm_time
            interm_time = time.time()
            if self.verbose_submap_construction:
                print("NEW SPHERE UPDATE time: " + str((meas_time ) * 1000) +  " ms")
            runtime_data.append(meas_time)
            runtime_new_spheres_update += meas_time

            # CONNECTIONS UPDATE AND SPARSIFICATION# #{
            self.spheremap.updateConnections(np.arange(n_spheres_before_adding, n_spheres_before_adding+n_spheres_to_add))
            new_idxs = np.arange(self.spheremap.radii.size)[self.spheremap.radii.size - n_spheres_to_add : self.spheremap.radii.size]
            self.spheremap.removeSpheresIfRedundant(new_idxs)
            self.spheremap.labelSpheresByConnectivity()
# # #}

            meas_time = time.time() - interm_time
            interm_time = time.time()
            if self.verbose_submap_construction:
                print("NEW SPHERES RECONNECTING UPDATE time: " + str((meas_time) * 1000) +  " ms")
            runtime_data.append(meas_time)
            runtime_connectivity_update += meas_time

            comp_start_time = time.time()
            self.spheremap.spheres_kdtree = KDTree(self.spheremap.points)
            self.spheremap.max_radius = np.max(self.spheremap.radii)

            kdtree_comp_time = time.time() - comp_start_time
            if self.verbose_submap_construction:
                print("Spheres KDTree computation: " + str((kdtree_comp_time) * 1000) +  " ms")
            comp_start_time = time.time()


            # self.spheremap.nodes_distmatrix = scipy.spatial.distance_matrix(self.spheremap.points, self.spheremap.points)
            n_now_nodes = self.spheremap.radii.size
            self.spheremap.nodes_distmatrix = np.zeros((n_now_nodes, n_now_nodes))
            for i in range(n_now_nodes):
                conns = self.spheremap.connections[i]
                if not conns is None:
                    # print("CONNS")
                    # print(conns)
                    dirvecs =  self.spheremap.points[conns, :] - self.spheremap.points[i, :].reshape((1,3))
                    # print(dirvecs.shape)
                    self.spheremap.nodes_distmatrix[i, conns] = np.linalg.norm(dirvecs, axis = 1)
            comp_time = time.time() - comp_start_time
            if self.verbose_submap_construction:
                print("distmatrix computation: " + str((comp_time) * 1000) +  " ms")
            runtime_data.append(kdtree_comp_time + comp_time)
            runtime_connectivity_update += kdtree_comp_time + comp_time

            # HANDLE ADDING/REMOVING VISIBLE 3D POINTS
            comp_start_time = time.time()

            positive_z_points = positive_z_points.T
            # pts_for_surfel_addition = positive_z_points[positive_z_points[:, 2] < max_sphere_update_dist]
            # pts_for_surfel_addition = positive_z_points[np.linalg.norm(positive_z_points, axis=1) < max_sphere_update_dist]
            pts_for_surfel_addition_tmp = pts_for_surfel_addition_tmp.T
            pts_for_surfel_addition = pts_for_surfel_addition_tmp[np.linalg.norm(pts_for_surfel_addition_tmp, axis=1) < max_sphere_update_dist]
            visible_pts_in_spheremap_frame = transformPoints(pts_for_surfel_addition, T_orig_to_current_cam)
            self.spheremap.updateSurfels(T_orig_to_current_cam, visible_pts_in_spheremap_frame , pixpos, tri.simplices, self.surfel_resolution, positive_z_slam_ids)

            if self.do_frontiers:
                # Sample some around nearby spheres too
                robotpos = T_orig_to_current_cam[:3, 3]
                all_pts = self.spheremap.points
                mask = np.all(
                    (all_pts >= (robotpos - self.max_sphere_update_dist)) & (all_pts <= (robotpos + self.max_sphere_update_dist)),
                    axis=1
                )
                
                # Select only points inside bounding box
                pts_in_bbox = all_pts[mask]
                radii_in_bbox = self.spheremap.radii[mask]

                # Sample
                N = int(self.n_frontier_samples / 2)
                # N = self.n_frontier_samples 
                fr_samples2 = self.sampleFrontiersAroundSpheres(pts_in_bbox, radii_in_bbox, N)

                # # Uniform sample
                # N = self.max_sphere_update_dist * 2  # half-extent of the cube
                # num_samples = self.n_frontier_samples
                
                # # Uniform random samples in [-N, N] in each axis, then shift by robotpos
                # fr_samples2 = robotpos + np.random.uniform(
                #     low=-N, high=N, size=(num_samples, 3)
                # )

                # Update
                fr_samples = transformPoints(fr_samples, T_orig_to_current_cam) # transform pts sampled around visible poly
                fr_samples = np.concatenate((fr_samples, fr_samples2))
                # fr_samples = fr_samples2
                # self.spheremap.updateFrontiers(transformPoints(fr_samples, T_orig_to_current_cam), self.frontier_resolution)

                self.potential_fr_sample_pts = fr_samples
                # Do this in explo planning thread
                # self.spheremap.updateFrontiers(fr_samples, self.frontier_resolution)

                # Visualize samples
                # fields = [
                #     PointField('x', 0, PointField.FLOAT32, 1),
                #     PointField('y', 4, PointField.FLOAT32, 1),
                #     PointField('z', 8, PointField.FLOAT32, 1),
                # ]
                
                # header = std_msgs.msg.Header()
                # header.stamp = rospy.Time.now()
                # header.frame_id = "map"
                
                # # Ensure float32 and contiguous memory
                # points_np = np.asarray(fr_samples, dtype=np.float32)
                # pc2_msg = pc2.create_cloud(header, fields, points_np.reshape(-1))
                
                # self.frontier_samples_pub.publish(pc2_msg)

            comp_time = time.time() - comp_start_time
            if self.verbose_submap_construction:
                print("SURFELS+FRONTIERS integration time: " + str((comp_time) * 1000) +  " ms")
            runtime_data.append(comp_time)
            runtime_obstacles_and_frontiers_update += comp_time

            # TOTAL COMPUTATION TIME
            comp_time = time.time() - update_start_time
            if self.verbose_submap_construction:
                print("SPHEREMAP integration time: " + str((comp_time) * 1000) +  " ms")
            runtime_data.append(comp_time)
            total_runtime = comp_time

            # PUBLISH RUNTIMES
            runtime_msg = mrs_msgs.msg.Float64ArrayStamped()
            h = std_msgs.msg.Header()
            h.stamp = rospy.Time.now()
            runtime_msg.header = h
            # runtime_msg.values = runtime_data
            runtime_msg.values = [runtime_polyhedron_update, runtime_obstacles_and_frontiers_update, runtime_old_spheres_update, runtime_new_spheres_update, runtime_connectivity_update]
            self.runtimes_pub.publish(runtime_msg)

            if self.verbose_submap_construction:
                print(" TOTAL UPDATE TIME:")
                print(total_runtime)
                print(" INTERM TIMES SUM:")
                print(np.sum(np.array(runtime_msg.values)))

            # VISUALIZE CURRENT SPHERES
            self.visualize_spheremap()
            comp_time = time.time() - comp_start_time
            if self.verbose_submap_construction:
                print("SPHEREMAP visualization time: " + str((comp_time) * 1000) +  " ms")
# # #}

    # def ofs_freespace_update(self):# #{
    def ofs_freespace_update(self):
        print("ofs FREESPACE")
        self.visualize_ofs_freespace_pts()
        # T_odom_to_cam = lookupTransformAsMatrix(self.fcu_frame, self.mapper.camera_frame, self.tf_listener) @ T_fcu_to_cam 

        T_odom_to_cam = lookupTransformAsMatrix(self.odom_frame, self.camera_frame, self.tf_listener)

        should_add_pose_to_buffer = False

        last_pose_trans_dif = None
        last_pose_angle_dif = None

        min_trans_dif = 0.3
        min_angle_dif = np.pi / 4

        if len(self.ff_pose_buffer) > 0:
            prev_T = self.ff_pose_buffer[-1]
            trans_dif = np.linalg.norm(T_odom_to_cam[:3,3] - prev_T[:3, 3])
            dotprod = np.dot(T_odom_to_cam[:3,2], prev_T[:3,2])
            angle_dif = np.arccos(dotprod)

            if trans_dif > min_trans_dif or angle_dif > min_angle_dif:
                should_add_pose_to_buffer = True
        else:
            should_add_pose_to_buffer = True


        # DECIDE IF ADD THIS POSE
        if should_add_pose_to_buffer:
            # print("ADDING POSE, BUFFERLEN: " + str(len(self.ff_pose_buffer)))
            self.ff_pose_buffer.append(T_odom_to_cam)
            if len(self.ff_pose_buffer) > self.max_ff_buffer_len:
                self.ff_pose_buffer.pop(0)
        else:
            return

        bufferlen = len(self.ff_pose_buffer)
        n_pts = self.egocentric_ff_pts.shape[0]
        if bufferlen < 2:
            return

        visibility_mask = np.full((bufferlen, n_pts), False)
        parallax_mask = np.full((bufferlen, n_pts), False)

        # GO THRU ALL PREV POSES N CHECK IF "TRACKED" AND HOW MUCH TRANSLATION/PARALLAX
        # ll_corner = np.array([0, 0])
        # ur_corner = np.array([self.width, self.height])

        ll_corner = np.array([self.ff_trim*self.width, self.ff_trim*self.height])
        ur_corner = np.array([(1 - self.ff_trim) * self.width, (1 - self.ff_trim) * self.height])

        # min_parallax = np.pi / 12
        min_parallax = self.ff_min_parallax 

        for i in range(len(self.ff_pose_buffer) - 1):
            # T_relative = np.linalg.inv(T_odom_to_cam) @ self.ff_pose_buffer[i]

            T_relative = np.linalg.inv(self.ff_pose_buffer[i]) @ T_odom_to_cam

            # print("T relative dist: " + str(np.linalg.norm(T_relative[:3,3])))
            translated_pts = transformPoints(self.egocentric_ff_pts, T_relative)

            dot_product = np.sum(self.egocentric_ff_pts * (self.egocentric_ff_pts + T_relative[:3,3]), axis=1)
            norms = np.linalg.norm(self.egocentric_ff_pts, axis=1) * np.linalg.norm(translated_pts, axis=1)
            cos_theta = dot_product / norms
            # angles = np.arccos(np.clip(cos_theta, -1.0, 1.0))
            angles = np.arccos(cos_theta)
            parallax_mask[i, :] = np.abs(angles) >= min_parallax

            # print("CUR PTS:")
            # print(self.egocentric_ff_pts[0, :])
            # print("PREV PTS:")
            # print(translated_pts[0, :])

            # print(translated_pts.shape)
            pixpos_there = getPixelPositions(translated_pts.T, self.new_K)
            # print("PIXPOS PTS:")
            # print(pixpos_there[0, :])

            in_img = np.all(np.logical_and(pixpos_there <= ur_corner, pixpos_there >= ll_corner), axis = 1)
            # print("N pts in img: " + str(np.sum(in_img)))
            visibility_mask[i, :] = in_img
        # print(visibility_mask)
        vis_lengths = np.sum(visibility_mask, axis = 0).flatten()
        # print(vis_lengths)
        passed_visibility = vis_lengths == bufferlen - 1

        if self.ff_points_ignore_tracking:
            passed_parallax = np.sum(parallax_mask, axis = 0).flatten() > 0
            self.ff_points_active_mask = passed_parallax
        else:
            # self.ff_points_active_mask = np.logical_and(passed_visibility, passed_parallax)

            self.ff_points_active_mask = np.full((n_pts,1), False).flatten()
            for i in range(n_pts):
                vis_len = np.argmax(np.logical_not(visibility_mask[:, i]).flatten()) #first idx at which visibility is false
                if vis_len == 0:
                    continue
                self.ff_points_active_mask[i] = np.any(parallax_mask[:vis_len, i])

        # print("N VISIBLE: " + str(np.sum(passed_visibility)))
        # print("N PARALLAX-OK: " + str(np.sum(passed_parallax)))
        print("N ACTIVE: " + str(np.sum(self.ff_points_active_mask)))

        if self.ff_points_always_active:
            self.ff_points_active_mask = np.full((n_pts,1), True).flatten()
                # # #}

    # def init_ofs_freespace_pts(self):# #{
    def init_ofs_freespace_pts(self):

        self.ofs_pixpos_lastframe = None
        self.ff_trim = 0.2

        xstart = self.ff_trim* self.width
        xstep = (1-2*self.ff_trim) * self.width / self.ff_pts_W
        ystart = self.ff_trim* self.height
        ystep = (1-2*self.ff_trim) * self.height / self.ff_pts_H



        pts_2d = np.array([[xstart + x * xstep, ystart + y * ystep] for x in range(self.ff_pts_W+1) for y in range(self.ff_pts_H+1)])
        n_pts = pts_2d.shape[0]

        print("INIT OFS FSPACE PTS:")
        print(pts_2d)

        print("KOCKA")
        # print(pts_2d)
        print("KOCKA")
        print(pts_2d[0, :])
        print(pts_2d[-1, :])
        # print("SHAPE")
        # print(pts_2d.shape)
        sampling_pts = np.concatenate((pts_2d.T, np.full((1, n_pts), 1)))
        invK = np.linalg.inv(self.new_K)
        self.egocentric_ff_pts = (invK @ sampling_pts).T
        print(self.egocentric_ff_pts[0, :])
        print(self.egocentric_ff_pts[-1, :])
        self.egocentric_ff_pts = self.ff_dist * (self.egocentric_ff_pts / np.linalg.norm(self.egocentric_ff_pts, axis = 1).reshape((n_pts, 1)))

        self.ff_points_active_mask = np.full((n_pts,1), False).flatten()
        # # #}

    # UTILS

    def sampleFrontiersAroundSpheres(self, old_pts, old_radii, N):# # #{
        M = len(old_pts)
        print("PES")
        
        # Pick random spheres
        sphere_indices = np.random.randint(0, M, size=N)
        centers = old_pts[sphere_indices]
        print(centers.shape)
        radii = old_radii[sphere_indices].flatten()
        print(radii.shape)

        # Random radii within [1.5r, 2.7r] with correct 3D distribution
        # (cube-root transform for uniform volume sampling)
        r_inner = radii + self.spheremap.frontier_deleting_inside_dist
        r_outer = radii + self.spheremap.frontier_toofar_from_fspace_dist
        # r_inner = 1.5 * radii
        # r_outer = 2.7 * radii

        u = np.random.rand(N)
        sampled_radii = u * (r_outer - r_inner) + r_inner
        print(sampled_radii.shape)

        # Random directions on unit sphere
        phi = 2 * np.pi * np.random.rand(N)
        cos_theta = 2 * np.random.rand(N) - 1
        sin_theta = np.sqrt(1 - cos_theta**2)

        directions = np.stack([
            sin_theta * np.cos(phi),
            sin_theta * np.sin(phi),
            cos_theta
        ], axis=1)

        # Final sampled points
        samples = centers + sampled_radii[:, None] * directions
        return samples
# # #}

    def saveEpisodeFull(self, req):# # #{
        # with ScopedLock(self.spheremap_mutex):
        print("SAVING EPISODE MEMORY CHUNK")

        fpath = rospkg.RosPack().get_path('monospheres') + "/memories/last_episode.pickle"
        self.mchunk.submaps.append(self.spheremap)
        self.mchunk.save(fpath)
        self.mchunk.submaps.pop()

        print("EPISODE SAVED TO " + str(fpath))
        return EmptySrvResponse()# # #}

    def get_visited_viewpoints_global(self):# # #{
        res_pts = None
        res_headings = None
        for smap in [self.spheremap] + self.mchunk.submaps:
            if len(smap.visual_keyframes) > 0:
                pts_smap = np.array([kf.position for kf in smap.visual_keyframes])
                headings_smap = np.array([kf.heading for kf in smap.visual_keyframes])
                pts_global, headings_global = transformViewpoints(pts_smap, headings_smap, smap.T_global_to_own_origin)

                if res_pts is None:
                    res_pts = pts_global
                    res_headings = headings_global
                else:
                    res_pts = np.concatenate((res_pts, pts_global))
                    res_headings = np.concatenate((res_headings, headings_global))

        return res_pts, res_headings# # #}

    def save_camera_img_for_visualization(self, msg):
        # self.latest_camera_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        np_arr = np.frombuffer(msg.data, np.uint8)
        self.latest_camera_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # VISUALIZE

    def visualize_ofs_freespace_pts(self):# # #{
        marker_array = MarkerArray()

        ms = self.marker_scale
        marker = Marker()
        marker.header.frame_id = self.camera_frame  # Adjust the frame_id as needed
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        # marker.scale.x = 1.2  # Adjust the size of the points
        # marker.scale.y = 1.2
        marker.scale.x = ms *2.2  # Adjust the size of the points
        marker.scale.y = ms *2.2
        marker.color.a = 1
        marker.color.r = 0
        marker.color.g = 1
        marker.color.b = 0

        # marker.id = marker_id
        # marker_id += 1

        # Convert the 3D points to Point messages
        spts = self.egocentric_ff_pts
        n_pts = spts.shape[0]
        points_msg = [Point(x=spts[i, 0], y=spts[i, 1], z=spts[i, 2]) for i in range(n_pts)]
        marker.points = points_msg
        marker_array.markers.append(marker)

        # TODO - colors based on activity!
        for i in range(self.egocentric_ff_pts.shape[0]):
            clr = ColorRGBA(r=0.0, g=0.0, b=0.0, a=0.5)
            if self.ff_points_active_mask[i]:
                clr = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
            marker.colors.append(clr)

        self.assumed_freespace_vis_pub.publish(marker_array)
# # #}

    def visualize_episode_submaps(self):# # #{
        marker_array = MarkerArray()

        max_maps_to_vis = 20

        if max_maps_to_vis > len(self.mchunk.submaps):
            max_maps_to_vis = len(self.mchunk.submaps)
        if len(self.mchunk.submaps) == 0:
            return

        print("VISUALIZING MAPS:" )
        print(max_maps_to_vis)

        clr_index = 0
        max_clrs = 4
        for i in range(max_maps_to_vis):
            idx = len(self.mchunk.submaps)-(1+i)
            print(idx)
            clr_index = idx % max_clrs
            # if self.mchunk.submaps[idx].memorized_transform_to_prev_map is None:
            #     break
            print("T")
            print(self.mchunk.submaps[idx].T_global_to_own_origin)
            self.get_spheremap_marker_array(marker_array, self.mchunk.submaps[idx], self.mchunk.submaps[idx].T_global_to_own_origin, alternative_look = True, do_connections = False, do_surfels = True, do_spheres = False, do_map2map_conns=False, ms=self.marker_scale, clr_index = clr_index, alpha = 0.5)

        self.recent_submaps_vis_pub.publish(marker_array)

        return
# # #}

    def visualize_spheremap(self):# # #{
        if self.spheremap is None:
            return

        marker_array = MarkerArray()
        self.get_spheremap_marker_array(marker_array, self.spheremap, self.spheremap.T_global_to_own_origin, ms=self.marker_scale, do_spheres=False, do_surfels=True, do_frontiers=True)
        self.spheremap_outline_pub.publish(marker_array)

        marker_array = MarkerArray()
        self.get_spheremap_marker_array(marker_array, self.spheremap, self.spheremap.T_global_to_own_origin, ms=self.marker_scale, do_spheres=True, do_surfels=False, do_keyframes=False)
        self.spheremap_freespace_pub.publish(marker_array)
# # #}
    
    def visualize_polygon(self, pixpos, tri, do_ffs = False):# # #{
        # rgb = np.repeat(copy.deepcopy(self.new_frame)[:, :, np.newaxis], 3, axis=2)
        # rgb = np.zeros((self.new_frame.shape[0], self.new_frame.shape[1], 3), dtype=np.uint8)
        rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        if not self.latest_camera_image is None:
            # rgb = self.latest_camera_image
            # print(self.latest_camera_image.shape)
            rgb = cv2.cvtColor(self.latest_camera_image, cv2.COLOR_RGB2GRAY)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)


        line_clr = (0, 255, 0)
        point_clr = (255, 0, 0)
        ffs_clr = (150, 150, 255)
        square_side_length = 14

        # LINES
        for i in range(len(tri.simplices)):
            a = pixpos[tri.simplices[i][0], :].astype(int)
            b = pixpos[tri.simplices[i][1], :].astype(int)
            c = pixpos[tri.simplices[i][2], :].astype(int)

            rgb = cv2.line(rgb, (a[0], a[1]), (b[0], b[1]), line_clr, 3)
            rgb = cv2.line(rgb, (a[0], a[1]), (c[0], c[1]), line_clr, 3)
            rgb = cv2.line(rgb, (c[0], c[1]), (b[0], b[1]), line_clr, 3)

        # POINTS
        for i in range(len(tri.simplices)):
            a = pixpos[tri.simplices[i][0], :].astype(int)
            b = pixpos[tri.simplices[i][1], :].astype(int)
            c = pixpos[tri.simplices[i][2], :].astype(int)

            for pt in [a,b,c]:
                x = pt[0]
                y = pt[1]
                top_left = (x - square_side_length  // 2, y - square_side_length    // 2)
                bottom_right = (x + square_side_length   // 2, y + square_side_length     // 2)
                cv2.rectangle(rgb, top_left, bottom_right, point_clr, -1)


        # OFS FSPACE
        if do_ffs:
            if not self.ofs_pixpos_lastframe is None:
                for i in range(self.ofs_pixpos_lastframe.shape[0]):
                    x = int(self.ofs_pixpos_lastframe[i, 0])
                    y = int(self.ofs_pixpos_lastframe[i, 1])
                    # rgb = cv2.circle(rgb, (x, y), 5, 
                    #                # (0, 255, 0), -1) 
                    #                ffs_clr, -1) 

                    top_left = (x - square_side_length  // 2, y - square_side_length    // 2)
                    bottom_right = (x + square_side_length   // 2, y + square_side_length     // 2)
                    cv2.rectangle(rgb, top_left, bottom_right, ffs_clr, -1)

        res = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return res
# # #}

    def get_spheremap_marker_array(self, marker_array, smap, T_inv, alternative_look=False, do_connections=False,  do_surfels=True, do_spheres=True, do_keyframes=False, do_normals=False, do_map2map_conns=True, do_frontiers=False, do_centroids=False, ms=1, clr_index =0, alpha = 1, rgb=None):# # #{
        # T_vis = np.linalg.inv(T_inv)
        T_vis = T_inv
        pts = transformPoints(smap.points, T_vis)

        marker_id = 0
        if len(marker_array.markers) > 0:
            marker_id = marker_array.markers[-1].id + 1

        if do_centroids:
            centroid = smap.centroid.reshape((1,3))
            t_centroid = transformPoints(centroid, T_vis).flatten()

            marker = Marker()
            marker.header.frame_id = self.odom_frame  # Adjust the frame_id as needed
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            vis_r = smap.freespace_bounding_radius / 4
            marker.scale.x = vis_r  # Adjust the size of the points
            marker.scale.y = vis_r
            marker.scale.z = vis_r
            marker.color.a = 1
            marker.color.r = 0.5
            marker.color.b = 1

            marker.id = marker_id
            marker_id += 1
            marker.pose.position = Point(t_centroid[0], t_centroid[1], t_centroid[2])

            marker_array.markers.append(marker)

        if do_map2map_conns and len(smap.map2map_conns) > 0:
            n_conns = len(smap.map2map_conns)

            # CONN PTS
            untr_pts = np.array([c.pt_in_first_map_frame for c in smap.map2map_conns])
            untr_pts = untr_pts.reshape((n_conns, 3))
            spts = transformPoints(untr_pts, T_vis)

            marker = Marker()
            # marker.header.frame_id = "global"  # Adjust the frame_id as needed
            marker.header.frame_id = self.odom_frame  # Adjust the frame_id as needed
            marker.type = Marker.POINTS
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            # marker.scale.x = 1.2  # Adjust the size of the points
            # marker.scale.y = 1.2
            marker.scale.x = ms *2.2  # Adjust the size of the points
            marker.scale.y = ms *2.2
            marker.color.a = 1
            marker.color.r = 0.5
            marker.color.b = 1
            # if alternative_look:
            #     marker.color.r = 0.5
            #     marker.color.b = 1

            marker.id = marker_id
            marker_id += 1

            # Convert the 3D points to Point messages
            points_msg = [Point(x=spts[i, 0], y=spts[i, 1], z=spts[i, 2]) for i in range(n_conns)]
            marker.points = points_msg
            marker_array.markers.append(marker)

            # LINES BETWEEN CONN POINTS
            line_marker = Marker()
            line_marker.header.frame_id = self.odom_frame  # Set your desired frame_id
            line_marker.type = Marker.LINE_LIST
            line_marker.action = Marker.ADD
            line_marker.scale.x =ms* 1  # Line width
            line_marker.color.a = 1  # Alpha
            line_marker.color.r = 0.5  # Alpha
            line_marker.color.b = 1  # Alpha

            # if alternative_look:
            #     line_marker.scale.x = 0.1

            line_marker.id = marker_id
            marker_id += 1

            for i in range(n_conns):
                for j in range(i+1, n_conns):
                    point1 = Point()
                    point2 = Point()

                    p1 = spts[i, :]
                    p2 = spts[j, :]
                    
                    point1.x = p1[0]
                    point1.y = p1[1]
                    point1.z = p1[2]
                    point2.x = p2[0]
                    point2.y = p2[1]
                    point2.z = p2[2]
                    line_marker.points.append(point1)
                    line_marker.points.append(point2)
            marker_array.markers.append(line_marker)

        if do_connections:
            line_marker = Marker()
            line_marker.header.frame_id = self.odom_frame  # Set your desired frame_id
            line_marker.type = Marker.LINE_LIST
            line_marker.action = Marker.ADD
            line_marker.scale.x = ms* 0.2  # Line width
            line_marker.color.a = 1.0  # Alpha
            if alternative_look:
                line_marker.scale.x =ms * 0.1

            line_marker.id = marker_id
            marker_id += 1

            for i in range(smap.connections.shape[0]):
                if smap.connections[i] is None:
                    continue
                for j in range(len(smap.connections[i])):
                    point1 = Point()
                    point2 = Point()

                    p1 = pts[i, :]
                    p2 = pts[smap.connections[i].flatten()[j], :]
                    
                    point1.x = p1[0]
                    point1.y = p1[1]
                    point1.z = p1[2]
                    point2.x = p2[0]
                    point2.y = p2[1]
                    point2.z = p2[2]
                    line_marker.points.append(point1)
                    line_marker.points.append(point2)
            marker_array.markers.append(line_marker)

        if do_surfels:
            if not smap.surfel_points is None:
                spts = transformPoints(smap.surfel_points, T_vis)

                marker = Marker()
                marker.header.frame_id = self.odom_frame  # Adjust the frame_id as needed
                marker.type = Marker.POINTS
                marker.action = Marker.ADD
                marker.pose.orientation.w = 1.0
                # marker.scale.x = 1.2  # Adjust the size of the points
                # marker.scale.y = 1.2
                marker.scale.x = ms *0.9  # Adjust the size of the points
                marker.scale.y = ms *0.9
                marker.color.a = alpha
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                if not rgb is None:
                    marker.color.r = rgb[0]
                    marker.color.g = rgb[1]
                    marker.color.b = rgb[2]
                if alternative_look:
                    # marker.color.a = 0.5
                    if rgb is None:
                        if clr_index == 0:
                            marker.color.r = 1.0
                            marker.color.g = 1.0
                            marker.color.b = 0.0
                        elif clr_index == 1:
                            marker.color.r = 0.0
                            marker.color.g = 1.0
                            marker.color.b = 1.0
                        elif clr_index == 2:
                            marker.color.r = 0.4
                            marker.color.g = 0.7
                            marker.color.b = 1.0
                        elif clr_index == 42:
                            marker.color.r = 0.0
                            marker.color.g = 0.0
                            marker.color.b = 0.4
                        else:
                            marker.color.r = 1.0
                            marker.color.g = 0.0
                            marker.color.b = 1.0

                marker.id = copy.deepcopy(marker_id)
                marker_id += 1

                # Convert the 3D points to Point messages
                n_surfels = spts.shape[0]
                points_msg = [Point(x=spts[i, 0], y=spts[i, 1], z=spts[i, 2]) for i in range(n_surfels)]
                marker.points = points_msg
                marker_array.markers.append(marker)
        if do_frontiers:
            if not smap.frontier_points is None:
                spts = transformPoints(smap.frontier_points, T_vis)

                marker = Marker()
                marker.header.frame_id = self.odom_frame  # Adjust the frame_id as needed
                marker.type = Marker.POINTS
                marker.action = Marker.ADD
                marker.pose.orientation.w = 1.0
                # marker.scale.x = 1.2  # Adjust the size of the points
                # marker.scale.y = 1.2
                marker.scale.x = ms *1.2  # Adjust the size of the points
                marker.scale.y = ms *1.2
                marker.color.a = 0.9
                marker.color.r = 0.5
                marker.color.g = 0.5
                marker.color.b = 1

                marker.id = copy.deepcopy(marker_id)
                marker_id += 1

                # Convert the 3D points to Point messages
                n_frontiers = spts.shape[0]
                points_msg = [Point(x=spts[i, 0], y=spts[i, 1], z=spts[i, 2]) for i in range(n_frontiers)]
                marker.points = points_msg
                marker_array.markers.append(marker)

        if do_normals:
            arrowlen = 0.6 * ms
            pts1 = transformPoints(smap.surfel_points, T_vis)
            pts2 = transformPoints(smap.surfel_points + smap.surfel_normals * arrowlen, T_vis)
            if not smap.surfel_points is None:
                for i in range(smap.surfel_points.shape[0]):
                    marker = Marker()
                    marker.header.frame_id = self.odom_frame  # Change this frame_id if necessary
                    marker.header.stamp = rospy.Time.now()
                    marker.type = Marker.ARROW
                    marker.action = Marker.ADD
                    marker.id = marker_id
                    marker_id += 1

                    # Set the scale
                    marker.scale.x = ms *0.25
                    marker.scale.y = ms *0.5
                    marker.scale.z = ms *0.5

                    marker.color.a = 1
                    marker.color.r = 1.0
                    marker.color.g = 0.5
                    marker.color.b = 0.0
                    if alternative_look:
                        marker.color.r = 0.7
                        marker.color.g = 0.0
                        marker.color.b = 1.0

                    pt1 = pts1[i]
                    pt2 = pts2[i]
                    points_msg = [Point(x=pt1[0], y=pt1[1], z=pt1[2]), Point(x=pt2[0], y=pt2[1], z=pt2[2])]
                    marker.points = points_msg

                    # Add the marker to the MarkerArray
                    marker_array.markers.append(marker)

        if do_keyframes:
            n_kframes = len(smap.visual_keyframes)
            if n_kframes > 0:
                kframes_pts = np.array([kf.position for kf in smap.visual_keyframes])
                kframes_pts = transformPoints(kframes_pts, T_vis)
                for i in range(n_kframes):
                    marker = Marker()
                    marker.header.frame_id = self.odom_frame  # Change this frame_id if necessary
                    marker.header.stamp = rospy.Time.now()
                    marker.type = Marker.ARROW
                    marker.action = Marker.ADD
                    marker.id = marker_id
                    marker_id += 1

                    # # Set the position (sphere center)
                    # marker.pose.position.x = kframes_pts[i][0]
                    # marker.pose.position.y = kframes_pts[i][1]
                    # marker.pose.position.z = kframes_pts[i][2]

                    # Set the scale
                    marker.scale.x = ms *0.4
                    marker.scale.y = ms *2.0
                    marker.scale.z = ms *1.0

                    marker.color.a = 1
                    marker.color.r = 0.0
                    marker.color.g = 0.0
                    marker.color.b = 0.0
                    if alternative_look:
                        marker.color.r = 1.0
                        marker.color.g = 0.0
                        marker.color.b = 1.0

                    arrowlen = 2
                    map_heading = transformationMatrixToHeading(T_vis)
                    xbonus = arrowlen * np.cos(smap.visual_keyframes[i].heading + map_heading)
                    ybonus = arrowlen * np.sin(smap.visual_keyframes[i].heading + map_heading)
                    points_msg = [Point(x=kframes_pts[i][0], y=kframes_pts[i][1], z=kframes_pts[i][2]), Point(x=kframes_pts[i][0]+xbonus, y=kframes_pts[i][1]+ybonus, z=kframes_pts[i][2])]
                    marker.points = points_msg

                    # Add the marker to the MarkerArray
                    marker_array.markers.append(marker)

        if do_spheres:
            for i in range(smap.points.shape[0]):
                marker = Marker()
                marker.header.frame_id = self.odom_frame  # Change this frame_id if necessary
                marker.header.stamp = rospy.Time.now()
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.id = marker_id
                marker_id += 1

                # Set the position (sphere center)
                marker.pose.position.x = pts[i][0]
                marker.pose.position.y = pts[i][1]
                marker.pose.position.z = pts[i][2]

                # Set the scale (sphere radius)
                marker.scale.x = 2 * smap.radii[i]
                marker.scale.y = 2 * smap.radii[i]
                marker.scale.z = 2 * smap.radii[i]

                marker.color.a = 0.1
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                if alternative_look:
                    marker.color.a = 0.05
                    marker.color.r = 0.0
                    marker.color.g = 0.0
                    marker.color.b = 1.0

                # Add the marker to the MarkerArray
                marker_array.markers.append(marker)

        return True
    # # #}

    def get_freespace_polyhedron_markers(self, T_global_to_own_origin, T_global_to_fcu, mesh_pts_smap_frame, tris, protected_pts_smap_frame = None):# # #{
        # cam_pos_global = T[:3,3].T

        # T_delta_odom = np.linalg.inv(T_global_to_own_origin) @ T_global_to_fcu
        T_delta_odom = T_global_to_fcu
        T_orig_to_current_cam = (T_delta_odom @ self.T_fcu_to_imu @ self.T_imu_to_cam)

        mesh_pts_global = transformPoints(mesh_pts_smap_frame, T_orig_to_current_cam)

        marker_array = MarkerArray()
        line_marker = Marker()
        line_marker.header.frame_id = self.odom_frame  # Set your desired frame_id
        line_marker.type = Marker.LINE_LIST
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.2 
        line_marker.color.a = 1.0  
        line_marker.color.r = 0.0  
        line_marker.color.g = 1.0  
        line_marker.color.b = 0.0  

        for tri in tris:
            point1 = Point()
            point2 = Point()
            point3 = Point()

            point1.x = mesh_pts_global[tri[0],0]
            point1.y = mesh_pts_global[tri[0],1]
            point1.z = mesh_pts_global[tri[0],2]

            point2.x = mesh_pts_global[tri[1],0]
            point2.y = mesh_pts_global[tri[1],1]
            point2.z = mesh_pts_global[tri[1],2]

            point3.x = mesh_pts_global[tri[2],0]
            point3.y = mesh_pts_global[tri[2],1]
            point3.z = mesh_pts_global[tri[2],2]

            line_marker.points.append(point1)
            line_marker.points.append(point2)

            line_marker.points.append(point2)
            line_marker.points.append(point3)

            line_marker.points.append(point3)
            line_marker.points.append(point1)

        marker_array.markers.append(line_marker)
        return marker_array
# # #}


