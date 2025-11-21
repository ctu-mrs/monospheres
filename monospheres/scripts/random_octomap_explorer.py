#!/usr/bin/env python3

import rospy
import random
import math
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point, PointStamped
from std_msgs.msg import String
from mrs_msgs.srv import Vec4, Vec4Request
import numpy as np
# mrs_modules_msgs/OctomapPlannerDiagnostics
from mrs_modules_msgs.msg import OctomapPlannerDiagnostics

import tf2_ros
import tf2_geometry_msgs
# import geometry_msgs.msg

class RandomExplorer:
    def __init__(self):
        rospy.init_node("random_explorer")

        # Subscribers
        self.marker_sub = rospy.Subscriber("/uav1/octomap_global_vis/free_cells_vis_array", MarkerArray, self.marker_callback)
        self.nav_diagnostics_sub = rospy.Subscriber("/uav1/octomap_planner/diagnostics", OctomapPlannerDiagnostics, self.nav_diagnostics_callback)

        # Service Client for /goto
        self.gotosrv_name = "/uav1/octomap_planner/goto"
        self.goto_client = rospy.ServiceProxy(self.gotosrv_name, Vec4)

        # Timer for periodic exploration
        self.timer = rospy.Timer(rospy.Duration(1.0), self.explore)

        # Storage
        # self.free_space_markers = []
        self.free_space_markers_pts = []
        self.nav_status = "IDLE"
        self.goal_active = False
        self.last_sent_goal_time = rospy.Time.now().to_sec()

        # Exploration bounds
        self.local_random_sampling_dist = 20
        self.goal_timeout = 20
        self.safety_area_enabled = rospy.get_param("exploration_safety_area/enabled")
        self.safety_area_x_min = rospy.get_param("exploration_safety_area/x_min")
        self.safety_area_x_max = rospy.get_param("exploration_safety_area/x_max")
        self.safety_area_y_min = rospy.get_param("exploration_safety_area/y_min")
        self.safety_area_y_max = rospy.get_param("exploration_safety_area/y_max")
        self.safety_area_z_min = rospy.get_param("exploration_safety_area/z_min")
        self.safety_area_z_max = rospy.get_param("exploration_safety_area/z_max")


        boundmax = 100
        # self.x_min, self.x_max = -boundmax, boundmax
        # self.y_min, self.y_max = -boundmax, boundmax
        # self.z_min = 0  # Assume exploration at z = 1.0
        # self.z_max = 6  # Assume exploration at z = 1.0
        self.free_threshold = 1  # Distance to consider near a free space marker
        # self.world_frame = "uav1/world_origin"
        # self.world_frame = "uav1/vio_origin"
        self.world_frame = "global"
        self.octomap_frame = "uav1/vio_origin"
        self.markers_frame = "uav1/vio_origin"

        rospy.loginfo("Random Explorer node started.")


        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    def transform_point(self, point, from_frame, to_frame):
        stampoint = PointStamped()
        stampoint.point = point
        stampoint.header.frame_id = from_frame
        stampoint.header.stamp = rospy.Time.now()

        try:
            # Wait for transform to become available
            transform = self.tf_buffer.lookup_transform(to_frame, from_frame, rospy.Time(0), rospy.Duration(1.0))
            
            # Transform the point
            transformed_point = tf2_geometry_msgs.do_transform_point(stampoint, transform)
            return transformed_point.point
    
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"Transform error: {e}")
            return None

    def get_fcu_world_position(self):
        try:
            # Lookup the transformation from world_origin to fcu
            transform = self.tf_buffer.lookup_transform(self.world_frame, 'uav1/fcu', rospy.Time(0), rospy.Duration(1.0))
    
            # Extract position
            position = transform.transform.translation
            print(f"Position of /uav1/fcu in {self.world_frame}: x={position.x}, y={position.y}, z={position.z}")
    
            return position
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"Could not get transform: {e}")
            return None


    def marker_callback(self, msg):
        """Updates the list of free-space markers."""
        # self.free_space_markers = [marker.pose.position for marker in msg.markers]
        # print(msg.markers[0].header.frame_id)
        self.markers_frame = msg.markers[0].header.frame_id
        self.free_space_markers_pts = []
        for i in range(len(msg.markers)):
            # print(len(msg.markers[i].points))
            self.free_space_markers_pts += msg.markers[i].points
        rospy.loginfo("Received %d free-space marker pts", len(self.free_space_markers_pts))

    def nav_diagnostics_callback(self, msg):
        """Checks if the robot has reached its goal or failed."""
        # self.nav_status = msg.data
        # if self.nav_status in ["REACHED", "FAILED"]:
        #     rospy.loginfo("Goal %s. Selecting a new goal...", self.nav_status)
        #     self.goal_active = False

        self.goal_active = not msg.idle

        # self.nav_status = msg.idle
        # if self.nav_status in ["REACHED", "FAILED"]:
        #     rospy.loginfo("Goal %s. Selecting a new goal...", self.nav_status)
        #     self.goal_active = False

    def explore(self, event):
        """Selects a random goal and sends it to the robot if no goal is active."""
        if not self.free_space_markers_pts:
            return

        time_since_last_sent = rospy.Time.now().to_sec() - self.last_sent_goal_time

        if self.goal_active and time_since_last_sent < self.goal_timeout:
            return

        # First get pos of robot
        robot_pos = self.get_fcu_world_position()
        if robot_pos == None:
            return

        if not self.is_inside_target_area(robot_pos.x, robot_pos.y, robot_pos.z):
            rospy.logwarn_throttle(1, "Robot is outside exploration area!")

        heading = np.random.rand() * 2 * 3.14159
        goal = self.sample_free_point(robot_pos)

        if goal:
            rospy.loginfo("Trying to send goal")
            rospy.loginfo("Goal in orig_frame: (%.2f, %.2f, %.2f)", goal.x, goal.y, goal.z)
            transformed_goal = self.transform_point(goal, self.world_frame , self.octomap_frame)
            try:
                rospy.wait_for_service(self.gotosrv_name, timeout=2.0)
                set_transformed_goal = Vec4Request(goal=[transformed_goal.x, transformed_goal.y, transformed_goal.z, heading])
                response = self.goto_client(set_transformed_goal)
                if response.success:
                    rospy.loginfo("Sent transformed_goal: (%.2f, %.2f, %.2f)", transformed_goal.x, transformed_goal.y, transformed_goal.z)
                    self.goal_active = True
                    self.last_sent_goal_time = rospy.Time.now().to_sec()
                else:
                    rospy.warn("Failed to send goal.")
            except rospy.ServiceException as e:
                rospy.logerr("Service call failed: %s", str(e))

    def is_inside_target_area(self, x, y, z):
        if x > self.safety_area_x_max or x < self.safety_area_x_min:
            return False
        if y > self.safety_area_y_max or y < self.safety_area_y_min:
            return False
        if z > self.safety_area_z_max or z < self.safety_area_z_min:
            return False
        return True

    def sample_free_point(self, robot_pos = None):
        """Samples a random free point within bounds that is near a free-space marker."""
        for _ in range(100):  # Try up to 100 times
            x = 0
            y = 0
            z = 0

            if not robot_pos is None:
                x = robot_pos.x + random.uniform(-self.local_random_sampling_dist, self.local_random_sampling_dist)
                y = robot_pos.y + random.uniform(-self.local_random_sampling_dist, self.local_random_sampling_dist)
                z = robot_pos.z + random.uniform(-2, 2)

                if self.safety_area_enabled:
                    z = random.uniform(self.safety_area_z_min, self.safety_area_z_max)
                # z = random.uniform(2, 50)
            # else:
            #     x = random.uniform(self.x_min, self.x_max)
            #     y = random.uniform(self.y_min, self.y_max)
            #     z = self.z_level + random.uniform(-2, 2)

            # CHECK SAFETY AREA
            if self.safety_area_enabled:
                if not self.is_inside_target_area(x, y, z):
                    continue


            # if self.is_near_free_marker(x, y, z, self.world_frame):
            if True:
                goal = Point(x=x, y=y, z=z)
                return goal
        rospy.logwarn("Could not find a free random point.")
        return None

    def is_near_free_marker(self, x, y, z, pt_frame):
        """Checks if the sampled point is near any free-space marker."""
        pt = Point(x=x, y=y, z=z)
        trans_pt = self.transform_point(pt, pt_frame,  self.markers_frame )
        x = trans_pt.x
        y = trans_pt.y
        z = trans_pt.z

        mindist = 10000
        for marker in self.free_space_markers_pts:
            # dist = math.sqrt((marker.x - x) ** 2 + (marker.y - y) ** 2 + (marker.z - z) ** 2)
            dist = math.sqrt((marker.x - x) ** 2 + (marker.y - y) ** 2 + (marker.z - z) ** 2)
            if dist < self.free_threshold:
                return True
            elif dist < mindist:
                mindist = dist
        # rospy.loginfo("best dist: " + str(mindist))
        return False

if __name__ == "__main__":
    try:
        RandomExplorer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
