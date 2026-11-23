#!/usr/bin/env python3

import rclpy
import rclpy.duration
import rclpy.time
from rclpy.node import Node
import random
import math
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point, PointStamped
from std_msgs.msg import String
from mrs_msgs.srv import Vec4
import numpy as np
# mrs_modules_msgs/OctomapPlannerDiagnostics
from mrs_modules_msgs.msg import OctomapPlannerDiagnostics

import tf2_ros
import tf2_geometry_msgs
# import geometry_msgs.msg

class RandomExplorer(Node):
    def __init__(self):
        super().__init__("random_explorer",
                         allow_undeclared_parameters=True,
                         automatically_declare_parameters_from_overrides=True)

        # Subscribers
        self.marker_sub = self.create_subscription(MarkerArray, self._param("free_cells_topic", "/uav1/octomap_global_vis/free_cells_vis_array"), self.marker_callback, 10)
        self.nav_diagnostics_sub = self.create_subscription(OctomapPlannerDiagnostics, self._param("planner_diagnostics_topic", "/uav1/octomap_planner/diagnostics"), self.nav_diagnostics_callback, 10)

        # Service Client for /goto
        self.gotosrv_name = self._param("goto_service", "/uav1/octomap_planner/goto")
        self.goto_client = self.create_client(Vec4, self.gotosrv_name)
        self.goto_future = None

        # Timer for periodic exploration
        self.timer = self.create_timer(1.0, self.explore)

        # Storage
        # self.free_space_markers = []
        self.free_space_markers_pts = []
        self.nav_status = "IDLE"
        self.goal_active = False
        self.last_sent_goal_time = self._now_sec()

        # Exploration bounds
        self.local_random_sampling_dist = 20
        self.goal_timeout = 20
        self.safety_area_enabled = self._param("exploration_safety_area/enabled")
        self.safety_area_x_min = self._param("exploration_safety_area/x_min")
        self.safety_area_x_max = self._param("exploration_safety_area/x_max")
        self.safety_area_y_min = self._param("exploration_safety_area/y_min")
        self.safety_area_y_max = self._param("exploration_safety_area/y_max")
        self.safety_area_z_min = self._param("exploration_safety_area/z_min")
        self.safety_area_z_max = self._param("exploration_safety_area/z_max")


        boundmax = 100
        # self.x_min, self.x_max = -boundmax, boundmax
        # self.y_min, self.y_max = -boundmax, boundmax
        # self.z_min = 0  # Assume exploration at z = 1.0
        # self.z_max = 6  # Assume exploration at z = 1.0
        self.free_threshold = 1  # Distance to consider near a free space marker
        # self.world_frame = "uav1/world_origin"
        # self.world_frame = "uav1/vio_origin"
        self.world_frame = self._param("world_frame", "global")
        self.octomap_frame = self._param("octomap_frame", "uav1/vio_origin")
        self.markers_frame = "uav1/vio_origin"

        self.get_logger().info("Random Explorer node started.")


        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def _param(self, name, default=None):
        """rospy.get_param equivalent ('a/b' -> 'a.b'); raises KeyError if missing and no default."""
        ros2_name = name.replace('/', '.')
        if self.has_parameter(ros2_name) and self.get_parameter(ros2_name).value is not None:
            return self.get_parameter(ros2_name).value
        if default is None:
            raise KeyError("parameter '%s' is not set" % name)
        return default

    def _now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def transform_point(self, point, from_frame, to_frame):
        stampoint = PointStamped()
        stampoint.point = point
        stampoint.header.frame_id = from_frame
        stampoint.header.stamp = self.get_clock().now().to_msg()

        try:
            # Wait for transform to become available
            transform = self.tf_buffer.lookup_transform(to_frame, from_frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0))
            
            # Transform the point
            transformed_point = tf2_geometry_msgs.do_transform_point(stampoint, transform)
            return transformed_point.point
    
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f"Transform error: {e}")
            return None

    def get_fcu_world_position(self):
        try:
            # Lookup the transformation from world_origin to fcu
            transform = self.tf_buffer.lookup_transform(self.world_frame, 'uav1/fcu', rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0))
    
            # Extract position
            position = transform.transform.translation
            print(f"Position of /uav1/fcu in {self.world_frame}: x={position.x}, y={position.y}, z={position.z}")
    
            return position
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f"Could not get transform: {e}")
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
        self.get_logger().info("Received %d free-space marker pts" % len(self.free_space_markers_pts))

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

    def explore(self):
        """Selects a random goal and sends it to the robot if no goal is active."""
        if not self.free_space_markers_pts:
            return

        time_since_last_sent = self._now_sec() - self.last_sent_goal_time

        if self.goal_active and time_since_last_sent < self.goal_timeout:
            return

        # First get pos of robot
        robot_pos = self.get_fcu_world_position()
        if robot_pos == None:
            return

        if not self.is_inside_target_area(robot_pos.x, robot_pos.y, robot_pos.z):
            self.get_logger().warn("Robot is outside exploration area!", throttle_duration_sec=1.0)

        heading = np.random.rand() * 2 * 3.14159
        goal = self.sample_free_point(robot_pos)

        if goal:
            self.get_logger().info("Trying to send goal")
            self.get_logger().info("Goal in orig_frame: (%.2f, %.2f, %.2f)" % (goal.x, goal.y, goal.z))
            transformed_goal = self.transform_point(goal, self.world_frame , self.octomap_frame)
            if transformed_goal is None:
                return
            if self.goto_future is not None and not self.goto_future.done():
                return
            if not self.goto_client.wait_for_service(timeout_sec=2.0):
                self.get_logger().error("Service %s not available" % self.gotosrv_name)
                return
            req = Vec4.Request(goal=[transformed_goal.x, transformed_goal.y, transformed_goal.z, heading])
            self.goto_future = self.goto_client.call_async(req)
            self.goto_future.add_done_callback(lambda f, g=transformed_goal: self.goto_done(f, g))

    def goto_done(self, future, transformed_goal):
        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error("Service call failed: %s" % str(e))
            return
        if response.success:
            self.get_logger().info("Sent transformed_goal: (%.2f, %.2f, %.2f)" % (transformed_goal.x, transformed_goal.y, transformed_goal.z))
            self.goal_active = True
            self.last_sent_goal_time = self._now_sec()
        else:
            self.get_logger().warn("Failed to send goal.")

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
        self.get_logger().warn("Could not find a free random point.")
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

def main(args=None):
    rclpy.init(args=args)
    node = RandomExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
