#!/usr/bin/env python3
"""Re-publish a PointCloud2 transformed into another TF frame (used to feed OpenVINS SLAM points
to the octomap baseline)."""

import rclpy
import rclpy.duration
import rclpy.time
import tf2_ros
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud


class PointCloudTransformer(Node):
    def __init__(self):
        super().__init__("pointcloud_transformer")

        self.input_topic = self.declare_parameter("input_topic", "/uav1/open_vins/points_slam").value
        self.output_topic = self.declare_parameter("output_topic", "/uav1/open_vins/points_slam/transformed").value
        self.target_frame = self.declare_parameter("target_frame", "uav1/fcu").value

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.subscriber = self.create_subscription(PointCloud2, self.input_topic, self.callback, qos_profile_sensor_data)
        self.publisher = self.create_publisher(PointCloud2, self.output_topic, 10)

        self.get_logger().info(f"Listening for point clouds on {self.input_topic} and transforming to {self.target_frame}")

    def callback(self, cloud_msg):
        try:
            # latest available transform
            transform = self.tf_buffer.lookup_transform(self.target_frame,
                                                        cloud_msg.header.frame_id,
                                                        rclpy.time.Time(),
                                                        timeout=rclpy.duration.Duration(seconds=1.0))
            transformed_cloud = do_transform_cloud(cloud_msg, transform)
            transformed_cloud.header.frame_id = self.target_frame
            self.publisher.publish(transformed_cloud)

        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.ConnectivityException) as e:
            self.get_logger().warn(f"{type(e).__name__}: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudTransformer()
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
