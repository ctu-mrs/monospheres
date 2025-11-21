#!/usr/bin/env python3

import rospy
import tf2_ros
import tf2_sensor_msgs.tf2_sensor_msgs
from sensor_msgs.msg import PointCloud2

class PointCloudTransformer:
    def __init__(self):
        rospy.init_node("pointcloud_transformer", anonymous=True)

        self.input_topic = rospy.get_param("~input_topic", "/ov_msckf/points_slam")
        self.output_topic = rospy.get_param("~output_topic", "/ov_msckf/points_slam/transformed")
        self.target_frame = rospy.get_param("~target_frame", "uav1/fcu")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.subscriber = rospy.Subscriber(self.input_topic, PointCloud2, self.callback)
        self.publisher = rospy.Publisher(self.output_topic, PointCloud2, queue_size=10)

        rospy.loginfo(f"Listening for point clouds on {self.input_topic} and transforming to {self.target_frame}")

    def callback(self, cloud_msg):
        try:
            # Get the latest transformation
            transform = self.tf_buffer.lookup_transform(self.target_frame, 
                                                        cloud_msg.header.frame_id,
                                                        rospy.Time(0),
                                                        rospy.Duration(1.0))
            
            # Transform the PointCloud2
            transformed_cloud = tf2_sensor_msgs.do_transform_cloud(cloud_msg, transform)
            transformed_cloud.header.frame_id = self.target_frame  # Update frame ID

            # Publish transformed point cloud
            self.publisher.publish(transformed_cloud)

        except tf2_ros.LookupException as e:
            rospy.logwarn(f"LookupException: {e}")
        except tf2_ros.ExtrapolationException as e:
            rospy.logwarn(f"ExtrapolationException: {e}")

if __name__ == "__main__":
    try:
        PointCloudTransformer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
