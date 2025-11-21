/* includes //{ */

#include <ros/ros.h>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>

#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/TwistWithCovarianceStamped.h>
#include <geometry_msgs/Vector3.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/String.h>

#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/static_transform_broadcaster.h>

#include <mrs_lib/param_loader.h>
#include <mrs_lib/transformer.h>
#include <mrs_lib/mutex.h>
#include <mrs_lib/attitude_converter.h>
#include <mrs_lib/msg_extractor.h>

#include <nodelet/nodelet.h>

#include <pluginlib/class_list_macros.h>

//}

namespace vins_republisher
{

/* class VinsRepublisher //{ */

class VinsRepublisher : public nodelet::Nodelet {
public:
  virtual void onInit();

private:
  /* flags */
  bool is_initialized_        = false;
  bool _rate_limiter_enabled_ = false;

  /* ros parameters */
  std::string _uav_name_;
  bool        _rotate_velocity_ = false;
  /* std::string _camera_frame_; */
  std::string _fcu_frame_;
  std::string _mrs_vins_world_frame_;
  std::string _vins_fcu_frame_;
  double      _rate_limiter_rate_;

  bool validateOdometry(const nav_msgs::Odometry &odometry);

  // | ------------------------ callbacks ----------------------- |
  ros::Subscriber subscriber_vins_;
  void            odometryCallback(const nav_msgs::OdometryConstPtr &odom);

  ros::Publisher publisher_odom_;
  ros::Time      publisher_odom_last_published_;

  /* transformation handler */
  mrs_lib::Transformer transformer_;
};

//}

/* onInit() //{ */

void VinsRepublisher::onInit() {
  const std::string node_name("VinsRepublisher");

  /* obtain node handle */
  /* ros::NodeHandle nh("~"); */
  ros::NodeHandle nh_ = nodelet::Nodelet::getMTPrivateNodeHandle();

  ROS_INFO("[%s]: Initializing", node_name.c_str());

  /* waits for the ROS to publish clock */
  ros::Time::waitForValid();

  publisher_odom_last_published_ = ros::Time(0);

  // | ---------- loading ros parameters using mrs_lib ---------- |
  ROS_INFO("[%s]: loading parameters using ParamLoader", node_name.c_str());
  ROS_INFO("pes");

  mrs_lib::ParamLoader param_loader(nh_, node_name);

  param_loader.loadParam("uav_name", _uav_name_);
  /* param_loader.loadParam("rotate_velocity", _rotate_velocity_); */
  _rotate_velocity_ = true;
  param_loader.loadParam("rotate_velocity", _rotate_velocity_);
  param_loader.loadParam("rate_limiter/enabled", _rate_limiter_enabled_);
  param_loader.loadParam("rate_limiter/max_rate", _rate_limiter_rate_);
  if (_rate_limiter_rate_ <= 1e-3) {
    ROS_ERROR("[%s]: the rate limit has to be > 0", ros::this_node::getName().c_str());
    ros::shutdown();
  }

  param_loader.loadParam("fcu_frame", _fcu_frame_);
  /* param_loader.loadParam("camera_frame", _camera_frame_); */
  param_loader.loadParam("mrs_vins_world_frame", _mrs_vins_world_frame_);
  param_loader.loadParam("vins_fcu_frame", _vins_fcu_frame_);

  if (!param_loader.loadedSuccessfully()) {
    ROS_ERROR("[%s]: parameter loading failure", node_name.c_str());
    ros::shutdown();
  }

  /* transformation handler */
  transformer_ = mrs_lib::Transformer("VinsRepublisher");

  // | ----------------------- subscribers ---------------------- |

  subscriber_vins_ = nh_.subscribe("vins_odom_in", 10, &VinsRepublisher::odometryCallback, this, ros::TransportHints().tcpNoDelay());

  // | ----------------------- publishers ----------------------- |

  publisher_odom_ = nh_.advertise<nav_msgs::Odometry>("vins_odom_out", 10);

  is_initialized_ = true;

  ROS_INFO_ONCE("[%s]: initialized", node_name.c_str());
}
//}

/* validateOdometry() //{ */

bool VinsRepublisher::validateOdometry(const nav_msgs::Odometry &odometry) {

  // check position

  if (!std::isfinite(odometry.pose.pose.position.x)) {
    ROS_ERROR_THROTTLE(1.0, "[ControlManager]: NaN detected in variable 'odometry.pose.pose.position.x'!!!");
    return false;
  }

  if (!std::isfinite(odometry.pose.pose.position.y)) {
    ROS_ERROR_THROTTLE(1.0, "[ControlManager]: NaN detected in variable 'odometry.pose.pose.position.y'!!!");
    return false;
  }

  if (!std::isfinite(odometry.pose.pose.position.z)) {
    ROS_ERROR_THROTTLE(1.0, "[ControlManager]: NaN detected in variable 'odometry.pose.pose.position.z'!!!");
    return false;
  }

  // check orientation

  if (!std::isfinite(odometry.pose.pose.orientation.x)) {
    ROS_ERROR_THROTTLE(1.0, "[ControlManager]: NaN detected in variable 'odometry.pose.pose.orientation.x'!!!");
    return false;
  }

  if (!std::isfinite(odometry.pose.pose.orientation.y)) {
    ROS_ERROR_THROTTLE(1.0, "[ControlManager]: NaN detected in variable 'odometry.pose.pose.orientation.y'!!!");
    return false;
  }

  if (!std::isfinite(odometry.pose.pose.orientation.z)) {
    ROS_ERROR_THROTTLE(1.0, "[ControlManager]: NaN detected in variable 'odometry.pose.pose.orientation.z'!!!");
    return false;
  }

  if (!std::isfinite(odometry.pose.pose.orientation.w)) {
    ROS_ERROR_THROTTLE(1.0, "[ControlManager]: NaN detected in variable 'odometry.pose.pose.orientation.w'!!!");
    return false;
  }

  // check if the quaternion is sound

  if (fabs(Eigen::Vector4d(odometry.pose.pose.orientation.x, odometry.pose.pose.orientation.y, odometry.pose.pose.orientation.z,
                           odometry.pose.pose.orientation.w)
               .norm() -
           1.0) > 1e-2) {
    ROS_ERROR_THROTTLE(1.0, "[ControlManager]: orientation is not sound!!!");
    return false;
  }

  // check velocity

  if (!std::isfinite(odometry.twist.twist.linear.x)) {
    ROS_ERROR_THROTTLE(1.0, "[ControlManager]: NaN detected in variable 'odometry.twist.twist.linear.x'!!!");
    return false;
  }

  if (!std::isfinite(odometry.twist.twist.linear.y)) {
    ROS_ERROR_THROTTLE(1.0, "[ControlManager]: NaN detected in variable 'odometry.twist.twist.linear.y'!!!");
    return false;
  }

  if (!std::isfinite(odometry.twist.twist.linear.z)) {
    ROS_ERROR_THROTTLE(1.0, "[ControlManager]: NaN detected in variable 'odometry.twist.twist.linear.z'!!!");
    return false;
  }

  return true;
}

//}

/* odometryCallback() //{ */

void VinsRepublisher::odometryCallback(const nav_msgs::OdometryConstPtr &odom) {

  if (!is_initialized_) {
    return;
  }

  if (!validateOdometry(*odom)) {
    ROS_ERROR("[VinsRepublisher]: input odometry is not numerically valid");
    return;
  }

  ROS_DEBUG("[VinsRepublisher]: %d ", odom->header.seq);
  ROS_DEBUG("[VinsRepublisher]: %d ", odom->header.stamp.sec);
  ROS_DEBUG("[VinsRepublisher]: %d ", odom->header.stamp.nsec);

  ROS_DEBUG("[now]: %f", ros::Time::now().toSec());
  ROS_DEBUG("[last published]: %f", publisher_odom_last_published_.toSec());
  ROS_DEBUG("[fabs diff]: %f", fabs((ros::Time::now() - publisher_odom_last_published_).toSec()));
  ROS_DEBUG("[diff]: %f", (ros::Time::now() - publisher_odom_last_published_).toSec());
  ROS_DEBUG("[rate_limiter]: %f", 1.0 / _rate_limiter_rate_);
  if (_rate_limiter_enabled_ && fabs((ros::Time::now() - publisher_odom_last_published_).toSec()) < (1.0 / (_rate_limiter_rate_))) {
    ROS_DEBUG("[%s]: skipping over", ros::this_node::getName().c_str());
    return;
  }

  ROS_DEBUG("[%s]: publishing", ros::this_node::getName().c_str());

  geometry_msgs::PoseStamped vins_pose;
  vins_pose.header = odom->header;
  vins_pose.pose   = mrs_lib::getPose(odom);

  geometry_msgs::Vector3Stamped vins_velocity;
  vins_velocity.header = odom->header;
  vins_velocity.vector = odom->twist.twist.linear;

  geometry_msgs::Vector3Stamped vins_ang_velocity;
  vins_ang_velocity.header = odom->header;
  vins_ang_velocity.vector = odom->twist.twist.angular;

  geometry_msgs::TransformStamped tf;

  /* get the transform from mrs_vins_world to vins_world //{ */

  /* Vins-mono uses different coordinate system. The y-axis is front, x-axis is right, z-axis is up */
  /* This transformation rotates in yaw of 90 deg */

  {
    auto res = transformer_.getTransform(odom->header.frame_id, _mrs_vins_world_frame_, odom->header.stamp);

    if (!res) {
      ROS_WARN_THROTTLE(1.0, "[%s]: could not find transform from '%s' to '%s' at time '%f'", ros::this_node::getName().c_str(), _mrs_vins_world_frame_.c_str(),
                        odom->header.frame_id.c_str(), odom->header.stamp.toSec());
      return;
    }

    tf = res.value();
  }

  //}

  /* transform the vins pose to mrs_world_frame //{ */

  geometry_msgs::PoseStamped vins_pose_mrs_world;

  {
    auto res = transformer_.transform(vins_pose, tf);

    if (!res) {
      ROS_WARN_THROTTLE(1.0, "[%s]: could not transform vins pose to '%s'", ros::this_node::getName().c_str(), _mrs_vins_world_frame_.c_str());
      return;
    }

    vins_pose_mrs_world = res.value();
  }

  Eigen::Matrix3d rotation;

  {
    // transform from the VINS body frame to the UAV FCU frame
    auto res = transformer_.getTransform(_fcu_frame_, _vins_fcu_frame_, odom->header.stamp);
    if (!res) {
      ROS_WARN_THROTTLE(1.0, "[%s]: could not find transform from '%s' to '%s'", ros::this_node::getName().c_str(), _fcu_frame_.c_str(),
                        _vins_fcu_frame_.c_str());
      return;
    }

    rotation                             = mrs_lib::AttitudeConverter(res.value().transform.rotation);
    Eigen::Matrix3d original_orientation = mrs_lib::AttitudeConverter(vins_pose_mrs_world.pose.orientation);
    vins_pose_mrs_world.pose.orientation = mrs_lib::AttitudeConverter(original_orientation * rotation);
    Eigen::Vector3d t;
    t << res.value().transform.translation.x, res.value().transform.translation.y, res.value().transform.translation.z;
    Eigen::Vector3d translation         = rotation.transpose() * t;
    vins_pose_mrs_world.pose.position.x = vins_pose_mrs_world.pose.position.x + translation(0);
    vins_pose_mrs_world.pose.position.y = vins_pose_mrs_world.pose.position.y + translation(1);
    vins_pose_mrs_world.pose.position.z = vins_pose_mrs_world.pose.position.z + translation(2);
  }

  //}

  geometry_msgs::Vector3Stamped vins_velocity_mrs_world;

  /* transform the vins velocity to mrs_world_frame //{ */

  {

    auto res = transformer_.transform(vins_velocity, tf);

    if (!res) {
      ROS_WARN_THROTTLE(1.0, "[%s]: could not transform vins velocity to '%s'", ros::this_node::getName().c_str(), _mrs_vins_world_frame_.c_str());
      return;
    }

    vins_velocity_mrs_world = res.value();
    if (_rotate_velocity_) {
      Eigen::Vector3d v2;
      v2 << vins_velocity.vector.x, vins_velocity.vector.y, vins_velocity.vector.z;
      v2                               = rotation.transpose() * v2;
      vins_velocity_mrs_world.vector.x = v2(0);
      vins_velocity_mrs_world.vector.y = v2(1);
      vins_velocity_mrs_world.vector.z = v2(2);
    }
  }

  //}

  geometry_msgs::Vector3Stamped vins_ang_velocity_mrs_world;

  /* transform the vins angular velocity to mrs_world_frame //{ */

  {
    auto res = transformer_.transform(vins_ang_velocity, tf);

    if (!res) {
      ROS_WARN_THROTTLE(1.0, "[%s]: could not transform vins angular velocity to '%s'", ros::this_node::getName().c_str(), _mrs_vins_world_frame_.c_str());
      return;
    }

    vins_ang_velocity_mrs_world = res.value();
    if (_rotate_velocity_) {
      Eigen::Vector3d v2;
      v2 << vins_ang_velocity.vector.x, vins_ang_velocity.vector.y, vins_ang_velocity.vector.z;
      v2                                   = rotation.transpose() * v2;
      vins_ang_velocity_mrs_world.vector.x = v2(0);
      vins_ang_velocity_mrs_world.vector.y = v2(1);
      vins_ang_velocity_mrs_world.vector.z = v2(2);
    }
  }

  //}

  /* geometry_msgs::Vector3 camera_to_fcu_translation; */

  /* /1* find transform from camera frame to fcu frame //{ *1/ */

  /* { */
  /*   auto res = transformer_.getTransform(_camera_frame_, _fcu_frame_, odom->header.stamp); */

  /*   if (!res) { */

  /*     ROS_WARN_THROTTLE(1.0, "[%s]: could not find transform from '%s' to '%s'", ros::this_node::getName().c_str(), _camera_frame_.c_str(), */
  /*                       _fcu_frame_.c_str()); */
  /*     return; */
  /*   } */

  /*   camera_to_fcu_translation = res.value().getTransform().transform.translation; */
  /* } */

  /* //} */

  // fill the new transformed odom message
  nav_msgs::Odometry odom_trans;

  odom_trans.header.stamp    = odom->header.stamp;
  odom_trans.header.frame_id = _mrs_vins_world_frame_;

  odom_trans.pose.pose           = vins_pose_mrs_world.pose;
  odom_trans.twist.twist.linear  = vins_velocity_mrs_world.vector;
  odom_trans.twist.twist.angular = vins_ang_velocity_mrs_world.vector;

  if (!validateOdometry(odom_trans)) {
    ROS_ERROR("[VinsRepublisher]: input odometry is not numerically valid");
    return;
  }

  try {
    publisher_odom_.publish(odom_trans);
    ROS_INFO_THROTTLE(1.0, "[%s]: Publishing", ros::this_node::getName().c_str());
    publisher_odom_last_published_ = ros::Time::now();
  }
  catch (...) {
    ROS_ERROR("exception caught during publishing topic '%s'", publisher_odom_.getTopic().c_str());
  }
}

//}

}  // namespace vins_republisher
/* every nodelet must export its class as nodelet plugin */
PLUGINLIB_EXPORT_CLASS(vins_republisher::VinsRepublisher, nodelet::Nodelet);
