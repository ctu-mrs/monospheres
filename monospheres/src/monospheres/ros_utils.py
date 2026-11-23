"""Small ROS 2 helpers that keep the ROS 1-era module code close to its original shape.

The whole pipeline runs in one rclpy node (created in scripts/ov_extension.py) that is passed
into every module. These helpers wrap the few rospy/tf idioms the modules relied on.
"""
import os

import numpy as np
import rclpy
import rclpy.duration
import rclpy.time
import tf2_ros
from builtin_interfaces.msg import Duration as DurationMsg
from builtin_interfaces.msg import Time as TimeMsg
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation

_REQUIRED = object()


# --PARAMETERS
def get_param(node, name, default=_REQUIRED):
    """rospy.get_param equivalent.

    `name` uses the ROS 1 '/' separator ("local_nav/enabled"); it maps to the ROS 2 dotted name
    ("local_nav.enabled") produced by nested YAML under `ros__parameters`. The node is created
    with automatically_declare_parameters_from_overrides=True, so every YAML key exists as a
    parameter with the type it has in the YAML. A missing parameter raises KeyError unless a
    default is given, as rospy.get_param did.
    """
    ros2_name = name.strip('/').replace('/', '.')
    if node.has_parameter(ros2_name):
        value = node.get_parameter(ros2_name).value
        if value is not None:
            return value
    if default is _REQUIRED:
        raise KeyError("parameter '%s' (ROS 2 name '%s') is not set" % (name, ros2_name))
    return default


# --TIME
def now(node):
    """rospy.Time.now() / rospy.get_rostime(): node clock time as rclpy.time.Time (sim time aware)."""
    return node.get_clock().now()


def now_msg(node):
    """Current node time as a builtin_interfaces/Time, for message headers."""
    return node.get_clock().now().to_msg()


def to_time(t):
    """builtin_interfaces/Time or rclpy Time -> rclpy.time.Time."""
    if isinstance(t, TimeMsg):
        return rclpy.time.Time.from_msg(t)
    return t


def to_sec(x):
    """Seconds as float for an rclpy Time/Duration or a builtin_interfaces Time/Duration message."""
    if isinstance(x, (rclpy.time.Time, rclpy.duration.Duration)):
        return x.nanoseconds * 1e-9
    if isinstance(x, (TimeMsg, DurationMsg)):
        return x.sec + x.nanosec * 1e-9
    raise TypeError("to_sec: unsupported type %s" % type(x))


def time_diff_sec(t1, t2):
    """(t1 - t2) in seconds; accepts any mix of rclpy Time and builtin_interfaces/Time."""
    return (to_time(t1) - to_time(t2)).nanoseconds * 1e-9


# --EXECUTOR / QOS
def own_group():
    """A fresh mutually exclusive callback group. Giving every timer and subscription its own group
    reproduces rospy's one-thread-per-timer/subscriber concurrency under a MultiThreadedExecutor."""
    return MutuallyExclusiveCallbackGroup()


def sensor_qos(depth=10):
    """Best-effort subscriber QoS. It matches both reliable and best-effort publishers, so it is safe
    for sensor/estimator topics whose publisher QoS we do not control."""
    return QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=depth,
                      reliability=ReliabilityPolicy.BEST_EFFORT)


def safe_callback(node, callback):
    """Wrap a timer/subscription callback so that an exception is logged with its traceback instead of
    stopping the executor (which would take down the whole single-node pipeline). rospy subscribers
    behaved like this; rospy timers died individually instead."""
    import functools
    import traceback

    @functools.wraps(callback)
    def wrapped(*args, **kwargs):
        try:
            return callback(*args, **kwargs)
        except Exception:
            node.get_logger().error("exception in callback %s:\n%s" % (getattr(callback, '__qualname__', callback), traceback.format_exc()))
    return wrapped


# --TF
def quaternion_matrix(quat_xyzw):
    """tf.transformations.quaternion_matrix equivalent: 4x4 homogeneous rotation from (x, y, z, w)."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    return T


class TfListener:
    """Minimal stand-in for the ROS 1 tf.TransformListener API used by the modules."""

    def __init__(self, node, cache_time_sec=30.0):
        self.node = node
        self.buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=cache_time_sec))
        self.listener = tf2_ros.TransformListener(self.buffer, node)

    def lookupTransform(self, target_frame, source_frame, time=None):
        """Latest available transform (ROS 1 rospy.Time(0) semantics) unless a time is given.
        Returns ((x, y, z), (qx, qy, qz, qw)) like the ROS 1 API."""
        if time is None:
            time = rclpy.time.Time()
        tf = self.buffer.lookup_transform(target_frame, source_frame, to_time(time))
        t = tf.transform.translation
        q = tf.transform.rotation
        return (t.x, t.y, t.z), (q.x, q.y, q.z, q.w)

    def waitForTransform(self, target_frame, source_frame, time=None, timeout_sec=4.0):
        """Block until the transform is available. The executor must be spinning in another thread."""
        if time is None:
            time = rclpy.time.Time()
        deadline = _wall_time() + timeout_sec
        while rclpy.ok():
            if self.buffer.can_transform(target_frame, source_frame, to_time(time)):
                return
            if _wall_time() > deadline:
                raise tf2_ros.LookupException(
                    "timed out waiting for transform %s -> %s" % (target_frame, source_frame))
            _sleep(0.05)


def _wall_time():
    import time
    return time.monotonic()


def _sleep(sec):
    import time
    time.sleep(sec)


# --MESSAGE FIELD TYPES
_FLOAT_TYPES = {'float', 'double'}
_INT_TYPES = {'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64', 'octet'}


def coerce_numeric_fields(*msg_classes):
    """Make numeric fields of the given ROS 2 message classes accept numpy scalars and ints.

    rospy serialized np.float32 / np.int64 / int values into float fields without complaint. The
    rclpy setters accept them too, but the C conversion at publish() time asserts PyFloat/PyLong and
    aborts the whole process. The ROS 1 code assigns numpy scalars to marker fields all over the
    place, so instead of chasing every call site the setters of the message classes we publish are
    wrapped to convert with float()/int(). Idempotent.
    """
    for cls in msg_classes:
        if getattr(cls, '_monospheres_coerced', False):
            continue
        for field, ftype in cls.get_fields_and_field_types().items():
            if ftype in _FLOAT_TYPES:
                conv = float
            elif ftype in _INT_TYPES:
                conv = int
            else:
                continue
            prop = getattr(cls, field)
            setattr(cls, field, property(prop.fget, _make_setter(prop.fset, conv), prop.fdel, prop.__doc__))
        cls._monospheres_coerced = True


def _make_setter(fset, conv):
    def setter(self, value):
        fset(self, conv(value))
    return setter


def _coerce_common_messages():
    from geometry_msgs.msg import Point, Point32, Pose, Quaternion, Vector3
    from mrs_msgs.msg import Reference
    from sensor_msgs.msg import PointField
    from std_msgs.msg import ColorRGBA
    from visualization_msgs.msg import Marker
    coerce_numeric_fields(Point, Point32, Pose, Quaternion, Vector3, ColorRGBA, Marker, Reference, PointField)


_coerce_common_messages()


# --PATHS
def package_data_dir(*subdirs):
    """Writable/readable data directory of the package (replaces rospkg.RosPack().get_path()).

    In ROS 1 this pointed at the source package. In ROS 2 the package is installed, so use the
    share directory (writable for workspace builds).
    """
    from ament_index_python.packages import get_package_share_directory
    path = os.path.join(get_package_share_directory('monospheres'), *subdirs)
    return path
