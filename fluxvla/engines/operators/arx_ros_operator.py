# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import re
import time
from collections import deque
from copy import deepcopy
from typing import Any, Optional

import numpy as np

from fluxvla.engines.utils.root import OPERATORS


def _import_ros_type(type_path: str):
    """Import a ROS message class from a string path."""
    module_path, class_name = type_path.split(':')
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _replace_last_segment(input_string, new_segment='camera_info'):
    """Replace the last segment of a path-like string."""
    last_slash_index = input_string.rfind('/')
    if last_slash_index != -1:
        return input_string[:last_slash_index + 1] + new_segment
    return new_segment


def _set_nested_attr(obj: Any, attr_path: str, value: Any):
    """Set a nested attribute using dot notation."""
    target = obj
    attrs = attr_path.split('.')
    for attr in attrs[:-1]:
        target = getattr(target, attr)
    match = re.fullmatch(r'([A-Za-z_]\w*)\[(\d+)\]', attrs[-1])
    if match:
        attr_name, index = match.group(1), int(match.group(2))
        container = list(getattr(target, attr_name))
        if index >= len(container):
            raise IndexError(
                f'Index {index} out of bounds for attribute {attr_name!r}')
        container[index] = value
        setattr(target, attr_name, container)
        return
    setattr(target, attrs[-1], value)


def _get_nested_attr(obj: Any, attr_path: str, default: Any = None):
    """Read a nested attribute using dot notation."""
    target = obj
    for attr in attr_path.split('.'):
        match = re.fullmatch(r'([A-Za-z_]\w*)\[(\d+)\]', attr)
        if match:
            attr_name, index = match.group(1), int(match.group(2))
            if not hasattr(target, attr_name):
                return default
            target = getattr(target, attr_name)
            if index >= len(target):
                return default
            target = target[index]
            continue
        if not hasattr(target, attr):
            return default
        target = getattr(target, attr)
    return target


@OPERATORS.register_module()
class ARXROSOperator:
    """ARX single-arm ROS operator for image/state synchronization.

    The operator mirrors the existing UROperator split of responsibilities:
    it owns ROS subscriptions, message buffering, timestamp synchronization,
    and conversion between ROS messages and numpy/OpenCV friendly payloads.
    """

    def __init__(self,
                 img_wrist_topic,
                 img_third_topic,
                 joint_state_topic,
                 gripper_state_topic,
                 ee_pose_topic=None,
                 use_depth_image=False,
                 img_wrist_depth_topic=None,
                 img_third_depth_topic=None,
                 joint_command_topic='/arx/command/joint',
                 gripper_command_topic='/arx/command/gripper',
                 pose_command_topic=None,
                 image_msg_type='sensor_msgs.msg:Image',
                 joint_state_msg_type='sensor_msgs.msg:JointState',
                 gripper_state_msg_type='std_msgs.msg:Float32',
                 ee_pose_msg_type='geometry_msgs.msg:PoseStamped',
                 joint_command_msg_type='sensor_msgs.msg:JointState',
                 gripper_command_msg_type='std_msgs.msg:Float32',
                 pose_command_msg_type='geometry_msgs.msg:PoseStamped',
                 joint_state_field='position',
                 gripper_state_field='data',
                 joint_command_field='position',
                 gripper_command_field='data',
                 pose_command_frame_id='base_link',
                 combine_joint_gripper_command=False,
                 gripper_command_index=None,
                 joint_command_length=None,
                 joint_names=None,
                 queue_size=1000):
        self.img_wrist_topic = img_wrist_topic
        self.img_third_topic = img_third_topic
        self.joint_state_topic = joint_state_topic
        self.gripper_state_topic = gripper_state_topic
        self.ee_pose_topic = ee_pose_topic
        self.use_depth_image = use_depth_image
        self.img_wrist_depth_topic = img_wrist_depth_topic
        self.img_third_depth_topic = img_third_depth_topic
        self.joint_command_topic = joint_command_topic
        self.gripper_command_topic = gripper_command_topic
        self.pose_command_topic = pose_command_topic
        self.joint_state_field = joint_state_field
        self.gripper_state_field = gripper_state_field
        self.joint_command_field = joint_command_field
        self.gripper_command_field = gripper_command_field
        self.pose_command_frame_id = pose_command_frame_id
        self.combine_joint_gripper_command = combine_joint_gripper_command
        self.gripper_command_index = gripper_command_index
        self.joint_command_length = joint_command_length
        self.joint_names = joint_names or []
        self.queue_size = queue_size

        self.image_msg_cls = _import_ros_type(image_msg_type)
        self.joint_state_msg_cls = _import_ros_type(joint_state_msg_type)
        self.gripper_state_msg_cls = _import_ros_type(gripper_state_msg_type)
        self.ee_pose_msg_cls = (_import_ros_type(ee_pose_msg_type)
                                if ee_pose_topic else None)
        self.joint_command_msg_cls = _import_ros_type(joint_command_msg_type)
        self.gripper_command_msg_cls = _import_ros_type(
            gripper_command_msg_type)
        self.pose_command_msg_cls = (_import_ros_type(pose_command_msg_type)
                                     if pose_command_topic else None)

        if self.use_depth_image:
            if not img_wrist_depth_topic or not img_third_depth_topic:
                raise ValueError(
                    'When use_depth_image=True, both wrist and third-person '
                    'depth topics must be provided')

        self._init_count()
        self._init()
        self._init_ros()

    def _init_count(self):
        self.rgb_wrist_count = 0
        self.rgb_third_count = 0
        self.depth_wrist_count = 0
        self.depth_third_count = 0

    def _init(self):
        from cv_bridge import CvBridge

        self.rgb_wrist = 0
        self.rgb_third = 0
        self.depth_wrist = 0
        self.depth_third = 0
        self.last_time_step = 0
        self.bridge = CvBridge()
        self.last_joint_command = None

        self.img_wrist_deque = deque()
        self.img_third_deque = deque()
        self.img_wrist_depth_deque = deque()
        self.img_third_depth_deque = deque()
        self.joint_state_deque = deque()
        self.gripper_state_deque = deque()
        self.ee_pose_deque = deque()

    def get_frame(self, slop=0.7):
        """Get synchronized frame data from all configured sensors."""
        required_queues_empty = (
            len(self.img_wrist_deque) == 0 or len(self.img_third_deque) == 0
            or len(self.joint_state_deque) == 0
            or len(self.gripper_state_deque) == 0)
        depth_queues_empty = (
            self.use_depth_image and
            (len(self.img_wrist_depth_deque) == 0
             or len(self.img_third_depth_deque) == 0))

        if required_queues_empty or depth_queues_empty:
            self._handle_empty_queues()
            return False

        frame_time = self._calculate_frame_time()
        if not self._check_sensor_data_availability(frame_time):
            return False

        self.last_time_step = frame_time
        self.rgb_wrist = 0
        self.rgb_third = 0
        self.depth_wrist = 0
        self.depth_third = 0

        frame_time_max = self._synchronize_queues(frame_time)
        if abs(frame_time_max - frame_time) > slop:
            self._flush_outdated_data(frame_time)
            return False

        return self._extract_synchronized_data()

    def _handle_empty_queues(self):
        if len(self.img_wrist_deque) == 0:
            self.rgb_wrist += 1
            if self.rgb_wrist > 3:
                print('Error wrist RGB', str(time.time()))

        if len(self.img_third_deque) == 0:
            self.rgb_third += 1
            if self.rgb_third > 3:
                print('Error third RGB', str(time.time()))

        if self.use_depth_image:
            if len(self.img_wrist_depth_deque) == 0:
                self.depth_wrist += 1
                if self.depth_wrist > 3:
                    print('Error wrist Depth', str(time.time()))

            if len(self.img_third_depth_deque) == 0:
                self.depth_third += 1
                if self.depth_third > 3:
                    print('Error third Depth', str(time.time()))

    def _calculate_frame_time(self):
        timestamps = [
            self.img_wrist_deque[-1].header.stamp.to_sec(),
            self.img_third_deque[-1].header.stamp.to_sec(),
            self.joint_state_deque[-1].header.stamp.to_sec(),
            self.gripper_state_deque[-1].header.stamp.to_sec(),
        ]
        if len(self.ee_pose_deque) > 0:
            timestamps.append(self.ee_pose_deque[-1].header.stamp.to_sec())
        if self.use_depth_image:
            timestamps.extend([
                self.img_wrist_depth_deque[-1].header.stamp.to_sec(),
                self.img_third_depth_deque[-1].header.stamp.to_sec(),
            ])
        return min(timestamps)

    def _check_sensor_data_availability(self, frame_time):
        checks = [
            self.img_wrist_deque,
            self.img_third_deque,
            self.joint_state_deque,
            self.gripper_state_deque,
        ]
        if len(self.ee_pose_deque) > 0:
            checks.append(self.ee_pose_deque)

        for queue in checks:
            if (len(queue) == 0
                    or queue[-1].header.stamp.to_sec() < frame_time):
                return False

        if self.use_depth_image:
            for queue in (self.img_wrist_depth_deque,
                          self.img_third_depth_deque):
                if (len(queue) == 0
                        or queue[-1].header.stamp.to_sec() < frame_time):
                    return False

        return True

    def _synchronize_queues(self, frame_time):
        frame_time_max = 0
        queues_to_sync = [
            self.img_wrist_deque,
            self.img_third_deque,
            self.joint_state_deque,
            self.gripper_state_deque,
        ]
        if len(self.ee_pose_deque) > 0:
            queues_to_sync.append(self.ee_pose_deque)

        for queue in queues_to_sync:
            while queue[0].header.stamp.to_sec() < frame_time:
                queue.popleft()
            frame_time_max = max(frame_time_max,
                                 queue[0].header.stamp.to_sec())

        if self.use_depth_image:
            for queue in (self.img_wrist_depth_deque,
                          self.img_third_depth_deque):
                while queue[0].header.stamp.to_sec() < frame_time:
                    queue.popleft()
                frame_time_max = max(frame_time_max,
                                     queue[0].header.stamp.to_sec())

        return frame_time_max

    def _flush_outdated_data(self, frame_time):
        queues_to_flush = [
            self.img_wrist_deque,
            self.img_third_deque,
            self.img_wrist_depth_deque,
            self.img_third_depth_deque,
            self.joint_state_deque,
            self.gripper_state_deque,
            self.ee_pose_deque,
        ]
        for queue in queues_to_flush:
            while (len(queue) > 0
                   and queue[0].header.stamp.to_sec() <= frame_time):
                queue.popleft()

    def _extract_synchronized_data(self):
        img_third = self.bridge.imgmsg_to_cv2(self.img_third_deque.popleft(),
                                              'passthrough')
        img_wrist = self.bridge.imgmsg_to_cv2(self.img_wrist_deque.popleft(),
                                              'passthrough')

        joint_state = self.joint_state_deque.popleft()
        gripper_state = self.gripper_state_deque.popleft()
        ee_pose = self.ee_pose_deque.popleft() if len(
            self.ee_pose_deque) > 0 else None

        img_third_depth = None
        img_wrist_depth = None
        if self.use_depth_image:
            img_third_depth = self.bridge.imgmsg_to_cv2(
                self.img_third_depth_deque.popleft(), 'passthrough')
            img_wrist_depth = self.bridge.imgmsg_to_cv2(
                self.img_wrist_depth_deque.popleft(), 'passthrough')

        return (
            img_third,
            img_wrist,
            img_third_depth,
            img_wrist_depth,
            joint_state,
            ee_pose,
            gripper_state,
            self.last_time_step,
            self.last_time_step,
        )

    def _append_with_limit(self, queue, msg):
        if len(queue) >= 20000:
            queue.popleft()
        queue.append(msg)

    def img_wrist_callback(self, msg):
        self._append_with_limit(self.img_wrist_deque, msg)

    def img_third_callback(self, msg):
        self._append_with_limit(self.img_third_deque, msg)

    def img_wrist_depth_callback(self, msg):
        self._append_with_limit(self.img_wrist_depth_deque, msg)

    def img_third_depth_callback(self, msg):
        self._append_with_limit(self.img_third_depth_deque, msg)

    def joint_state_callback(self, msg):
        self._append_with_limit(self.joint_state_deque, msg)

    def gripper_state_callback(self, msg):
        self._append_with_limit(self.gripper_state_deque, msg)

    def ee_pose_callback(self, msg):
        self._append_with_limit(self.ee_pose_deque, msg)

    def _init_ros(self):
        import rospy

        rospy.init_node('record_episodes', anonymous=True)
        camera_info_topics = []

        rospy.Subscriber(
            self.img_wrist_topic,
            self.image_msg_cls,
            self.img_wrist_callback,
            queue_size=self.queue_size,
            tcp_nodelay=True)
        camera_info_topics.append(_replace_last_segment(self.img_wrist_topic))

        rospy.Subscriber(
            self.img_third_topic,
            self.image_msg_cls,
            self.img_third_callback,
            queue_size=self.queue_size,
            tcp_nodelay=True)
        camera_info_topics.append(_replace_last_segment(self.img_third_topic))

        if self.use_depth_image:
            rospy.Subscriber(
                self.img_wrist_depth_topic,
                self.image_msg_cls,
                self.img_wrist_depth_callback,
                queue_size=self.queue_size,
                tcp_nodelay=True)
            camera_info_topics.append(
                _replace_last_segment(self.img_wrist_depth_topic))

            rospy.Subscriber(
                self.img_third_depth_topic,
                self.image_msg_cls,
                self.img_third_depth_callback,
                queue_size=self.queue_size,
                tcp_nodelay=True)
            camera_info_topics.append(
                _replace_last_segment(self.img_third_depth_topic))

        rospy.Subscriber(
            self.joint_state_topic,
            self.joint_state_msg_cls,
            self.joint_state_callback,
            queue_size=self.queue_size,
            tcp_nodelay=True)
        rospy.Subscriber(
            self.gripper_state_topic,
            self.gripper_state_msg_cls,
            self.gripper_state_callback,
            queue_size=self.queue_size,
            tcp_nodelay=True)

        if self.ee_pose_topic:
            rospy.Subscriber(
                self.ee_pose_topic,
                self.ee_pose_msg_cls,
                self.ee_pose_callback,
                queue_size=self.queue_size,
                tcp_nodelay=True)

        self.joint_command_pub = rospy.Publisher(
            self.joint_command_topic, self.joint_command_msg_cls, queue_size=10)
        self.gripper_command_pub = rospy.Publisher(
            self.gripper_command_topic,
            self.gripper_command_msg_cls,
            queue_size=10)
        self.pose_command_pub = None
        if self.pose_command_topic:
            self.pose_command_pub = rospy.Publisher(
                self.pose_command_topic,
                self.pose_command_msg_cls,
                queue_size=10)

        self.cam_info_dict = {}
        camera_info_cls = _import_ros_type('sensor_msgs.msg:CameraInfo')
        for topic in camera_info_topics:
            try:
                camera_info = rospy.wait_for_message(
                    topic, camera_info_cls, timeout=5)
            except Exception:
                continue
            self.cam_info_dict[topic] = {
                'rostopic': topic,
                'height': camera_info.height,
                'width': camera_info.width,
                'distortion_model': camera_info.distortion_model,
                'D': camera_info.D,
                'K': camera_info.K,
                'R': camera_info.R,
                'P': camera_info.P,
                'binning_x': camera_info.binning_x,
                'binning_y': camera_info.binning_y
            }

    def movej(self, qpos):
        if self.combine_joint_gripper_command:
            self.command_joints_and_gripper(qpos, publish=True)
            return
        msg = self.joint_command_msg_cls()
        if hasattr(msg, 'name') and self.joint_names:
            msg.name = list(self.joint_names)
        _set_nested_attr(msg, self.joint_command_field, list(qpos))
        self.joint_command_pub.publish(msg)

    def servoj(self, qpos):
        self.movej(qpos)

    def movegrip(self, gripper_position):
        if self.combine_joint_gripper_command:
            self.command_joints_and_gripper(
                joint_positions=None,
                gripper_position=gripper_position,
                publish=True)
            return
        msg = self.gripper_command_msg_cls()
        value = deepcopy(gripper_position)
        if isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
        if isinstance(value, np.ndarray):
            if value.size != 1:
                raise ValueError(
                    'Gripper command array must contain exactly one value')
            value = value.reshape(-1)[0]
        _set_nested_attr(msg, self.gripper_command_field, value)
        self.gripper_command_pub.publish(msg)

    def servop(self, eepose, frame_id: Optional[str] = None):
        if self.pose_command_pub is None:
            raise RuntimeError('pose_command_topic is not configured')

        msg = self.pose_command_msg_cls()
        if hasattr(msg, 'header'):
            msg.header.frame_id = frame_id or self.pose_command_frame_id

        if len(eepose) != 7:
            raise ValueError(
                'End-effector pose must contain exactly 7 elements')

        if hasattr(msg, 'pose'):
            pose = msg.pose
        elif hasattr(msg, 'position') and hasattr(msg, 'orientation'):
            pose = msg
        else:
            raise AttributeError(
                'Pose command message must expose pose or '
                'position/orientation fields')
        pose.position.x = eepose[0]
        pose.position.y = eepose[1]
        pose.position.z = eepose[2]
        pose.orientation.x = eepose[3]
        pose.orientation.y = eepose[4]
        pose.orientation.z = eepose[5]
        pose.orientation.w = eepose[6]
        self.pose_command_pub.publish(msg)

    def movep(self, eepose, frame_id: Optional[str] = None):
        self.servop(eepose, frame_id=frame_id)

    def _coerce_gripper_scalar(self, value):
        value = deepcopy(value)
        if isinstance(value, np.ndarray):
            if value.size != 1:
                raise ValueError(
                    'Gripper command array must contain exactly one value')
            return float(value.reshape(-1)[0])
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError(
                    'Gripper command sequence must contain exactly one value')
            return value[0]
        return value

    def _infer_joint_command_length(self, joint_positions):
        if self.joint_command_length is not None:
            return self.joint_command_length
        current = self.last_joint_command
        if current is None and len(self.joint_state_deque) > 0:
            current = self.get_joint_positions(self.joint_state_deque[-1])
        if current is not None:
            return len(current)
        if self.gripper_command_index is not None:
            return max(len(joint_positions), self.gripper_command_index + 1)
        return len(joint_positions)

    def command_joints_and_gripper(self,
                                   joint_positions=None,
                                   gripper_position=None,
                                   publish=True):
        if joint_positions is None and gripper_position is None:
            raise ValueError('At least one of joint_positions or '
                             'gripper_position must be provided')

        if joint_positions is not None:
            joint_positions = list(joint_positions)

        current = self.last_joint_command
        if current is None and len(self.joint_state_deque) > 0:
            current = list(self.get_joint_positions(self.joint_state_deque[-1]))

        target_length = self._infer_joint_command_length(
            joint_positions if joint_positions is not None else [])
        if current is None:
            full_command = [0.0] * target_length
        else:
            full_command = list(current)
            if len(full_command) < target_length:
                full_command.extend([0.0] * (target_length - len(full_command)))

        if joint_positions is not None:
            full_command[:len(joint_positions)] = joint_positions

        if gripper_position is not None:
            gripper_value = self._coerce_gripper_scalar(gripper_position)
            if self.gripper_command_index is None:
                if len(full_command) == 0:
                    full_command = [gripper_value]
                elif joint_positions is not None and len(full_command) > len(
                        joint_positions):
                    full_command[len(joint_positions)] = gripper_value
                else:
                    full_command[-1] = gripper_value
            else:
                if self.gripper_command_index >= len(full_command):
                    full_command.extend([0.0] *
                                        (self.gripper_command_index + 1 -
                                         len(full_command)))
                full_command[self.gripper_command_index] = gripper_value

        self.last_joint_command = list(full_command)

        if publish:
            msg = self.joint_command_msg_cls()
            if hasattr(msg, 'name') and self.joint_names:
                msg.name = list(self.joint_names)
            _set_nested_attr(msg, self.joint_command_field, full_command)
            self.joint_command_pub.publish(msg)

        return full_command

    def get_joint_positions(self, joint_state_msg):
        values = _get_nested_attr(joint_state_msg, self.joint_state_field)
        if values is None:
            raise AttributeError(
                f'Joint state field {self.joint_state_field!r} not found')
        return list(values)

    def get_gripper_position(self, gripper_state_msg):
        value = _get_nested_attr(gripper_state_msg, self.gripper_state_field)
        if value is None:
            raise AttributeError(
                f'Gripper state field {self.gripper_state_field!r} not found')
        if isinstance(value, np.ndarray):
            if value.size != 1:
                raise ValueError(
                    'Gripper state array must contain exactly one value')
            return float(value.reshape(-1)[0])
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError(
                    'Gripper state sequence must contain exactly one value')
            return value[0]
        return value
