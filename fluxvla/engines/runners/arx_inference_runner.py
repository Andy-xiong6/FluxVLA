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

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..utils.root import RUNNERS
from .base_inference_runner import BaseInferenceRunner


@RUNNERS.register_module()
class ARXInferenceRunner(BaseInferenceRunner):
    """Runner for ARX X5 real-robot inference.

    The implementation intentionally mirrors the UR real-robot path so that
    `scripts/inference_real_robot.py` can keep using the shared base loop.

    Quick usage:
        1. Set `type='ARXInferenceRunner'` in your inference config.
        2. If you do not pass an `operator`, this runner will create a default
           `ARXROSOperator` config.
        3. Before first real deployment, replace the TODO-marked topics,
           message types, field paths, and joint ordering below.
        4. Launch through the shared entrypoint:
           `python scripts/inference_real_robot.py --config <your_config> `
           `--ckpt-path <your_ckpt>`

    Real-robot bring-up checklist:
        - wrist RGB topic
        - third-person RGB topic
        - joint state topic
        - gripper state topic
        - optional EE pose topic
        - joint and gripper command topics
        - ROS message types / nested field paths
        - `joint_indices` ordering relative to training
    """

    def __init__(self,
                 joint_indices: Optional[List[int]] = None,
                 arm_action_dim: int = 6,
                 prepare_pose: Optional[List[float]] = None,
                 prepare_gripper: Optional[float] = None,
                 joint_command_mode: str = 'servoj',
                 *args,
                 **kwargs):
        if 'camera_names' not in kwargs or kwargs['camera_names'] is None:
            kwargs['camera_names'] = ['cam_high', 'cam_left_wrist']

        if 'operator' not in kwargs or kwargs['operator'] is None:
            # -----------------------------------------------------------------
            # TODO(arx): Real-robot setup usually only needs edits in this block.
            #
            # Replace these placeholder values with the real ARX X5 ROS topics
            # and message conventions found on your robot:
            #   1. RGB topics: wrist + third-person camera
            #   2. State topics: joints + gripper + optional ee pose
            #   3. Command topics: joints + gripper + optional pose
            #   4. Message types if your deployment uses custom wrappers
            #   5. Field paths if the payload is nested instead of using the
            #      standard `position` / `data` attributes
            #   6. `joint_names` if the downstream controller requires them
            #
            # If your current setup does not publish end-effector pose, set:
            #   `ee_pose_topic=None`
            # -----------------------------------------------------------------
            kwargs['operator'] = {
                'type': 'ARXROSOperator',
                # TODO(arx): Replace with the real wrist RGB topic on ARX X5.
                'img_wrist_topic': '/arx/camera/wrist/color/image_raw',
                # TODO(arx): Replace with the real third-person RGB topic.
                'img_third_topic': '/arx/camera/third/color/image_raw',
                # TODO(arx): Replace with the real arm joint state topic.
                'joint_state_topic': '/arx/joint_states',
                # TODO(arx): Replace with the real gripper state topic.
                'gripper_state_topic': '/arx/gripper/state',
                # TODO(arx): Set to the real EE pose topic, or None if absent.
                'ee_pose_topic': '/arx/ee_pose',
                # TODO(arx): Replace with the real joint command topic.
                'joint_command_topic': '/arx/command/joint',
                # TODO(arx): Replace with the real gripper command topic.
                'gripper_command_topic': '/arx/command/gripper',
                # TODO(arx): Keep for future pose control; replace if used.
                'pose_command_topic': '/arx/command/pose',
                # TODO(arx): Adjust these if ARX uses custom ROS message types.
                'joint_state_msg_type': 'sensor_msgs.msg:JointState',
                'gripper_state_msg_type': 'std_msgs.msg:Float32',
                'joint_command_msg_type': 'sensor_msgs.msg:JointState',
                'gripper_command_msg_type': 'std_msgs.msg:Float32',
                # TODO(arx): Adjust field paths if values are wrapped.
                'joint_state_field': 'position',
                'gripper_state_field': 'data',
                'joint_command_field': 'position',
                'gripper_command_field': 'data',
                # TODO(arx): Fill joint names if the downstream controller
                # requires named JointState commands.
                'joint_names': [],
                'use_depth_image': False,
            }

        if 'task_descriptions' not in kwargs or kwargs[
                'task_descriptions'] is None:
            kwargs['task_descriptions'] = {
                '1': 'pick up the target object',
                '2': 'place the target object into the container',
            }

        super().__init__(*args, **kwargs)

        self.arm_action_dim = arm_action_dim
        # ---------------------------------------------------------------------
        # TODO(arx): Real action/state alignment usually only needs edits here.
        #
        # `joint_indices` reorders incoming ROS joint states before building:
        #   qpos = [arm_joints..., gripper]
        #
        # Keep `arm_action_dim` aligned with your trained action layout.
        # For the current Pi0.5 joint-control MVP, the default expectation is:
        #   - 6 arm joints
        #   - 1 gripper value
        #
        # Example if ROS publishes joints in a different order:
        #   joint_indices = [2, 1, 0, 3, 4, 5]
        #
        # `joint_command_mode`:
        #   - 'servoj' for servo-style joint streaming
        #   - 'movej'  for discrete joint moves
        # ---------------------------------------------------------------------
        # TODO(arx): Update this mapping to match the joint order used during
        # training if ROS JointState.position is ordered differently on ARX.
        self.joint_indices = joint_indices
        # TODO(arx): Optionally set a real prepare pose for task reset.
        self.prepare_pose = prepare_pose
        # TODO(arx): Optionally set the gripper opening used during reset.
        self.prepare_gripper = prepare_gripper
        # TODO(arx): Switch to 'movej' if the ARX controller expects
        # non-servo joint commands during inference.
        self.joint_command_mode = joint_command_mode

    def get_ros_observation(
        self
    ) -> Tuple[np.ndarray, np.ndarray, 'JointState', 'Any', 'Any']:  # noqa: F821
        import rospy

        from ..utils import initialize_overwatch

        overwatch = initialize_overwatch(__name__)

        rate = rospy.Rate(self.publish_rate)
        print_flag = True
        rate.sleep()

        while not rospy.is_shutdown():
            result = self.ros_operator.get_frame()
            if not result:
                if print_flag:
                    overwatch.info(
                        'Synchronization failed in get_ros_observation')
                    print_flag = False
                rate.sleep()
                continue

            print_flag = True
            (img_third, img_wrist, img_third_depth, img_wrist_depth,
             joint_state, ee_pose, gripper_state, frame_time_min,
             frame_time_max) = result

            return (img_third, img_wrist, joint_state, ee_pose,
                    gripper_state)

    def _select_arm_joints(self, joint_positions: List[float]) -> np.ndarray:
        joint_positions = np.asarray(joint_positions)
        if self.joint_indices is None:
            return joint_positions[:self.arm_action_dim]
        return joint_positions[self.joint_indices]

    def update_observation_window(self) -> Dict:
        from collections import deque

        if self.observation_window is None:
            self.observation_window = deque(maxlen=2)
            dummy_obs = {'qpos': None}
            for camera_name in self.camera_names:
                dummy_obs[camera_name] = None
            self.observation_window.append(dummy_obs)

        img_third, img_wrist, joint_state, ee_pose, gripper_state = (
            self.get_ros_observation())

        img_third = self._apply_jpeg_compression(img_third)
        img_wrist = self._apply_jpeg_compression(img_wrist)

        arm_joints = self._select_arm_joints(
            self.ros_operator.get_joint_positions(joint_state))
        gripper = self.ros_operator.get_gripper_position(gripper_state)
        gripper = np.asarray([gripper], dtype=np.float32).reshape(-1)
        qpos = np.concatenate([arm_joints, gripper], axis=0)

        observation = {
            'qpos': qpos,
            self.camera_names[0]: img_third,
            self.camera_names[1]: img_wrist,
        }
        if ee_pose is not None:
            observation['ee_pose'] = ee_pose

        self.observation_window.append(observation)
        return self.observation_window[-1]

    def _move_to_prepare_pose(self):
        if self.prepare_pose is not None:
            self.ros_operator.movej(self.prepare_pose)
        if self.prepare_gripper is not None:
            self.ros_operator.movegrip(self.prepare_gripper)

    def execute_task_pose(self, task_id: str):
        if task_id in self.task_pose_sequences:
            pose_sequence = self.task_pose_sequences[task_id]
            for pose_item in pose_sequence:
                joint_angles = pose_item.get('joint_positions')
                gripper_position = pose_item.get('gripper')
                ee_pose = pose_item.get('ee_pose')
                if joint_angles is not None:
                    self.ros_operator.movej(joint_angles)
                if ee_pose is not None:
                    self.ros_operator.movep(ee_pose)
                if gripper_position is not None:
                    self.ros_operator.movegrip(gripper_position)

    def _execute_actions(self, actions: np.ndarray, rate):
        joint_cmd = getattr(self.ros_operator, self.joint_command_mode)
        for action in actions:
            joint_cmd(action[:self.arm_action_dim])
            if action.shape[0] > self.arm_action_dim:
                self.ros_operator.movegrip(action[self.arm_action_dim])
            rate.sleep()
