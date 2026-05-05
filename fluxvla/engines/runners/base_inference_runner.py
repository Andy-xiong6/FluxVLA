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

import contextlib
import io
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from safetensors.torch import load_file

from fluxvla.engines.utils.torch_utils import set_seed_everywhere
from ..utils import (build_metrics_manager_from_cfg, build_operator_from_cfg,
                     initialize_overwatch)
from ..utils.name_map import str_to_dtype

overwatch = initialize_overwatch(__name__)


class BaseInferenceRunner:
    """Base class for robot inference runners.

    Provides model setup, observation handling, task management, and a
    template inference loop with four overridable phases:

        _preprocess  → observe + build model inputs
        _predict_action  → model inference (normalized actions)
        _postprocess_actions → denormalize to robot command space
        _execute_actions → send commands to the robot (abstract)

    When ``remote_inference`` is provided, the runner delegates model
    inference to a remote ZMQ GPU server instead of loading the model
    locally.  Subclasses (URInferenceRunner, AlohaInferenceRunner) work
    identically in both local and remote modes.

    Args:
        cfg (Dict): Configuration dictionary for the VLA model.
        seed (str): Random seed for reproducibility.
        ckpt_path (str): Path to model checkpoint file.
        dataset (Dict): Dataset configuration dictionary.
        denormalize_action (Dict): Action denormalization configuration.
        task_suite_name (str): Name of task suite.
        state_dim (int): Dimension of robot state vector.
        action_chunk (int): Number of actions to predict at once.
        publish_rate (int): ROS publishing rate in Hz.
        max_publish_step (int): Maximum steps per episode.
        use_eval_collector (bool): Whether to use evaluation data collector.
        use_robot_base (bool): Whether to use mobile base.
        disable_puppet_arm (bool): Whether to disable puppet arm.
        camera_names (List[str]): Names of camera feeds.
        operator (Dict): ROS operator configuration.
        task_descriptions (Dict): Task descriptions mapping.
        task_pose_sequences (Dict): Task pose sequences mapping.
        mixed_precision_dtype (str): Data type for mixed precision.
        enable_mixed_precision (bool): Whether to enable mixed precision.
        remote_inference (Dict): Remote inference config.  When provided,
            model loading is skipped and inference is delegated to a ZMQ
            server.  Keys: ``server_host``, ``server_port``, ``timeout_s``,
            ``serializer``, ``compress``, ``enable_profiling``.
    """

    def __init__(self,
                 cfg: Dict = None,
                 seed: str = 7,
                 ckpt_path: str = None,
                 dataset: Dict = None,
                 denormalize_action: Dict = None,
                 task_suite_name: str = 'private',
                 state_dim: int = 7,
                 action_chunk: int = 32,
                 publish_rate: int = 30,
                 max_publish_step: int = 10000,
                 use_eval_collector: bool = False,
                 use_robot_base: bool = False,
                 disable_puppet_arm: bool = False,
                 camera_names: Optional[List[str]] = None,
                 operator: Dict = None,
                 task_descriptions: Dict = None,
                 task_pose_sequences: Dict = None,
                 mixed_precision_dtype: str = 'float32',
                 enable_mixed_precision: bool = True,
                 remote_inference: Dict = None,
                 metrics: Dict = None,
                 config_path: Optional[str] = None,
                 **kwargs):
        from fluxvla.engines import (build_dataset_from_cfg,
                                     build_transform_from_cfg,
                                     build_vla_from_cfg)

        self.ckpt_path = ckpt_path
        self._use_remote = remote_inference is not None

        if self._use_remote:
            self.dataset = None
            self.denormalize_action = None
            self.vla = None
            self._init_zmq_client(remote_inference)
        elif ckpt_path is not None:
            data_stat_path = os.path.join(
                Path(ckpt_path).resolve().parent.parent,
                'dataset_statistics.json')
            assert os.path.exists(data_stat_path), (
                f'Dataset statistics file not found at {data_stat_path}!')
            denormalize_action['norm_stats'] = data_stat_path
            self.denormalize_action = build_transform_from_cfg(
                denormalize_action)
            dataset['norm_stats'] = data_stat_path
            dataset['model_path'] = os.path.dirname(os.path.dirname(ckpt_path))
            self.dataset = build_dataset_from_cfg(dataset)

            self.vla = build_vla_from_cfg(cfg.inference_model)
            assert Path.exists(Path(ckpt_path)), \
                f'Checkpoint path {ckpt_path} does not exist!'
            if ckpt_path.endswith('.safetensors'):
                state_dict = load_file(ckpt_path, device='cpu')
            else:
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                if isinstance(checkpoint, dict) and 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
            self.vla.load_state_dict(state_dict, strict=True)
        else:
            self.dataset = None
            self.denormalize_action = None
            self.vla = None

        # Store configuration parameters
        self.seed = seed
        self.state_dim = state_dim
        self.action_chunk = action_chunk
        self.publish_rate = publish_rate
        self.max_publish_step = max_publish_step
        self.use_eval_collector = use_eval_collector
        self.use_robot_base = use_robot_base
        self.disable_puppet_arm = disable_puppet_arm
        self.camera_names = camera_names or []
        self.task_suite_name = task_suite_name

        # Initialize ROS operator and observation window
        self.ros_operator = build_operator_from_cfg(operator)
        self.observation_window = None

        # Initialize task configurations
        self.task_descriptions = task_descriptions or {}
        self.task_pose_sequences = task_pose_sequences or {}

        # Mixed precision settings
        self.mixed_precision_dtype = str_to_dtype(mixed_precision_dtype)
        self.enable_mixed_precision = enable_mixed_precision

        # Action context: SimpleNamespace shared between _predict_action,
        # _postprocess_actions, and _execute_actions within one iteration.
        # Becomes _prev_ctx in the next iteration for cross-chunk continuity.
        self._prev_ctx = None
        self._action_ctx = SimpleNamespace()

        # ------------------------------------------------------------------
        # Metrics recorder (independent module, optional).
        # ``metrics`` is the user-facing config dict from the inference
        # section; ``config_path`` is forwarded by inference_real_robot.py.
        # The MetricsManager is constructed AFTER ros_operator so that
        # rospy.init_node has already been invoked (see ARXROSOperator).
        # ------------------------------------------------------------------
        self.metrics_cfg = metrics
        self.config_path = config_path
        self._last_task_id: Optional[str] = None
        self._last_num_times: int = 0
        self.metrics = None
        self._inference_cfg_dict: Dict = self._extract_inference_cfg_snapshot(
            kwargs.get('cfg'))
        try:
            metrics_cfg = self._prepare_metrics_cfg(self.metrics_cfg)
            self.metrics = build_metrics_manager_from_cfg(
                metrics_cfg,
                runtime_meta_provider=self._build_runtime_meta,
                inference_config_provider=self._build_inference_config,
                ckpt_path=self.ckpt_path,
                config_path=self.config_path)
        except Exception as e:
            overwatch.warning(f'Metrics initialization failed: {e}. '
                              f'Continuing without metrics recording.')
            self.metrics = None

    def _prepare_metrics_cfg(self, metrics_cfg: Optional[Dict]):
        if metrics_cfg is None:
            return None
        cfg = dict(metrics_cfg)
        save_video = cfg.pop('save_cam_high_video', True)
        if save_video and 'video_topic' not in cfg:
            video_topic = getattr(self.ros_operator, 'img_third_topic', None)
            if video_topic:
                cfg['video_topic'] = video_topic
                cfg.setdefault(
                    'video_msg_type',
                    getattr(self.ros_operator, 'image_msg_type',
                            'sensor_msgs.msg:Image'))
                cfg.setdefault('video_filename', 'cam_high.mp4')
                cfg.setdefault('video_fps', 30.0)
        return cfg

    def _init_zmq_client(self, cfg: Dict):
        """Initialize ZMQ client for remote inference.

        Args:
            cfg (Dict): Remote inference config with keys server_host,
                server_port, timeout_s, serializer, compress,
                enable_profiling.
        """
        import zmq

        host = cfg.get('server_host', 'localhost')
        port = cfg.get('server_port', 5555)
        timeout_s = cfg.get('timeout_s', 30.0)
        serializer = cfg.get('serializer', 'msgpack')
        assert serializer in ('msgpack', 'protobuf'), \
            f"serializer must be 'msgpack' or 'protobuf', got '{serializer}'"

        self._serializer = serializer
        self._compress = cfg.get('compress', True)
        self._server_address = f'tcp://{host}:{port}'
        self._enable_profiling = cfg.get('enable_profiling', True)

        self._zmq_context = zmq.Context()
        self._zmq_socket = self._zmq_context.socket(zmq.REQ)
        self._zmq_socket.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
        self._zmq_socket.setsockopt(zmq.SNDTIMEO, int(timeout_s * 1000))
        self._zmq_socket.connect(self._server_address)
        self._zmq_lock = threading.Lock()

        self._call_count = 0
        self._t_serialize = 0.0
        self._t_zmq = 0.0
        self._t_deserialize = 0.0
        self._t_total = 0.0
        self._t_server_infer = 0.0
        self._t_network = 0.0
        self._payload_bytes = 0
        self._resp_bytes = 0
        self.last_profile = {}

    # ------------------------------------------------------------------
    # Metrics integration helpers.
    # ------------------------------------------------------------------
    def _extract_inference_cfg_snapshot(self, cfg) -> Dict:
        """Extract a JSON-friendly snapshot of cfg.inference (sans cycles)."""
        from ..utils.metrics_recorder import _jsonable
        if cfg is None or not hasattr(cfg, 'inference'):
            return {}
        try:
            inference_cfg = cfg.inference
            if hasattr(inference_cfg, 'to_dict'):
                raw = inference_cfg.to_dict()
            elif hasattr(inference_cfg, '_cfg_dict'):
                raw = dict(inference_cfg._cfg_dict)
            else:
                raw = dict(inference_cfg)
            for drop_key in ('cfg', 'config_path'):
                raw.pop(drop_key, None)
            return _jsonable(raw)
        except Exception as e:
            overwatch.warning(f'Failed to snapshot inference cfg: {e}')
            return {}

    def _build_runtime_meta(self) -> Dict:
        return {
            'publish_rate': getattr(self, 'publish_rate', None),
            'action_chunk': getattr(self, 'action_chunk', None),
            'dt': (1.0 / self.publish_rate
                   if getattr(self, 'publish_rate', None) else None),
            'arm_action_dim': getattr(self, 'arm_action_dim', None),
            'joint_indices': getattr(self, 'joint_indices', None),
            'dry_run': bool(getattr(self, 'dry_run', False)),
            'async_execution': bool(getattr(self, 'async_execution', False)),
            'binarize_gripper': bool(getattr(self, 'binarize_gripper',
                                             False)),
        }

    def _build_inference_config(self) -> Dict:
        return self._inference_cfg_dict or {}

    def _metrics_episode_ctx(self, task_id: str, instruction: str,
                             num_times: int):
        if self.metrics is None:
            return contextlib.nullcontext()
        return self.metrics.episode(task_id, instruction, num_times)

    def _metrics_inference_ctx(self, ctx):
        if self.metrics is None:
            return contextlib.nullcontext()
        return self.metrics.inference(ctx)

    def _emit_action_publish(self, ctx, n_actions: int, dt: float,
                             arm_action_dim: int, gripper_dim: int,
                             is_dry_run: bool):
        if self.metrics is None:
            return
        try:
            self.metrics.action_publish(
                ctx,
                n_actions=n_actions,
                dt=dt,
                arm_action_dim=arm_action_dim,
                gripper_dim=gripper_dim,
                is_dry_run=is_dry_run)
        except Exception as e:
            overwatch.warning(f'metrics.action_publish failed: {e}')

    def ping(self) -> bool:
        """Health-check the remote ZMQ server."""
        if not self._use_remote:
            return False
        import msgpack
        import zmq
        try:
            request = msgpack.packb({'endpoint': 'ping'})
            with self._zmq_lock:
                self._zmq_socket.send(request)
                raw = self._zmq_socket.recv()
            resp = msgpack.unpackb(raw, raw=False)
            return resp.get('status') == 'ok'
        except zmq.error.ZMQError:
            return False

    def _apply_jpeg_compression(self, img: np.ndarray) -> np.ndarray:
        """Apply JPEG compression and decompression to image.

        This transformation aligns the inference images with training data
        by applying the same JPEG compression artifacts that may have been
        present during dataset collection.

        Args:
            img (np.ndarray): Input BGR image array.

        Returns:
            np.ndarray: JPEG-processed BGR image array.
        """
        encoded_img = cv2.imencode('.jpg', img)[1].tobytes()
        decoded_img = cv2.imdecode(
            np.frombuffer(encoded_img, np.uint8), cv2.IMREAD_COLOR)
        return decoded_img

    def _get_task_description(self, task_id: str) -> str:
        """Get task description for given task ID.

        Args:
            task_id (str): Task identifier string.

        Returns:
            str: Human-readable task description.
        """
        return self.task_descriptions.get(
            task_id, 'place it in the brown paper bag with right arm')

    def execute_task_pose(self, task_id: str):
        """Execute pose sequence for a specific task.

        Base implementation does nothing.  Subclasses should override to
        implement robot-specific pose execution.

        Args:
            task_id (str): Task identifier string.
        """
        if task_id in self.task_pose_sequences:
            overwatch.info(f'Executing pose sequence for task {task_id}')

    def run_setup(self):
        """Set up the inference environment.

        In local mode, configures the model for evaluation, moves it to
        GPU, and sets random seeds.  In remote mode, pings the ZMQ
        server to verify connectivity.
        """
        set_seed_everywhere(self.seed)
        if self._use_remote:
            if not self.ping():
                raise ConnectionError(
                    f'Cannot reach VLA server at {self._server_address}')
            overwatch.info(f'Remote server OK at {self._server_address}. '
                           f'Seed set to {self.seed}')
        else:
            self.vla.eval()
            if self.enable_mixed_precision:
                self.vla.to(device='cuda', dtype=self.mixed_precision_dtype)
            else:
                self.vla.cuda()
            overwatch.info(
                f'Model loaded (dtype={self.mixed_precision_dtype}). '
                f'Seed set to {self.seed}')

    def run(self,
            initial_instruction:
            str = 'place it in the brown paper bag with right arm'):
        """Run the main inference loop.

        Executes continuous robotic manipulation tasks based on vision-language
        instructions. The loop handles task selection, action prediction,
        and robot control with proper error handling and user interaction.

        Args:
            initial_instruction (str, optional): Default task instruction.
                Defaults to 'place it in the brown paper bag with right arm'.

        Note:
            This method runs indefinitely until ROS shutdown is requested.
            It provides interactive task selection and automatic robot
            control based on VLA model predictions.
        """
        import rospy

        overwatch.info('Starting inference runner')

        # Main inference loop
        with torch.inference_mode():
            while not rospy.is_shutdown():
                self._run_episode(initial_instruction)

    def _run_episode(self, default_instruction: str):
        """Run a single episode: preprocess → predict → postprocess → execute.

        Subclasses should override individual phases rather than this method.

        Args:
            default_instruction (str): Default task instruction to use
        """
        """Run a single episode: preprocess → predict → postprocess → execute.

        Subclasses should override individual phases rather than this method.

        Args:
            default_instruction (str): Default task instruction to use
        """
        import rospy

        t = 0
        rate = rospy.Rate(self.publish_rate)

        while t < self.max_publish_step and not rospy.is_shutdown():
            instructions = self._get_user_task_instruction(default_instruction)
            task_id = self._last_task_id or ''
            num_times = self._last_num_times or len(instructions)
            instruction_for_meta = (instructions[0]
                                    if instructions else default_instruction)
            with self._metrics_episode_ctx(task_id, instruction_for_meta,
                                           num_times):
                self._prev_ctx = None
                for instruction in instructions:
                    self._action_ctx = SimpleNamespace()
                    self._action_ctx.instruction = instruction
                    inputs = self._preprocess(instruction)

                    with torch.autocast(
                            'cuda',
                            dtype=self.mixed_precision_dtype,
                            enabled=(self.enable_mixed_precision
                                     and not self._use_remote)):
                        with self._metrics_inference_ctx(self._action_ctx):
                            raw_action = self._predict_action(inputs)

                    actions = self._postprocess_actions(raw_action)
                    self._execute_actions(actions, rate)

                    self._prev_ctx = self._action_ctx
                    t += self.action_chunk
                    overwatch.info(f'Published Step {t}')

    def _preprocess(self, instruction: str) -> dict:
        """Observe environment and build model inputs.

        In local mode, runs the dataset transform pipeline.  In remote
        mode, returns raw observation for server-side preprocessing.

        Args:
            instruction (str): Task description for this chunk.

        Returns:
            dict: Model-ready inputs (local) or raw obs (remote).
        """
        obs = self.update_observation_window()
        obs['task_description'] = instruction
        if self._use_remote:
            obs['unnorm_key'] = self.task_suite_name
            return obs
        return self.dataset(obs)

    def _predict_action(self, inputs: dict):
        """Run model inference to produce actions.

        Dispatches to local model or remote ZMQ server depending on
        ``self._use_remote``.

        Args:
            inputs (dict): Model inputs from _preprocess.

        Returns:
            torch.Tensor: Action tensor.
        """
        if self._use_remote:
            return self._predict_action_remote(inputs)
        return self.vla.predict_action(**inputs)

    def _predict_action_remote(self, inputs: dict):
        """Send observation to remote server and receive action tensor.

        Handles serialization, ZMQ round-trip, deserialization, and
        latency profiling.

        Args:
            inputs (dict): Raw observation dict with unnorm_key.

        Returns:
            torch.Tensor: Action tensor from remote server.
        """
        from .serving.serializers import (FORMAT_PROTOBUF,
                                          decode_predict_response,
                                          encode_predict_request)

        t_total_start = time.perf_counter()
        unnorm_key = inputs.pop('unnorm_key', '')

        t0 = time.perf_counter()
        obs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                obs[k] = v.cpu().numpy()
            else:
                obs[k] = v
        request = encode_predict_request(
            obs,
            str(unnorm_key),
            fmt=self._serializer,
            compress=self._compress)
        payload_size = len(request)
        t_serialize = time.perf_counter() - t0

        t1 = time.perf_counter()
        with self._zmq_lock:
            self._zmq_socket.send(request)
            raw_response = self._zmq_socket.recv()
        fmt_tag = FORMAT_PROTOBUF if self._serializer == 'protobuf' else 0
        response = decode_predict_response(raw_response, fmt=fmt_tag)
        t_zmq = time.perf_counter() - t1

        if isinstance(response, dict) and 'error' in response:
            raise RuntimeError(f"ZMQ server error: {response['error']}")

        t2 = time.perf_counter()
        action_buf = io.BytesIO(response['action_data'])
        arr = np.load(action_buf, allow_pickle=False)
        actions = torch.from_numpy(arr.copy())
        t_deserialize = time.perf_counter() - t2

        t_total = time.perf_counter() - t_total_start
        server_infer = response.get('infer_time', 0.0)
        resp_size = len(raw_response)
        t_network = t_zmq - server_infer

        self.last_profile = {
            'serialize_ms': t_serialize * 1000,
            'zmq_roundtrip_ms': t_zmq * 1000,
            'server_infer_ms': server_infer * 1000,
            'network_ms': t_network * 1000,
            'deserialize_ms': t_deserialize * 1000,
            'total_ms': t_total * 1000,
            'payload_kb': payload_size / 1024,
            'response_kb': resp_size / 1024,
        }

        if self._enable_profiling:
            self._call_count += 1
            self._t_serialize += t_serialize
            self._t_zmq += t_zmq
            self._t_deserialize += t_deserialize
            self._t_total += t_total
            self._t_server_infer += server_infer
            self._t_network += t_network
            self._payload_bytes += payload_size
            self._resp_bytes += resp_size

            if self._call_count % 50 == 0:
                n = self._call_count
                overwatch.info(
                    f'[RemoteInference] calls={n}  '
                    f'avg_total={self._t_total/n*1000:.1f}ms  '
                    f'avg_serialize={self._t_serialize/n*1000:.1f}ms  '
                    f'avg_zmq={self._t_zmq/n*1000:.1f}ms  '
                    f'avg_server={self._t_server_infer/n*1000:.1f}ms  '
                    f'avg_network={self._t_network/n*1000:.1f}ms  '
                    f'avg_deser={self._t_deserialize/n*1000:.1f}ms  '
                    f'avg_payload={self._payload_bytes/n/1024:.0f}KB  '
                    f'avg_resp={self._resp_bytes/n/1024:.0f}KB')

        return actions

    def _postprocess_actions(self, raw_action):
        """Denormalize raw actions into robot command space.

        In remote mode the server already denormalized, so this just
        converts to numpy and truncates.

        Args:
            raw_action (torch.Tensor): Action tensor from _predict_action.

        Returns:
            np.ndarray: Denormalized actions, truncated to action_chunk.
        """
        if self._use_remote:
            return raw_action.cpu().numpy()[:self.action_chunk]
        denormalized = self.denormalize_action(
            dict(action=raw_action.cpu().numpy()))
        return denormalized[:self.action_chunk]

    def _get_user_task_instruction(self, default_instruction: str) -> str:
        """Get task instruction from user input.

        Side effects:
            ``self._last_task_id`` and ``self._last_num_times`` are updated
            so that the metrics episode-start hook can pick them up.  The
            return signature is unchanged (List[str]) for backwards
            compatibility with subclasses.

        Args:
            default_instruction (str): Default instruction if no valid input.

        Returns:
            str: Task instruction string.
        """
        task_id = input('Enter task ID (or press Enter for default): ').strip()
        if task_id == '0':
            # Reset to preparation pose
            self._move_to_prepare_pose()
            task_id = input('Enter task ID after reset: ').strip()

        if task_id in self.task_pose_sequences:
            self.execute_task_pose(task_id)
            input('Enter task ID (or press Enter for default): ').strip()

        num_times = int(input('Number of times to repeat the task: '))
        task_description = self._get_task_description(task_id)
        self._last_task_id = task_id
        self._last_num_times = int(num_times)
        return [task_description] * num_times

    def get_observation_statistics(self) -> Dict:
        """Get statistics about current observation data.

        Returns:
            Dict: Statistics including queue lengths and timing information
        """
        if self.observation_window is None:
            return {'status': 'not_initialized'}

        return {
            'window_length': len(self.observation_window),
            'window_maxlen': self.observation_window.maxlen,
            'has_current_obs': len(self.observation_window) > 0,
            'camera_names': self.camera_names,
            'state_dim': self.state_dim,
            'action_chunk': self.action_chunk,
        }

    def cleanup(self):
        """Clean up resources and shutdown gracefully.

        Releases ZMQ resources in remote mode, clears observation window
        and action context.
        """
        overwatch.info('Cleaning up BaseInferenceRunner')
        if self.metrics is not None:
            try:
                self.metrics.close()
            except Exception as e:
                overwatch.warning(f'metrics.close failed: {e}')
            self.metrics = None
        if self._use_remote:
            import zmq
            if hasattr(self, '_zmq_socket') and not self._zmq_socket.closed:
                self._zmq_socket.setsockopt(zmq.LINGER, 0)
                self._zmq_socket.close()
            if hasattr(self, '_zmq_context'):
                self._zmq_context.term()
        self._prev_ctx = None
        self._action_ctx = SimpleNamespace()
        if self.observation_window is not None:
            self.observation_window.clear()

        overwatch.info('BaseInferenceRunner cleanup completed')

    # Abstract methods that subclasses must implement
    def get_ros_observation(self):
        """Get synchronized observation data from ROS topics.

        This method should be implemented by subclasses to handle
        robot-specific observation collection.

        Returns:
            Tuple: Robot-specific observation data.
        """
        raise NotImplementedError(
            'Subclasses must implement get_ros_observation method')

    def update_observation_window(self) -> Dict:
        """Update the observation window with latest sensor data.

        This method should be implemented by subclasses to handle
        robot-specific observation window management.

        Returns:
            Dict: Latest observation data.
        """
        raise NotImplementedError(
            'Subclasses must implement update_observation_window method')

    def _execute_actions(self, actions: np.ndarray, rate):
        """Execute a sequence of robot actions.

        This method should be implemented by subclasses to handle
        robot-specific action execution.

        Args:
            actions (np.ndarray): Array of denormalized robot actions
            rate: ROS rate limiter for action timing
        """
        raise NotImplementedError(
            'Subclasses must implement _execute_actions method')
