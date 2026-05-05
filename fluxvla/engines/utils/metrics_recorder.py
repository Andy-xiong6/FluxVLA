

"""Metrics recorder for inference runners.

Implements the design described in the project plan:
    * MetricsManager (Runner-scoped, owns the 100Hz ROS subscription).
    * EpisodeWriter (per-episode JSONL writer, driven via ctx hooks).
    * build_metrics_manager_from_cfg factory.
"""

import contextlib
import csv
import importlib
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from .overwatch import initialize_overwatch

overwatch = initialize_overwatch(__name__)

# ---------------------------------------------------------------------------
# Module-level constants (plan section 4.1)
# ---------------------------------------------------------------------------
EVENTS_FILENAME = 'events.jsonl'
JOINTSTATE_FILENAME = 'jointstate.jsonl'
META_FILENAME = 'meta.json'
MANIFEST_FILENAME = 'manifest.jsonl'
SUCCESS_CSV_FILENAME = 'success.csv'
DEFAULT_VIDEO_FILENAME = 'cam_high.mp4'


# ---------------------------------------------------------------------------
# Module-level helpers (plan section 4.2)
# ---------------------------------------------------------------------------
def _iso_now() -> str:
    return datetime.now().isoformat(timespec='microseconds')


def _make_episode_id(task_id: str, repeat_idx: int, ts: float) -> str:
    dt = datetime.fromtimestamp(ts)
    safe_task_id = re.sub(r'[^A-Za-z0-9_-]', '_', str(task_id))
    return (f'{dt.strftime("%Y%m%d_%H%M%S")}_{safe_task_id}_'
            f'{repeat_idx:03d}')


def _safe_git_info(repo_root: str):
    try:
        commit = subprocess.check_output(
            ['git', '-C', repo_root, 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL,
            timeout=2,
            text=True).strip()
    except Exception:
        commit = None
    try:
        status = subprocess.check_output(
            ['git', '-C', repo_root, 'status', '--porcelain'],
            stderr=subprocess.DEVNULL,
            timeout=2,
            text=True)
        dirty = bool(status.strip())
    except Exception:
        dirty = None
    return commit, dirty


def _import_ros_msg_cls(type_or_cls):
    if type_or_cls is None:
        return None
    if not isinstance(type_or_cls, str):
        return type_or_cls
    module_path, cls_name = type_or_cls.split(':')
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


def _jsonable(obj: Any) -> Any:
    """Recursively convert obj into a JSON-serializable structure.

    Drops callables, breaks reference cycles, and falls back to repr() for
    unknown leaf objects.
    """
    seen = set()

    def _convert(x):
        x_id = id(x)
        if isinstance(x, (str, bool)) or x is None:
            return x
        if isinstance(x, (int, float)):
            return x
        if x_id in seen:
            return '<cycle>'
        if isinstance(x, (list, tuple)):
            seen.add(x_id)
            try:
                return [_convert(v) for v in x]
            finally:
                seen.discard(x_id)
        if isinstance(x, dict):
            seen.add(x_id)
            try:
                out = {}
                for k, v in x.items():
                    if callable(v):
                        continue
                    out[str(k)] = _convert(v)
                return out
            finally:
                seen.discard(x_id)
        if callable(x):
            return None
        for attr in ('to_dict', '_cfg_dict'):
            if hasattr(x, attr):
                try:
                    val = getattr(x, attr)
                    if callable(val):
                        val = val()
                    return _convert(val)
                except Exception:
                    pass
        try:
            return repr(x)
        except Exception:
            return None

    return _convert(obj)


# ---------------------------------------------------------------------------
# EpisodeWriter (plan section 4.3)
# ---------------------------------------------------------------------------
class EpisodeWriter:
    """Per-episode JSONL writer.

    Writes events.jsonl, jointstate.jsonl, and meta.json under
    ``<output_root>/<episode_id>/``.
    """

    def __init__(self, episode_id: str, episode_dir: str, task_id: str,
                 instruction: str, num_times: int,
                 ckpt_path: Optional[str], config_path: Optional[str],
                 runtime_meta: Dict, inference_config_snapshot: Dict,
                 video_filename: Optional[str] = None,
                 video_fps: float = 30.0):
        self.episode_id = episode_id
        self.episode_dir = episode_dir
        self.task_id = task_id
        self.instruction = instruction
        self.num_times = num_times
        self.ckpt_path = ckpt_path
        self.config_path = config_path
        self.runtime_meta = dict(runtime_meta or {})
        self.inference_config_snapshot = inference_config_snapshot or {}
        self.video_filename = video_filename
        self.video_fps = float(video_fps or 30.0)

        self.events_path = os.path.join(episode_dir, EVENTS_FILENAME)
        self.jointstate_path = os.path.join(episode_dir, JOINTSTATE_FILENAME)
        self.meta_path = os.path.join(episode_dir, META_FILENAME)
        self.video_path = (os.path.join(episode_dir, video_filename)
                           if video_filename else None)

        os.makedirs(episode_dir, exist_ok=True)
        self._events_fp = open(self.events_path, 'a', buffering=1)
        self._jointstate_fp = open(self.jointstate_path, 'a', buffering=1)
        self._lock = threading.Lock()

        self._t_episode_start_wall = time.time()
        self._start_iso = _iso_now()
        self._n_inferences = 0
        self._n_actions_published = 0
        self._first_publish_done = False
        self._repeat_counter = 0
        self._end_iso: Optional[str] = None
        self._duration_s: Optional[float] = None
        self._video_writer = None
        self._video_frame_size = None
        self._n_video_frames = 0

    def next_repeat_idx(self) -> int:
        self._repeat_counter += 1
        return self._repeat_counter

    def _write_event_line(self, payload: Dict):
        line = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        with self._lock:
            self._events_fp.write(line + '\n')

    def write_event_episode_start(self, publish_rate, action_chunk,
                                  arm_action_dim, dry_run, async_execution,
                                  binarize_gripper):
        payload = {
            'event': 'episode_start',
            'episode_id': self.episode_id,
            't_wall': self._t_episode_start_wall,
            'start_iso': self._start_iso,
            'task_id': self.task_id,
            'instruction': self.instruction,
            'num_times': self.num_times,
            'ckpt_path': self.ckpt_path,
            'config_path': self.config_path,
            'dry_run': bool(dry_run),
            'async_execution': bool(async_execution),
            'binarize_gripper': bool(binarize_gripper),
            'publish_rate': (int(publish_rate)
                             if publish_rate is not None else None),
            'action_chunk': (int(action_chunk)
                             if action_chunk is not None else None),
            'arm_action_dim': (int(arm_action_dim)
                               if arm_action_dim is not None else None),
        }
        self._write_event_line(payload)

    def write_event_inference(self, ctx, instruction: str):
        repeat_idx = self.next_repeat_idx()
        t_inference_start = float(getattr(ctx, 'inference_start', 0.0) or 0.0)
        elapsed = float(getattr(ctx, 'inference_elapsed', 0.0) or 0.0)
        t_inference_end = t_inference_start + elapsed
        t_obs = getattr(ctx, 't_obs', None)
        latency_ms = None
        if t_obs is not None:
            try:
                latency_ms = (t_inference_end - float(t_obs)) * 1000.0
            except (TypeError, ValueError):
                latency_ms = None
        raw_actions = getattr(ctx, 'raw_actions', None)
        action_shape = None
        if raw_actions is not None:
            try:
                action_shape = [int(s) for s in raw_actions.shape]
            except Exception:
                action_shape = None
        is_dry_run = bool(self.runtime_meta.get('dry_run', False))
        payload = {
            'event': 'inference',
            'episode_id': self.episode_id,
            't_wall': time.time(),
            'repeat_idx': repeat_idx,
            'instruction': instruction,
            't_obs': float(t_obs) if t_obs is not None else None,
            't_inference_start': t_inference_start,
            't_inference_end': t_inference_end,
            'inference_elapsed_ms': elapsed * 1000.0,
            'latency_ms': latency_ms,
            'action_chunk_shape': action_shape,
            'is_dry_run': is_dry_run,
        }
        self._write_event_line(payload)
        self._n_inferences += 1
        try:
            ctx._metrics_repeat_idx = repeat_idx
        except Exception:
            pass

    def write_event_action_publish(self, ctx, n_actions: int, dt: float,
                                   arm_action_dim: int, gripper_dim: int,
                                   is_dry_run: bool):
        repeat_idx = getattr(ctx, '_metrics_repeat_idx', None)
        is_first = not self._first_publish_done
        if is_first:
            self._first_publish_done = True
        t_first_publish = float(
            getattr(ctx, 't_first_publish', None) or time.time())
        payload = {
            'event': 'action_publish',
            'episode_id': self.episode_id,
            't_wall': time.time(),
            'repeat_idx': repeat_idx,
            't_first_publish': t_first_publish,
            'n_actions': int(n_actions),
            'dt': float(dt),
            'arm_action_dim': int(arm_action_dim),
            'gripper_dim': int(gripper_dim),
            'is_dry_run': bool(is_dry_run),
            'is_first_action_in_episode': is_first,
        }
        self._write_event_line(payload)
        self._n_actions_published += int(n_actions)

    def write_event_episode_end(self, reason: str = 'completed') -> Dict:
        end_wall = time.time()
        self._end_iso = _iso_now()
        self._duration_s = end_wall - self._t_episode_start_wall
        payload = {
            'event': 'episode_end',
            'episode_id': self.episode_id,
            't_wall': end_wall,
            'end_iso': self._end_iso,
            'duration_s': self._duration_s,
            'n_inferences': self._n_inferences,
            'n_actions_published': self._n_actions_published,
            'reason': reason,
        }
        if self.video_path is not None:
            payload['video_path'] = os.path.basename(self.video_path)
            payload['n_video_frames'] = self._n_video_frames
        self._write_event_line(payload)
        self._flush_meta()
        summary = {
            'episode_id': self.episode_id,
            'start_iso': self._start_iso,
            'end_iso': self._end_iso,
            'duration_s': self._duration_s,
            'task_id': self.task_id,
            'instruction': self.instruction,
            'num_times': self.num_times,
            'n_inferences': self._n_inferences,
            'n_actions_published': self._n_actions_published,
            'episode_dir': self.episode_id,
            'events_path': f'{self.episode_id}/{EVENTS_FILENAME}',
            'jointstate_path': f'{self.episode_id}/{JOINTSTATE_FILENAME}',
            'meta_path': f'{self.episode_id}/{META_FILENAME}',
            'reason': reason,
        }
        if self.video_path is not None:
            summary['video_path'] = f'{self.episode_id}/{self.video_filename}'
            summary['n_video_frames'] = self._n_video_frames
        return summary

    def write_video_frame(self, frame):
        if self.video_path is None:
            return
        import cv2
        import numpy as np

        frame = np.asarray(frame)
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        if frame.ndim != 3 or frame.shape[2] != 3:
            return
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        height, width = frame.shape[:2]
        frame_size = (int(width), int(height))
        with self._lock:
            if self._video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self._video_writer = cv2.VideoWriter(
                    self.video_path, fourcc, self.video_fps, frame_size)
                self._video_frame_size = frame_size
            if frame_size != self._video_frame_size:
                frame = cv2.resize(frame, self._video_frame_size)
            self._video_writer.write(frame)
            self._n_video_frames += 1

    def write_jointstate(self, msg):
        try:
            t_ros = float(msg.header.stamp.to_sec())
        except Exception:
            t_ros = None
        try:
            joint_pos = [float(x) for x in getattr(msg, 'joint_pos', []) or []]
        except Exception:
            joint_pos = []
        try:
            joint_vel = [float(x) for x in getattr(msg, 'joint_vel', []) or []]
        except Exception:
            joint_vel = []
        try:
            joint_cur = [float(x) for x in getattr(msg, 'joint_cur', []) or []]
        except Exception:
            joint_cur = []
        try:
            mode = int(getattr(msg, 'mode', 0))
        except Exception:
            mode = 0
        payload = {
            't_wall': time.time(),
            't_ros': t_ros,
            'joint_pos': joint_pos,
            'joint_vel': joint_vel,
            'joint_cur': joint_cur,
            'mode': mode,
            'episode_id': self.episode_id,
        }
        line = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        with self._lock:
            self._jointstate_fp.write(line + '\n')

    def _flush_meta(self):
        git_commit, git_dirty = _safe_git_info(os.getcwd())
        meta = {
            'episode_id': self.episode_id,
            'start_iso': self._start_iso,
            'end_iso': self._end_iso,
            'duration_s': self._duration_s,
            'task_id': self.task_id,
            'instruction': self.instruction,
            'num_times': self.num_times,
            'ckpt_path': self.ckpt_path,
            'config_path': self.config_path,
            'inference_config_snapshot': self.inference_config_snapshot,
            'runtime': self.runtime_meta,
            'video': {
                'path': os.path.basename(self.video_path)
                if self.video_path else None,
                'fps': self.video_fps if self.video_path else None,
                'n_frames': self._n_video_frames
                if self.video_path else None,
            },
            'host': {
                'hostname': socket.gethostname(),
                'pid': os.getpid(),
                'python_version': sys.version,
            },
            'code_version': {
                'git_commit': git_commit,
                'git_dirty': git_dirty,
            },
        }
        try:
            with open(self.meta_path, 'w') as fp:
                json.dump(meta, fp, ensure_ascii=False, indent=2)
        except Exception as e:
            overwatch.warning(f'Failed to write meta.json: {e}')

    def close(self):
        with self._lock:
            try:
                if self._video_writer is not None:
                    self._video_writer.release()
                    self._video_writer = None
            except Exception:
                pass
            try:
                self._events_fp.close()
            except Exception:
                pass
            try:
                self._jointstate_fp.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# MetricsManager (plan section 4.4)
# ---------------------------------------------------------------------------
class MetricsManager:
    """Runner-scoped manager for metrics collection.

    Owns the 100Hz ROS subscription on ``joint_state_topic`` and dispatches
    callbacks to whichever EpisodeWriter is currently active.
    """

    def __init__(self,
                 output_root: str,
                 joint_state_topic: str,
                 joint_state_msg_type: str,
                 enabled: bool = True,
                 subscribe_queue_size: int = 200,
                 video_topic: Optional[str] = None,
                 video_msg_type: Optional[str] = None,
                 video_filename: str = DEFAULT_VIDEO_FILENAME,
                 video_fps: float = 30.0,
                 video_queue_size: int = 30,
                 runtime_meta_provider: Optional[Callable[[], Dict]] = None,
                 inference_config_provider:
                 Optional[Callable[[], Dict]] = None,
                 ckpt_path: Optional[str] = None,
                 config_path: Optional[str] = None):
        self.output_root = os.path.abspath(output_root)
        self.joint_state_topic = joint_state_topic
        self.joint_state_msg_type = joint_state_msg_type
        self.enabled = enabled
        self.subscribe_queue_size = subscribe_queue_size
        self.video_topic = video_topic
        self.video_msg_type = video_msg_type or 'sensor_msgs.msg:Image'
        self.video_filename = video_filename
        self.video_fps = float(video_fps or 30.0)
        self.video_queue_size = int(video_queue_size)
        self.runtime_meta_provider = (runtime_meta_provider
                                      if runtime_meta_provider is not None
                                      else lambda: {})
        self.inference_config_provider = (inference_config_provider
                                          if inference_config_provider
                                          is not None else lambda: {})
        self.ckpt_path = ckpt_path
        self.config_path = config_path

        self._active_writer: Optional[EpisodeWriter] = None
        self._active_lock = threading.Lock()
        self._episode_counters: Dict[str, int] = {}
        self._subscriber = None
        self._video_subscriber = None
        self._cv_bridge = None
        self._manifest_path = os.path.join(self.output_root,
                                           MANIFEST_FILENAME)
        self._manifest_lock = threading.Lock()

    def start(self):
        os.makedirs(self.output_root, exist_ok=True)
        self._init_manifest_if_missing()
        if not self.enabled:
            return
        import rospy
        msg_cls = _import_ros_msg_cls(self.joint_state_msg_type)
        self._subscriber = rospy.Subscriber(
            self.joint_state_topic,
            msg_cls,
            self._on_joint_state,
            queue_size=self.subscribe_queue_size,
            tcp_nodelay=True)
        overwatch.info(
            f'MetricsManager subscribed to {self.joint_state_topic} '
            f'(msg_type={self.joint_state_msg_type})')
        if self.video_topic:
            try:
                from cv_bridge import CvBridge
                self._cv_bridge = CvBridge()
                video_msg_cls = _import_ros_msg_cls(self.video_msg_type)
                self._video_subscriber = rospy.Subscriber(
                    self.video_topic,
                    video_msg_cls,
                    self._on_video_image,
                    queue_size=self.video_queue_size,
                    tcp_nodelay=True)
                overwatch.info(
                    f'MetricsManager recording video from {self.video_topic} '
                    f'to {self.video_filename}')
            except Exception as e:
                overwatch.warning(f'Failed to initialize video recording: {e}')

    def _init_manifest_if_missing(self):
        if not os.path.exists(self._manifest_path):
            try:
                with open(self._manifest_path, 'a'):
                    pass
            except Exception as e:
                overwatch.warning(f'Failed to init manifest: {e}')

    def _on_joint_state(self, msg):
        with self._active_lock:
            writer = self._active_writer
        if writer is None:
            return
        try:
            writer.write_jointstate(msg)
        except Exception as e:
            overwatch.warning(f'Failed to write jointstate: {e}')

    def _on_video_image(self, msg):
        with self._active_lock:
            writer = self._active_writer
        if writer is None or self._cv_bridge is None:
            return
        try:
            frame = self._cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
            writer.write_video_frame(frame)
        except Exception as e:
            overwatch.warning(f'Failed to write video frame: {e}')

    def start_episode(self, task_id: str, instruction: str,
                      num_times: int) -> EpisodeWriter:
        ts = time.time()
        cnt = self._episode_counters.get(task_id, 0) + 1
        self._episode_counters[task_id] = cnt
        episode_id = _make_episode_id(task_id, cnt, ts)
        episode_dir = os.path.join(self.output_root, episode_id)
        try:
            runtime_meta = self.runtime_meta_provider()
        except Exception as e:
            overwatch.warning(f'runtime_meta_provider failed: {e}')
            runtime_meta = {}
        try:
            inference_cfg = self.inference_config_provider()
        except Exception as e:
            overwatch.warning(f'inference_config_provider failed: {e}')
            inference_cfg = {}
        writer = EpisodeWriter(
            episode_id=episode_id,
            episode_dir=episode_dir,
            task_id=task_id,
            instruction=instruction,
            num_times=num_times,
            ckpt_path=self.ckpt_path,
            config_path=self.config_path,
            runtime_meta=runtime_meta,
            inference_config_snapshot=inference_cfg,
            video_filename=self.video_filename if self.video_topic else None,
            video_fps=self.video_fps)
        writer.write_event_episode_start(
            publish_rate=runtime_meta.get('publish_rate'),
            action_chunk=runtime_meta.get('action_chunk'),
            arm_action_dim=runtime_meta.get('arm_action_dim'),
            dry_run=runtime_meta.get('dry_run', False),
            async_execution=runtime_meta.get('async_execution', False),
            binarize_gripper=runtime_meta.get('binarize_gripper', False))
        with self._active_lock:
            self._active_writer = writer
        return writer

    def close_current_episode(self, reason: str = 'completed'):
        with self._active_lock:
            writer = self._active_writer
            self._active_writer = None
        if writer is None:
            return
        try:
            summary = writer.write_event_episode_end(reason=reason)
            self._append_manifest(summary)
        finally:
            writer.close()

    def _append_manifest(self, summary: Dict):
        line = json.dumps(summary, ensure_ascii=False, allow_nan=False)
        with self._manifest_lock:
            try:
                with open(self._manifest_path, 'a', buffering=1) as fp:
                    fp.write(line + '\n')
            except Exception as e:
                overwatch.warning(f'Failed to append manifest: {e}')

    @contextlib.contextmanager
    def episode(self, task_id: str, instruction: str, num_times: int):
        self.start_episode(task_id, instruction, num_times)
        reason = 'completed'
        try:
            yield self._active_writer
        except KeyboardInterrupt:
            reason = 'interrupted'
            raise
        except Exception:
            reason = 'exception'
            raise
        finally:
            self.close_current_episode(reason=reason)

    @contextlib.contextmanager
    def inference(self, ctx):
        try:
            yield
        finally:
            with self._active_lock:
                writer = self._active_writer
            if writer is None:
                return
            instruction = getattr(ctx, 'instruction', '')
            try:
                writer.write_event_inference(ctx, instruction)
            except Exception as e:
                overwatch.warning(f'Failed to write inference event: {e}')

    def action_publish(self, ctx, n_actions: int, dt: float,
                       arm_action_dim: int, gripper_dim: int,
                       is_dry_run: bool):
        with self._active_lock:
            writer = self._active_writer
        if writer is None:
            return
        try:
            writer.write_event_action_publish(
                ctx,
                n_actions=n_actions,
                dt=dt,
                arm_action_dim=arm_action_dim,
                gripper_dim=gripper_dim,
                is_dry_run=is_dry_run)
        except Exception as e:
            overwatch.warning(f'Failed to write action_publish event: {e}')

    def close(self):
        try:
            if self._subscriber is not None:
                self._subscriber.unregister()
                self._subscriber = None
        except Exception:
            pass
        try:
            if self._video_subscriber is not None:
                self._video_subscriber.unregister()
                self._video_subscriber = None
        except Exception:
            pass
        with self._active_lock:
            writer = self._active_writer
            self._active_writer = None
        if writer is not None:
            try:
                summary = writer.write_event_episode_end(reason='shutdown')
                self._append_manifest(summary)
            finally:
                writer.close()


# ---------------------------------------------------------------------------
# Factory (plan section 4.5)
# ---------------------------------------------------------------------------
def build_metrics_manager_from_cfg(
        cfg: Optional[Dict],
        runtime_meta_provider: Callable[[], Dict],
        inference_config_provider: Callable[[], Dict],
        ckpt_path: Optional[str] = None,
        config_path: Optional[str] = None) -> Optional[MetricsManager]:
    if cfg is None:
        return None
    if not cfg.get('enabled', True):
        return None
    manager = MetricsManager(
        output_root=cfg.get('output_root', 'work_dirs/metrics'),
        joint_state_topic=cfg['joint_state_topic'],
        joint_state_msg_type=cfg['joint_state_msg_type'],
        enabled=True,
        subscribe_queue_size=cfg.get('subscribe_queue_size', 200),
        video_topic=cfg.get('video_topic'),
        video_msg_type=cfg.get('video_msg_type', 'sensor_msgs.msg:Image'),
        video_filename=cfg.get('video_filename', DEFAULT_VIDEO_FILENAME),
        video_fps=cfg.get('video_fps', 30.0),
        video_queue_size=cfg.get('video_queue_size', 30),
        runtime_meta_provider=runtime_meta_provider,
        inference_config_provider=inference_config_provider,
        ckpt_path=ckpt_path,
        config_path=config_path)
    manager.start()
    return manager


class SimMetricsManager:
    """Metrics manager for simulator rollouts.

    Writes the same episode directory layout as ``MetricsManager`` without a
    ROS subscription. Simulator state samples are written into
    ``jointstate.jsonl`` so the existing offline analyzer can be reused.
    """

    def __init__(self,
                 output_root: str,
                 control_freq_hz: float = 30.0,
                 runtime_meta_provider: Optional[Callable[[], Dict]] = None,
                 inference_config_provider:
                 Optional[Callable[[], Dict]] = None,
                 ckpt_path: Optional[str] = None,
                 config_path: Optional[str] = None,
                 worker_id: Optional[str] = None):
        self.output_root = os.path.abspath(output_root)
        self.control_freq_hz = float(control_freq_hz)
        self.runtime_meta_provider = (runtime_meta_provider
                                      if runtime_meta_provider is not None
                                      else lambda: {})
        self.inference_config_provider = (inference_config_provider
                                          if inference_config_provider
                                          is not None else lambda: {})
        self.ckpt_path = ckpt_path
        self.config_path = config_path
        self.worker_id = worker_id
        self._active_writer: Optional[EpisodeWriter] = None
        self._active_lock = threading.Lock()
        self._episode_counters: Dict[str, int] = {}
        self._manifest_path = os.path.join(self.output_root,
                                           MANIFEST_FILENAME)
        self._success_csv_path = os.path.join(self.output_root,
                                              SUCCESS_CSV_FILENAME)
        self._manifest_lock = threading.Lock()
        self._success_lock = threading.Lock()

    def start(self):
        os.makedirs(self.output_root, exist_ok=True)
        self._init_file_if_missing(self._manifest_path)
        self._init_success_csv_if_missing()
        overwatch.info(
            f'SimMetricsManager writing LIBERO metrics to {self.output_root}')

    def _init_file_if_missing(self, path: str):
        if not os.path.exists(path):
            try:
                with open(path, 'a'):
                    pass
            except Exception as e:
                overwatch.warning(f'Failed to init {path}: {e}')

    def _init_success_csv_if_missing(self):
        try:
            with open(self._success_csv_path, 'x', newline='') as fp:
                writer = csv.DictWriter(
                    fp,
                    fieldnames=[
                        'episode_id', 'success', 'task_id', 'instruction',
                        'reason'
                    ])
                writer.writeheader()
        except FileExistsError:
            return
        except Exception as e:
            overwatch.warning(f'Failed to init success.csv: {e}')

    def start_episode(self, task_id: str, instruction: str,
                      num_times: int) -> EpisodeWriter:
        ts = time.time()
        task_key = f'{self.worker_id}_{task_id}' if self.worker_id else task_id
        cnt = self._episode_counters.get(task_key, 0) + 1
        self._episode_counters[task_key] = cnt
        episode_id = _make_episode_id(task_key, cnt, ts)
        episode_dir = os.path.join(self.output_root, episode_id)
        try:
            runtime_meta = self.runtime_meta_provider()
        except Exception as e:
            overwatch.warning(f'runtime_meta_provider failed: {e}')
            runtime_meta = {}
        try:
            inference_cfg = self.inference_config_provider()
        except Exception as e:
            overwatch.warning(f'inference_config_provider failed: {e}')
            inference_cfg = {}
        runtime_meta = dict(runtime_meta or {})
        runtime_meta.setdefault('publish_rate', self.control_freq_hz)
        runtime_meta.setdefault('dt', 1.0 / self.control_freq_hz
                                if self.control_freq_hz > 0 else None)
        runtime_meta.setdefault('simulator', 'libero')
        writer = EpisodeWriter(
            episode_id=episode_id,
            episode_dir=episode_dir,
            task_id=task_id,
            instruction=instruction,
            num_times=num_times,
            ckpt_path=self.ckpt_path,
            config_path=self.config_path,
            runtime_meta=runtime_meta,
            inference_config_snapshot=inference_cfg)
        writer.write_event_episode_start(
            publish_rate=runtime_meta.get('publish_rate'),
            action_chunk=runtime_meta.get('action_chunk'),
            arm_action_dim=runtime_meta.get('arm_action_dim'),
            dry_run=runtime_meta.get('dry_run', False),
            async_execution=runtime_meta.get('async_execution', False),
            binarize_gripper=runtime_meta.get('binarize_gripper', False))
        with self._active_lock:
            self._active_writer = writer
        return writer

    def close_current_episode(self,
                              reason: str = 'completed',
                              success: Optional[bool] = None):
        with self._active_lock:
            writer = self._active_writer
            self._active_writer = None
        if writer is None:
            return
        try:
            summary = writer.write_event_episode_end(reason=reason)
            self._append_manifest(summary)
            if success is not None:
                self._append_success(summary, success)
        finally:
            writer.close()

    def _append_manifest(self, summary: Dict):
        line = json.dumps(summary, ensure_ascii=False, allow_nan=False)
        with self._manifest_lock:
            try:
                with open(self._manifest_path, 'a', buffering=1) as fp:
                    fp.write(line + '\n')
            except Exception as e:
                overwatch.warning(f'Failed to append manifest: {e}')

    def _append_success(self, summary: Dict, success: bool):
        row = {
            'episode_id': summary.get('episode_id'),
            'success': int(bool(success)),
            'task_id': summary.get('task_id'),
            'instruction': summary.get('instruction'),
            'reason': summary.get('reason'),
        }
        with self._success_lock:
            try:
                with open(self._success_csv_path, 'a', newline='') as fp:
                    writer = csv.DictWriter(
                        fp,
                        fieldnames=[
                            'episode_id', 'success', 'task_id',
                            'instruction', 'reason'
                        ])
                    writer.writerow(row)
            except Exception as e:
                overwatch.warning(f'Failed to append success.csv: {e}')

    @contextlib.contextmanager
    def episode(self, task_id: str, instruction: str, num_times: int):
        self.start_episode(task_id, instruction, num_times)
        reason = 'completed'
        success = None
        try:
            yield self._active_writer
        except KeyboardInterrupt:
            reason = 'interrupted'
            raise
        except Exception:
            reason = 'exception'
            raise
        finally:
            with self._active_lock:
                writer = self._active_writer
            if writer is not None and hasattr(writer, '_sim_success'):
                success = bool(getattr(writer, '_sim_success'))
            self.close_current_episode(reason=reason, success=success)

    @contextlib.contextmanager
    def inference(self, ctx):
        try:
            yield
        finally:
            with self._active_lock:
                writer = self._active_writer
            if writer is None:
                return
            instruction = getattr(ctx, 'instruction', '')
            try:
                writer.write_event_inference(ctx, instruction)
            except Exception as e:
                overwatch.warning(f'Failed to write inference event: {e}')

    def action_publish(self, ctx, n_actions: int, dt: float,
                       arm_action_dim: int, gripper_dim: int,
                       is_dry_run: bool):
        with self._active_lock:
            writer = self._active_writer
        if writer is None:
            return
        try:
            writer.write_event_action_publish(
                ctx,
                n_actions=n_actions,
                dt=dt,
                arm_action_dim=arm_action_dim,
                gripper_dim=gripper_dim,
                is_dry_run=is_dry_run)
        except Exception as e:
            overwatch.warning(f'Failed to write action_publish event: {e}')

    def write_sim_jointstate(self, t_sim: float, joint_pos, joint_vel):
        with self._active_lock:
            writer = self._active_writer
        if writer is None:
            return
        try:
            payload = {
                't_wall': time.time(),
                't_ros': float(t_sim),
                'joint_pos': [float(x) for x in joint_pos],
                'joint_vel': [float(x) for x in joint_vel],
                'joint_cur': [],
                'mode': 0,
                'episode_id': writer.episode_id,
            }
            line = json.dumps(payload, ensure_ascii=False, allow_nan=False)
            with writer._lock:
                writer._jointstate_fp.write(line + '\n')
        except Exception as e:
            overwatch.warning(f'Failed to write sim jointstate: {e}')

    def mark_success(self, success: bool):
        with self._active_lock:
            writer = self._active_writer
        if writer is not None:
            setattr(writer, '_sim_success', bool(success))

    def close(self):
        self.close_current_episode(reason='shutdown')


def build_sim_metrics_manager_from_cfg(
        cfg: Optional[Dict],
        runtime_meta_provider: Callable[[], Dict],
        inference_config_provider: Callable[[], Dict],
        ckpt_path: Optional[str] = None,
        config_path: Optional[str] = None,
        worker_id: Optional[str] = None) -> Optional[SimMetricsManager]:
    if cfg is None:
        return None
    if not cfg.get('enabled', True):
        return None
    manager = SimMetricsManager(
        output_root=cfg.get('output_root', 'work_dirs/metrics_libero'),
        control_freq_hz=cfg.get('control_freq_hz', 30.0),
        runtime_meta_provider=runtime_meta_provider,
        inference_config_provider=inference_config_provider,
        ckpt_path=ckpt_path,
        config_path=config_path,
        worker_id=worker_id)
    manager.start()
    return manager
