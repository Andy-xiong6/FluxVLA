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

import gc
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
import tqdm
from libero.libero import benchmark
from safetensors.torch import load_file

from fluxvla.engines.utils import (build_sim_metrics_manager_from_cfg,
                                   initialize_overwatch)
from fluxvla.engines.utils.eval_utils import (get_libero_dummy_action,
                                              get_libero_env,
                                              quat2axisangle,
                                              save_rollout_video)
from fluxvla.engines.utils.name_map import str_to_dtype
from fluxvla.engines.utils.torch_utils import set_seed_everywhere
from fluxvla.engines.utils.trajectory_postprocessor import (
    RuckigActionSmoother, TrajectoryStitcher, interpolate_action_trajectory)
from ..utils.root import RUNNERS

overwatch = initialize_overwatch(__name__)


def resample_remaining(traj, offset):
    """Linearly interpolate a remaining action trajectory."""
    n_steps = traj.shape[0]
    out_steps = n_steps - int(offset)
    if out_steps <= 0:
        return traj[:0]
    idx = np.clip(offset + np.arange(out_steps), 0.0, n_steps - 1.0)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, n_steps - 1)
    alpha = (idx - lo)[:, np.newaxis]
    return traj[lo] + alpha * (traj[hi] - traj[lo])


@RUNNERS.register_module()
class LiberoEvalRunner:
    """Runner for evaluating models using Hugging Face Transformers.
    This class sets up the evaluation environment, loads the model,
    and runs the evaluation process.
    Args:
        cfg (Dict): Configuration dictionary containing model and
            evaluation settings.
        seed (int): Random seed for reproducibility.
        ckpt_path (str): Path to the model checkpoint.
        model_family (str): Model family for evaluation.
        task_suite_name (str): Name of the task suite for evaluation.
        dataset (Dict): Configuration for the dataset to be used in evaluation.
        denormalize_action (Dict): Configuration for denormalizing actions.
        eval_chunk_size (int): Size of the chunks for evaluation.
            Default is 1.
        resize_size (int): Size to which images will be resized.
            Default is 224.
        num_trials_per_task (int): Number of trials per task in the evaluation.
            Default is 50.
        num_steps_wait (int): Number of steps to wait before
            starting evaluation.
            Default is 10.
        mixed_precision_dtype (str): Data type for mixed precision training.
            Default is 'bf16'.
        enable_mixed_precision_training (bool): Whether to enable mixed
            precision training.
            Default is True.
    """

    def __init__(self,
                 cfg: Dict,
                 seed: int,
                 ckpt_path: str,
                 model_family: str,
                 task_suite_name: str,
                 dataset: Dict,
                 denormalize_action: Dict,
                 eval_chunk_size: int = 1,
                 resize_size: int = 224,
                 num_trials_per_task: int = 50,
                 num_steps_wait: int = 10,
                 mixed_precision_dtype: str = 'bf16',
                 enable_mixed_precision_training: bool = True,
                 metrics: Dict = None,
                 rtc_config: Dict = None,
                 execute_horizon: int = None,
                 arm_action_dim: int = 6,
                 trajectory_stitching: Dict = None,
                 ruckig_smoothing: Dict = None,
                 interpolate_actions: bool = False,
                 action_interpolation_factor: int = 1,
                 config_path: str = None):
        from fluxvla.engines import (build_dataset_from_cfg,
                                     build_transform_from_cfg,
                                     build_vla_from_cfg)
        self.device_id = overwatch.local_rank()
        if hasattr(cfg, 'inference_model'):
            self.vla = build_vla_from_cfg(cfg.inference_model).eval()
        else:
            self.vla = build_vla_from_cfg(cfg.model).eval()
        # Load checkpoint weights if ckpt_path is provided
        if ckpt_path is not None:
            assert Path.exists(Path(ckpt_path)), \
                f'Checkpoint path {ckpt_path} does not exist!'

            if ckpt_path.endswith('.safetensors'):
                state_dict = load_file(ckpt_path, device='cpu')
            else:
                # A sibling .safetensors is preferred when available because
                # the .pt file also contains the optimizer/scheduler state
                # which is unnecessary for inference and quickly exhausts
                # CPU RAM when loaded on every rank (SIGKILL / exit -9).
                sf_candidate = (
                    ckpt_path[:-len('.pt')] +
                    '.safetensors' if ckpt_path.endswith('.pt') else None)
                if sf_candidate is not None and os.path.exists(sf_candidate):
                    state_dict = load_file(sf_candidate, device='cpu')
                else:
                    # mmap=True avoids copying the whole checkpoint into RAM
                    # on every rank.
                    try:
                        checkpoint = torch.load(
                            ckpt_path, map_location='cpu', mmap=True)
                    except TypeError:
                        checkpoint = torch.load(ckpt_path, map_location='cpu')
                    if isinstance(checkpoint, dict) and 'model' in checkpoint:
                        state_dict = checkpoint['model']
                        # Drop optimizer/scheduler state ASAP to reclaim RAM.
                        checkpoint.pop('optimizer_state_dict', None)
                        checkpoint.pop('scheduler_state_dict', None)
                        checkpoint.pop('optimizer_state_index_to_name', None)
                    else:
                        state_dict = checkpoint
                    del checkpoint
                    gc.collect()
            self.vla.load_state_dict(state_dict, strict=True)
            del state_dict
            gc.collect()
        self.cfg = cfg
        self.seed = seed
        self.ckpt_path = ckpt_path
        data_stat_path = os.path.join(
            Path(self.ckpt_path).resolve().parent.parent,
            'dataset_statistics.json')  # noqa: E501
        assert os.path.exists(data_stat_path), \
            f'Dataset statistics file not found at {data_stat_path}!'
        # Load dataset and denormalization action
        denormalize_action['norm_stats'] = data_stat_path
        dataset['task_suite_name'] = task_suite_name
        dataset['norm_stats'] = data_stat_path
        self.dataset = build_dataset_from_cfg(dataset)
        self.denormalize_action = build_transform_from_cfg(denormalize_action)
        self.eval_chunk_size = eval_chunk_size
        self.model_family = model_family
        self.task_suite_name = task_suite_name
        self.resize_size = resize_size
        self.num_trials_per_task = num_trials_per_task
        self.num_steps_wait = num_steps_wait
        self.mixed_precision_dtype = str_to_dtype(mixed_precision_dtype)
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.distributed_state = overwatch.distributed_state
        self.metrics_cfg = metrics
        self.rtc_config = rtc_config
        self.execute_horizon = execute_horizon
        self.arm_action_dim = int(arm_action_dim)
        self.trajectory_stitcher = TrajectoryStitcher(
            **(trajectory_stitching or {}))
        self.ruckig_smoother = RuckigActionSmoother(
            **(ruckig_smoothing or {}))
        self.interpolate_actions = bool(interpolate_actions)
        self.action_interpolation_factor = int(action_interpolation_factor)
        self.config_path = config_path
        self.metrics = None
        self._eval_cfg_snapshot = self._extract_eval_cfg_snapshot(cfg)
        try:
            self.metrics = build_sim_metrics_manager_from_cfg(
                self.metrics_cfg,
                runtime_meta_provider=self._build_metrics_runtime_meta,
                inference_config_provider=self._build_metrics_eval_config,
                ckpt_path=self.ckpt_path,
                config_path=self.config_path,
                worker_id=f'rank{overwatch.rank()}')
        except Exception as e:
            overwatch.warning(f'Sim metrics initialization failed: {e}. '
                              f'Continuing without metrics recording.')
            self.metrics = None

        if os.path.isfile(data_stat_path):
            with open(data_stat_path, 'r') as f:
                norm_stats = json.load(f)
            self.vla.norm_stats = norm_stats
        else:
            overwatch.warning(
                'WARNING: No local dataset_statistics.json file found for current checkpoint.\n'  # noqa: E501
                'You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint.'  # noqa: E501
                'Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`.'  # noqa: E501
            )

    def _extract_eval_cfg_snapshot(self, cfg) -> Dict:
        from fluxvla.engines.utils.metrics_recorder import _jsonable
        try:
            if hasattr(cfg, 'eval'):
                eval_cfg = cfg.eval
                if hasattr(eval_cfg, 'to_dict'):
                    raw = eval_cfg.to_dict()
                elif hasattr(eval_cfg, '_cfg_dict'):
                    raw = dict(eval_cfg._cfg_dict)
                else:
                    raw = dict(eval_cfg)
            else:
                raw = {}
            for drop_key in ('cfg', 'config_path'):
                raw.pop(drop_key, None)
            return _jsonable(raw)
        except Exception as e:
            overwatch.warning(f'Failed to snapshot eval cfg: {e}')
            return {}

    def _build_metrics_runtime_meta(self) -> Dict:
        control_freq_hz = 30.0
        if self.metrics_cfg is not None:
            control_freq_hz = self.metrics_cfg.get('control_freq_hz',
                                                   control_freq_hz)
        return {
            'publish_rate': control_freq_hz,
            'dt': 1.0 / control_freq_hz if control_freq_hz else None,
            'action_chunk': self.eval_chunk_size,
            'arm_action_dim': self.arm_action_dim,
            'gripper_dim': 1,
            'dry_run': False,
            'async_execution': False,
            'binarize_gripper': False,
            'simulator': 'libero',
            'task_suite_name': self.task_suite_name,
            'execute_horizon': self.execute_horizon,
            'rtc_config': self.rtc_config,
            'trajectory_stitching_enabled': bool(
                getattr(self.trajectory_stitcher, 'enabled', False)),
            'ruckig_smoothing_enabled': bool(
                getattr(self.ruckig_smoother, 'enabled', False)),
            'interpolate_actions': self.interpolate_actions,
            'action_interpolation_factor': self.action_interpolation_factor,
        }

    def _build_metrics_eval_config(self) -> Dict:
        return self._eval_cfg_snapshot or {}

    def _metrics_episode_ctx(self, task_id: str, instruction: str,
                             num_times: int):
        if self.metrics is None:
            return nullcontext()
        return self.metrics.episode(task_id, instruction, num_times)

    def _metrics_inference_ctx(self, ctx):
        if self.metrics is None:
            return nullcontext()
        return self.metrics.inference(ctx)

    def _emit_action_publish(self, ctx, n_actions: int):
        if self.metrics is None or n_actions <= 0:
            return
        dt = getattr(ctx, 'executed_dt', None)
        if dt is None:
            control_freq_hz = self._build_metrics_runtime_meta()[
                'publish_rate']
            dt = 1.0 / control_freq_hz if control_freq_hz else 0.0
        self.metrics.action_publish(
            ctx,
            n_actions=n_actions,
            dt=dt,
            arm_action_dim=self.arm_action_dim,
            gripper_dim=1,
            is_dry_run=False)

    def _mark_episode_success(self, success: bool):
        if self.metrics is not None:
            self.metrics.mark_success(success)

    def _get_tensor_device_dtype(self, batch: Dict):
        if torch.is_tensor(batch.get('states')):
            return batch['states'].device, batch['states'].dtype
        for value in batch.values():
            if torch.is_tensor(value):
                dtype = value.dtype if value.dtype.is_floating_point else (
                    self.mixed_precision_dtype)
                return value.device, dtype
        return (torch.device(f'cuda:{self.device_id}'),
                self.mixed_precision_dtype)

    def _rtc_enabled(self):
        return bool(self.rtc_config and self.rtc_config.get('enabled', False))

    def _apply_rtc_inputs(self, batch: Dict, prev_ctx, sim_step: int):
        if not self._rtc_enabled():
            return
        if prev_ctx is None or getattr(prev_ctx, 'raw_actions', None) is None:
            return
        offset = sim_step - getattr(prev_ctx, 'action_sim_step', sim_step)
        remaining = resample_remaining(prev_ctx.raw_actions[0], offset)
        if remaining.shape[0] == 0:
            return
        prefix_len = self.rtc_config.get('prefix_len')
        if prefix_len is None:
            prefix_len = remaining.shape[0]
        prefix_len = min(int(prefix_len), remaining.shape[0])
        if prefix_len <= 0:
            return
        device, dtype = self._get_tensor_device_dtype(batch)
        batch['prev_actions'] = torch.from_numpy(remaining[None]).to(
            device=device, dtype=dtype)
        batch['prefix_len'] = prefix_len
        batch['rtc_config'] = self.rtc_config

    def _actions_to_numpy(self, actions):
        if len(actions.shape) == 3:
            action_np = actions.float().cpu().numpy()
            return action_np, action_np[0, :self.eval_chunk_size, :]
        assert len(actions.shape) == 2, \
            f'Unexpected action shape: {actions.shape}'
        action_np = actions.float().cpu().numpy()
        return action_np[None], action_np[0, None, :]

    def _num_actions_to_execute(self, actions):
        if self.execute_horizon is None:
            if self._rtc_enabled():
                prefix_len = self.rtc_config.get('prefix_len')
                if prefix_len is not None and int(prefix_len) > 0:
                    prefix_len = min(int(prefix_len), actions.shape[0] - 1)
                    return max(actions.shape[0] - prefix_len, 1)
            return actions.shape[0]
        return min(max(int(self.execute_horizon), 1), actions.shape[0])

    def _control_dt(self) -> float:
        control_freq_hz = self._build_metrics_runtime_meta()['publish_rate']
        return 1.0 / control_freq_hz if control_freq_hz else 1.0 / 30.0

    def _denormalize_action_chunk(self, actions: np.ndarray) -> np.ndarray:
        denormed = []
        for action in actions:
            inputs = dict(
                action=action,
                task_suite_name=self.task_suite_name,
            )
            denormed.append(np.asarray(self.denormalize_action(inputs)))
        return np.asarray(denormed)

    def _postprocess_action_chunk(self, actions: np.ndarray, ctx, prev_ctx,
                                  sim_step: int):
        publish_dt = self._control_dt()
        if prev_ctx is not None:
            prev_actions = getattr(prev_ctx, 'executed_actions', None)
            prev_dt = getattr(prev_ctx, 'executed_dt', None)
            prev_start = getattr(prev_ctx, 'executed_sim_step', None)
            elapsed = None
            if prev_start is not None:
                elapsed = (sim_step - int(prev_start)) * float(prev_dt)
            actions = self.trajectory_stitcher.stitch(
                actions,
                arm_action_dim=self.arm_action_dim,
                previous_actions=prev_actions,
                elapsed_since_previous_start=elapsed,
                previous_dt=prev_dt)

        ruckig_meta = {'applied': False, 'error': None}
        if getattr(self.ruckig_smoother, 'enabled', False):
            actions, ruckig_meta = self.ruckig_smoother.smooth(
                actions, arm_action_dim=self.arm_action_dim, dt=publish_dt)

        interpolation_factor = int(self.action_interpolation_factor)
        should_interpolate = (
            self.interpolate_actions and interpolation_factor > 1 and
            (not ruckig_meta.get('applied')))
        if should_interpolate:
            actions = interpolate_action_trajectory(
                actions, interpolation_factor, self.arm_action_dim)
            publish_dt = publish_dt / interpolation_factor

        if (ruckig_meta.get('error') and
                not self.ruckig_smoother.fallback_to_linear):
            raise RuntimeError('Ruckig smoothing failed: '
                               f"{ruckig_meta['error']}")

        ctx.executed_actions = np.asarray(actions).copy()
        ctx.executed_dt = publish_dt
        ctx.executed_sim_step = sim_step
        return actions

    def _sim_joint_pos_from_obs(self, obs: Dict):
        try:
            pos = np.asarray(obs.get('robot0_eef_pos', []),
                             dtype=np.float64).reshape(-1)[:3]
            quat = np.asarray(obs.get('robot0_eef_quat', []),
                              dtype=np.float64).reshape(-1)[:4]
            if quat.shape[0] == 4:
                axisangle = quat2axisangle(quat.copy())
            else:
                axisangle = np.zeros(3, dtype=np.float64)
            gripper = np.asarray(obs.get('robot0_gripper_qpos', []),
                                 dtype=np.float64).reshape(-1)
            grip_val = gripper[:1] if gripper.size else np.zeros(1)
            joint_pos = np.concatenate([pos, axisangle, grip_val])
            if joint_pos.shape[0] < 7:
                joint_pos = np.pad(joint_pos, (0, 7 - joint_pos.shape[0]))
            return joint_pos[:7]
        except Exception:
            return np.zeros(7, dtype=np.float64)

    def _write_sim_jointstate(self, obs: Dict, action, sim_step: int):
        if self.metrics is None:
            return
        control_freq_hz = self._build_metrics_runtime_meta()['publish_rate']
        dt = 1.0 / control_freq_hz if control_freq_hz else 0.0
        joint_pos = self._sim_joint_pos_from_obs(obs)
        joint_vel = np.asarray(action, dtype=np.float64).reshape(-1)[:7]
        if joint_vel.shape[0] < 7:
            joint_vel = np.pad(joint_vel, (0, 7 - joint_vel.shape[0]))
        self.metrics.write_sim_jointstate(
            t_sim=sim_step * dt,
            joint_pos=joint_pos,
            joint_vel=joint_vel)

    def run_setup(self):
        """Set up the evaluation environment and model."""
        set_seed_everywhere(self.seed)
        torch.cuda.set_device(device_id := self.device_id)  # noqa: F841
        self.vla.eval()
        self.vla.freeze_vision_backbone = True
        self.vla.freeze_llm_backbone = True
        self.vla.freeze_projector = True
        self.vla.freeze_vlm_backbone = True
        if self.enable_mixed_precision_training:
            self.vla.to(
                device=self.device_id, dtype=self.mixed_precision_dtype)
        else:
            self.vla.cuda(self.device_id)

    def run(self):
        """Run the evaluation process."""
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[self.task_suite_name]()
        num_tasks_in_suite = task_suite.n_tasks
        global_episodes = list(
            range(num_tasks_in_suite * self.num_trials_per_task))
        overwatch.info(f'Task suite: {self.task_suite_name}')
        overwatch.info(f'Running evaluation on {num_tasks_in_suite} tasks '
                       f'with {self.num_trials_per_task} trials each.')
        overwatch.info(f'Using model family: {self.model_family}')
        overwatch.info(f'Using resize size: {self.resize_size}')
        overwatch.info(f'Using evaluation chunk size: {self.eval_chunk_size}')
        overwatch.info(
            f'Using mixed precision dtype: {self.mixed_precision_dtype}')
        rank = overwatch.rank()
        world_size = overwatch.world_size()
        local_episodes = global_episodes[rank::world_size]
        num_local_episodes = math.ceil(len(global_episodes) / world_size)
        data_time = time.strftime('%Y_%m_%d-%H_%M_%S')
        run_id = f'EVAL-{self.task_suite_name}-{self.model_family}-{data_time}'  # noqa: E501
        local_log_filepath = os.path.join(
            Path(self.ckpt_path).resolve().parent.parent, run_id + '.txt')
        log_file = open(local_log_filepath, 'w')
        total_episodes, total_successes = torch.zeros(
            1, device=torch.cuda.current_device()), torch.zeros(
                1, device=torch.cuda.current_device())
        unnorm_key = self.task_suite_name
        if rank == 0:
            pbar = tqdm.tqdm(
                total=len(global_episodes),
                desc='Evaluation',
                dynamic_ncols=True)
        else:
            pbar = None
        if self.model_family == 'openvla':
            # In some cases, the key must be manually modified (e.g. after
            # training on a modified version of the dataset
            # with the suffix "_no_noops" in the dataset name)
            candidate_unnorm_key = f'{unnorm_key}_no_noops'
            if (unnorm_key not in self.vla.norm_stats
                    and candidate_unnorm_key in self.vla.norm_stats):
                unnorm_key = candidate_unnorm_key
            assert unnorm_key in self.vla.norm_stats, (
                f'Action un-norm key {unnorm_key} '
                'not found in VLA norm_stats!')
        for id in range(num_local_episodes):
            done = False
            if id >= len(local_episodes):
                step_tensor = torch.zeros(
                    1, device=torch.cuda.current_device())
            else:
                local_id = local_episodes[id]
                # Get task ID from local episode index
                task_id = local_id // self.num_trials_per_task
                # Get trial ID within the task
                trial_id = local_id % self.num_trials_per_task

                # Log the current task and trial
                overwatch.info(f'Evaluating Task {task_id}, Trial {trial_id}')
                log_file.write(
                    f'Evaluating Task {task_id}, Trial {trial_id}\n')

                # Initialize the task suite and environment
                # Get task
                task = task_suite.get_task(task_id)

                # Get default LIBERO initial states
                initial_states = task_suite.get_task_init_states(task_id)

                # Initialize LIBERO environment and task description
                env, task_description = get_libero_env(task, resolution=256)
                overwatch.info(f'\nTask: {task_description}')
                log_file.write(f'\nTask: {task_description}\n')

                # Reset environment
                env.reset()

                # Set initial states
                obs = env.set_init_state(initial_states[trial_id])
                is_new_episode = True

                # Setup
                t = 0
                replay_images = []
                if self.task_suite_name == 'libero_spatial':
                    max_steps = 220  # longest training demo has 193 steps
                elif self.task_suite_name == 'libero_object':
                    max_steps = 280  # longest training demo has 254 steps
                elif self.task_suite_name == 'libero_goal':
                    max_steps = 300  # longest training demo has 270 steps
                elif self.task_suite_name == 'libero_10':
                    max_steps = 520  # longest training demo has 505 steps
                elif self.task_suite_name == 'libero_90':
                    max_steps = 400  # longest training demo has 373 steps

                overwatch.info(f'Starting episode {trial_id+1}...')

                log_file.write(f'Starting episode {trial_id+1}...\n')
                metrics_task_id = f'task{task_id:02d}_trial{trial_id:03d}'
                sim_step = 0
                prev_ctx = None
                with self._metrics_episode_ctx(
                        metrics_task_id, task_description,
                        self.num_trials_per_task):
                    while t < max_steps + self.num_steps_wait:
                        # IMPORTANT: Do nothing for the first
                        # few timesteps
                        # because the simulator drops objects
                        # and we need to wait for them to fall
                        if t < self.num_steps_wait:
                            obs, reward, done, info = env.step(
                                get_libero_dummy_action())
                            t += 1
                            continue
                        obs['task_description'] = task_description
                        obs['is_new_episode'] = is_new_episode
                        obs_time = time.time()
                        batch, replay_img = self.dataset(obs)
                        is_new_episode = False
                        batch['unnorm_key'] = unnorm_key
                        if len(replay_images) == 0:
                            replay_images.append(replay_img)
                        ctx = SimpleNamespace(
                            instruction=task_description,
                            t_obs=obs_time,
                            inference_start=time.time(),
                            inference_elapsed=0.0,
                            raw_actions=None,
                            action_sim_step=sim_step,
                            executed_actions=None,
                            executed_dt=None,
                            executed_sim_step=None,
                            t_first_publish=None)
                        self._apply_rtc_inputs(batch, prev_ctx, sim_step)
                        with torch.autocast(
                                'cuda',
                                dtype=self.mixed_precision_dtype,
                                enabled=self.enable_mixed_precision_training):
                            with torch.no_grad():
                                with self._metrics_inference_ctx(ctx):
                                    actions = self.vla.predict_action(**batch)
                                    ctx.inference_elapsed = (
                                        time.time() - ctx.inference_start)
                                    ctx.raw_actions = actions
                        raw_actions, actions = self._actions_to_numpy(actions)
                        ctx.raw_actions = raw_actions
                        n_execute = self._num_actions_to_execute(actions)
                        actions = self._denormalize_action_chunk(actions)
                        actions = self._postprocess_action_chunk(
                            actions, ctx, prev_ctx, sim_step)
                        n_execute = min(n_execute, actions.shape[0])
                        published_actions = 0
                        for action in actions[:n_execute]:
                            if ctx.t_first_publish is None:
                                ctx.t_first_publish = time.time()
                            obs, reward, done, info = env.step(
                                action.tolist())
                            published_actions += 1
                            self._write_sim_jointstate(
                                obs, action, sim_step)
                            sim_step += 1
                            obs['task_description'] = task_description
                            batch, replay_img = self.dataset(obs)
                            replay_images.append(replay_img)
                            if done:
                                total_successes += 1
                                break
                            t += 1
                        self._emit_action_publish(ctx, published_actions)
                        prev_ctx = ctx
                        if done:
                            break
                    self._mark_episode_success(bool(done))
                total_episodes += 1
                step_tensor = torch.ones(1, device=torch.cuda.current_device())
                # Save a replay video of the episode
                save_rollout_video(
                    replay_images,
                    local_id,
                    success=done,
                    task_description=task_description,
                    work_dir=Path(self.ckpt_path).resolve().parent.parent,
                    log_file=log_file)
                env.close()

                # except Exception as e:
                #     print(f'Error during action prediction: {e}')
                #     log_file.write(f'Caught exception: {e}\n')
                #     action = get_libero_dummy_action()
            dist.barrier()
            dist.all_reduce(step_tensor, op=dist.ReduceOp.SUM)
            if rank == 0 and pbar is not None:
                pbar.update(int(step_tensor.item()))

            global_episodes = total_episodes.clone()
            global_successes = total_successes.clone()
            dist.all_reduce(global_episodes, op=dist.ReduceOp.SUM)
            dist.all_reduce(global_successes, op=dist.ReduceOp.SUM)
            done = done.item() if isinstance(done, torch.Tensor) else done
            if rank == 0:
                # Log current results
                overwatch.info(
                    f'# episodes completed so far: {int(global_episodes[0])}')
                success_rate = (global_successes[0] / global_episodes[0] * 100)
                success_text = (f'# successes: {int(global_successes[0])} '
                                f'({success_rate:.1f}%)')  # noqa: E231
                overwatch.info(success_text)
                log_file.write(f'Success: {done}\n')
                log_file.write(
                    f'# episodes completed so far: {global_episodes[0]}\n')
                success_log = (f'# successes: {global_successes[0]} '
                               f'({success_rate:.1f}%)\n')  # noqa: E231
                log_file.write(success_log)
                log_file.flush()
        dist.barrier()
        exit(0)
