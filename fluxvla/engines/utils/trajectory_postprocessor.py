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

from typing import Dict, Optional, Tuple

import numpy as np


def _resample_remaining(traj: np.ndarray, offset: float) -> np.ndarray:
    traj = np.asarray(traj)
    if traj.ndim != 2:
        raise ValueError('traj must be a 2D array')
    n_steps = traj.shape[0]
    out_len = n_steps - int(offset)
    if out_len <= 0:
        return traj[:0]
    idx = np.clip(offset + np.arange(out_len), 0.0, n_steps - 1.0)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, n_steps - 1)
    alpha = (idx - lo)[:, np.newaxis]
    return traj[lo] + alpha * (traj[hi] - traj[lo])


def _smoothstep(n_steps: int) -> np.ndarray:
    if n_steps <= 1:
        return np.ones((n_steps,), dtype=np.float64)
    x = np.linspace(0.0, 1.0, n_steps, dtype=np.float64)
    return x * x * (3.0 - 2.0 * x)


def _as_limit(values, dofs: int, default) -> list:
    if values is None:
        if not np.isscalar(default):
            default = list(default)
            if len(default) != dofs:
                raise ValueError(
                    f'Expected {dofs} default values, got {len(default)}')
            return [float(v) for v in default]
        return [float(default)] * dofs
    if np.isscalar(values):
        return [float(values)] * dofs
    values = list(values)
    if len(values) != dofs:
        raise ValueError(f'Expected {dofs} limit values, got {len(values)}')
    return [float(v) for v in values]


class TrajectoryStitcher:
    """Blend the next action chunk into the currently executing trajectory."""

    def __init__(self,
                 enabled: bool = False,
                 horizon: int = 5,
                 gripper_mode: str = 'hold_then_switch',
                 **kwargs):
        del kwargs
        self.enabled = bool(enabled)
        self.horizon = int(horizon)
        self.gripper_mode = gripper_mode

    def stitch(self,
               actions: np.ndarray,
               *,
               arm_action_dim: int,
               previous_actions: Optional[np.ndarray],
               elapsed_since_previous_start: Optional[float],
               previous_dt: Optional[float]) -> np.ndarray:
        actions = np.asarray(actions)
        if (not self.enabled or self.horizon <= 0 or previous_actions is None
                or elapsed_since_previous_start is None or previous_dt is None
                or previous_dt <= 0.0 or actions.ndim != 2
                or len(actions) == 0):
            return actions

        previous_actions = np.asarray(previous_actions)
        if (previous_actions.ndim != 2
                or previous_actions.shape[1] != actions.shape[1]):
            return actions

        offset = float(elapsed_since_previous_start) / float(previous_dt)
        remaining = _resample_remaining(previous_actions, offset)
        blend_len = min(self.horizon, len(actions), len(remaining))
        if blend_len <= 0:
            return actions

        stitched = actions.copy()
        arm_dim = min(int(arm_action_dim), actions.shape[1])
        weights = _smoothstep(blend_len)[:, np.newaxis]
        stitched[:blend_len, :arm_dim] = (
            (1.0 - weights) * remaining[:blend_len, :arm_dim] +
            weights * actions[:blend_len, :arm_dim])

        if actions.shape[1] > arm_dim:
            if self.gripper_mode == 'hold_then_switch':
                stitched[:blend_len, arm_dim:] = remaining[:blend_len,
                                                           arm_dim:]
            elif self.gripper_mode == 'previous_first':
                stitched[0, arm_dim:] = remaining[0, arm_dim:]
            elif self.gripper_mode != 'new':
                raise ValueError(
                    f'Unsupported gripper stitching mode: {self.gripper_mode}')

        return stitched


class RuckigActionSmoother:
    """Apply jerk-limited Ruckig smoothing to arm joints only."""

    DEFAULT_MAX_VELOCITY = (0.4, 0.4, 0.4, 0.6, 0.6, 0.8)
    DEFAULT_MAX_ACCELERATION = (0.8, 0.8, 0.8, 1.2, 1.2, 1.6)
    DEFAULT_MAX_JERK = (2.0, 2.0, 2.0, 3.0, 3.0, 4.0)

    def __init__(self,
                 enabled: bool = False,
                 max_velocity=None,
                 max_acceleration=None,
                 max_jerk=None,
                 fallback_to_linear: bool = True,
                 max_segment_steps: int = 1000,
                 **kwargs):
        del kwargs
        self.enabled = bool(enabled)
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.max_jerk = max_jerk
        self.fallback_to_linear = bool(fallback_to_linear)
        self.max_segment_steps = int(max_segment_steps)
        self._last_error = None

    @property
    def last_error(self):
        return self._last_error

    def _limits(self, dofs: int) -> Tuple[list, list, list]:
        def default_list(values):
            if dofs <= len(values):
                return list(values[:dofs])
            return list(values) + [values[-1]] * (dofs - len(values))

        return (
            _as_limit(self.max_velocity, dofs,
                      default_list(self.DEFAULT_MAX_VELOCITY)),
            _as_limit(self.max_acceleration, dofs,
                      default_list(self.DEFAULT_MAX_ACCELERATION)),
            _as_limit(self.max_jerk, dofs,
                      default_list(self.DEFAULT_MAX_JERK)),
        )

    def smooth(self, actions: np.ndarray, *, arm_action_dim: int,
               dt: float) -> Tuple[np.ndarray, Dict]:
        actions = np.asarray(actions)
        meta = {'enabled': self.enabled, 'applied': False, 'error': None}
        if (not self.enabled or actions.ndim != 2 or len(actions) <= 1
                or dt <= 0.0):
            return actions, meta

        try:
            from ruckig import InputParameter, OutputParameter, Result, Ruckig
        except Exception as exc:
            self._last_error = str(exc)
            meta['error'] = str(exc)
            return actions, meta

        try:
            arm_dim = min(int(arm_action_dim), actions.shape[1])
            if arm_dim <= 0:
                return actions, meta
            max_vel, max_acc, max_jerk = self._limits(arm_dim)

            otg = Ruckig(arm_dim, float(dt))
            inp = InputParameter(arm_dim)
            out = OutputParameter(arm_dim)
            inp.max_velocity = max_vel
            inp.max_acceleration = max_acc
            inp.max_jerk = max_jerk
            inp.current_position = actions[0, :arm_dim].astype(
                np.float64).tolist()
            inp.current_velocity = [0.0] * arm_dim
            inp.current_acceleration = [0.0] * arm_dim

            smoothed = [actions[0].copy()]
            for seg_idx in range(1, len(actions)):
                inp.target_position = actions[seg_idx, :arm_dim].astype(
                    np.float64).tolist()
                inp.target_velocity = [0.0] * arm_dim
                inp.target_acceleration = [0.0] * arm_dim

                for _ in range(self.max_segment_steps):
                    result = otg.update(inp, out)
                    if result not in (Result.Working, Result.Finished):
                        raise RuntimeError(f'Ruckig update failed: {result}')
                    row = actions[seg_idx].copy()
                    row[:arm_dim] = np.asarray(out.new_position,
                                               dtype=actions.dtype)
                    if row.shape[0] > arm_dim:
                        if result == Result.Finished:
                            row[arm_dim:] = actions[seg_idx, arm_dim:]
                        else:
                            row[arm_dim:] = actions[seg_idx - 1, arm_dim:]
                    smoothed.append(row)
                    out.pass_to_input(inp)
                    if result == Result.Finished:
                        break
                else:
                    raise RuntimeError(
                        f'Ruckig segment {seg_idx} exceeded '
                        f'{self.max_segment_steps} update steps')

            smoothed[-1] = actions[-1]
            meta['applied'] = True
            return np.asarray(smoothed, dtype=actions.dtype), meta
        except Exception as exc:
            self._last_error = str(exc)
            meta['error'] = str(exc)
            return actions, meta
