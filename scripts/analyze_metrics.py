

"""Offline analyzer for FluxVLA inference metrics.

Reads manifest + per-episode JSONL files written by ``MetricsManager`` /
``EpisodeWriter`` and computes the metrics required by the project plan:
real-time (latency/percentiles/control freq/R), control quality (success
rate/completion time), smoothness (acceleration/jerk), and jitter (FFT
high-frequency energy).
"""

import argparse
import csv
import json
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

EVENTS_FILENAME = 'events.jsonl'
JOINTSTATE_FILENAME = 'jointstate.jsonl'
META_FILENAME = 'meta.json'
MANIFEST_FILENAME = 'manifest.jsonl'
SUCCESS_CSV_FILENAME = 'success.csv'


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze FluxVLA inference metrics.')
    parser.add_argument('--metrics-root', type=str,
                        default='work_dirs/metrics')
    parser.add_argument('--episodes', type=str, nargs='*', default=None,
                        help='Optional list of episode_id to include.')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Default <metrics_root>/_analysis')
    parser.add_argument('--success-csv', type=str, default=None,
                        help='Default <metrics_root>/success.csv')
    parser.add_argument('--filter', choices=['none', 'lowpass', 'savgol'],
                        default='savgol')
    parser.add_argument('--filter-cutoff-hz', type=float, default=5.0)
    parser.add_argument('--savgol-window', type=int, default=21)
    parser.add_argument('--savgol-polyorder', type=int, default=3)
    parser.add_argument('--fft-cutoff-ratio', type=float, default=0.2)
    parser.add_argument('--control-freq-hz', type=float, default=None,
                        help='Override control freq; default reads manifest.')
    parser.add_argument('--update-manifest', action='store_true')
    parser.add_argument('--only-success', action='store_true')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_manifest(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, 'r') as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def load_success(path: str) -> Dict[str, Optional[bool]]:
    if not os.path.exists(path):
        return {}
    out: Dict[str, Optional[bool]] = {}
    with open(path, 'r') as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            ep = (row.get('episode_id') or '').strip()
            if not ep:
                continue
            raw = (row.get('success') or '').strip()
            if raw in ('1', 'true', 'True', 'TRUE'):
                out[ep] = True
            elif raw in ('0', 'false', 'False', 'FALSE'):
                out[ep] = False
            else:
                out[ep] = None
    return out


def list_episode_dirs(metrics_root: str,
                      ids_filter: Optional[List[str]]) -> List[str]:
    if not os.path.isdir(metrics_root):
        return []
    entries = []
    for name in sorted(os.listdir(metrics_root)):
        full = os.path.join(metrics_root, name)
        if not os.path.isdir(full):
            continue
        if name.startswith('_'):
            continue
        if not os.path.exists(os.path.join(full, EVENTS_FILENAME)):
            continue
        if ids_filter is not None and name not in ids_filter:
            continue
        entries.append(full)
    return entries


def load_episode_events(episode_dir: str) -> List[Dict]:
    path = os.path.join(episode_dir, EVENTS_FILENAME)
    out = []
    if not os.path.exists(path):
        return out
    with open(path, 'r') as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def load_episode_jointstate(episode_dir: str) -> pd.DataFrame:
    path = os.path.join(episode_dir, JOINTSTATE_FILENAME)
    if not os.path.exists(path):
        return pd.DataFrame()
    rows = []
    with open(path, 'r') as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df


def split_events(events: List[Dict]) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {
        'episode_start': [],
        'inference': [],
        'action_publish': [],
        'episode_end': [],
    }
    for ev in events:
        kind = ev.get('event')
        if kind in out:
            out[kind].append(ev)
    return out


# ---------------------------------------------------------------------------
# Real-time metrics
# ---------------------------------------------------------------------------
def _percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), q))


def compute_realtime_metrics(events_split: Dict[str, List[Dict]],
                             control_freq_hz: float) -> Dict:
    inferences = events_split.get('inference', [])
    latency_vals = [ev['latency_ms'] for ev in inferences
                    if ev.get('latency_ms') is not None]
    elapsed_vals = [ev['inference_elapsed_ms'] for ev in inferences
                    if ev.get('inference_elapsed_ms') is not None]
    n = len(inferences)
    mean_latency = float(np.mean(latency_vals)) if latency_vals else None
    std_latency = float(np.std(latency_vals)) if latency_vals else None
    p50 = _percentile(latency_vals, 50)
    p95 = _percentile(latency_vals, 95)
    p99 = _percentile(latency_vals, 99)
    mean_elapsed = float(np.mean(elapsed_vals)) if elapsed_vals else None
    R = None
    if (mean_latency is not None and control_freq_hz
            and control_freq_hz > 0.0):
        T_control_ms = 1000.0 / control_freq_hz
        R = mean_latency / T_control_ms
    return {
        'n_inferences': n,
        'mean_latency_ms': mean_latency,
        'std_latency_ms': std_latency,
        'p50_latency_ms': p50,
        'p95_latency_ms': p95,
        'p99_latency_ms': p99,
        'mean_inference_elapsed_ms': mean_elapsed,
        'control_freq_hz': control_freq_hz,
        'realtime_ratio_R': R,
    }


def compute_first_action_response(events_split: Dict[str,
                                                     List[Dict]]
                                  ) -> Optional[float]:
    starts = events_split.get('episode_start', [])
    pubs = events_split.get('action_publish', [])
    if not starts or not pubs:
        return None
    t_start = starts[0].get('t_wall')
    first = None
    for ev in pubs:
        if ev.get('is_first_action_in_episode'):
            first = ev.get('t_first_publish')
            break
    if first is None:
        first = pubs[0].get('t_first_publish')
    if t_start is None or first is None:
        return None
    return float(first) - float(t_start)


def compute_completion_time(events_split: Dict[str,
                                               List[Dict]]) -> Optional[float]:
    ends = events_split.get('episode_end', [])
    if ends:
        d = ends[-1].get('duration_s')
        if d is not None:
            return float(d)
    starts = events_split.get('episode_start', [])
    if starts and ends:
        try:
            return float(ends[-1]['t_wall'] - starts[0]['t_wall'])
        except Exception:
            return None
    return None


def compute_publish_frequency_actual(events_split: Dict[str, List[Dict]]
                                     ) -> Optional[float]:
    pubs = events_split.get('action_publish', [])
    if len(pubs) < 1:
        return None
    total_actions = sum(int(ev.get('n_actions') or 0) for ev in pubs)
    if total_actions <= 0:
        return None
    starts = events_split.get('episode_start', [])
    ends = events_split.get('episode_end', [])
    if not starts or not ends:
        return None
    try:
        duration = float(ends[-1]['t_wall'] - starts[0]['t_wall'])
        if duration <= 0:
            return None
        return total_actions / duration
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------
def filter_signal(arr: np.ndarray, fs: float, mode: str,
                  cutoff_hz: float = 5.0, savgol_window: int = 21,
                  savgol_polyorder: int = 3) -> np.ndarray:
    if mode == 'none' or arr.shape[0] < 5:
        return arr
    from scipy.signal import butter, filtfilt, savgol_filter
    if mode == 'savgol':
        win = min(savgol_window, arr.shape[0] - (1 - arr.shape[0] % 2))
        if win < savgol_polyorder + 2:
            return arr
        if win % 2 == 0:
            win -= 1
        if arr.ndim == 1:
            return savgol_filter(arr, win, savgol_polyorder)
        out = np.empty_like(arr)
        for j in range(arr.shape[1]):
            out[:, j] = savgol_filter(arr[:, j], win, savgol_polyorder)
        return out
    if mode == 'lowpass':
        nyq = fs / 2.0
        if nyq <= 0 or cutoff_hz >= nyq:
            return arr
        wn = cutoff_hz / nyq
        b, a = butter(4, wn, btype='low')
        if arr.ndim == 1:
            return filtfilt(b, a, arr)
        out = np.empty_like(arr)
        for j in range(arr.shape[1]):
            out[:, j] = filtfilt(b, a, arr[:, j])
        return out
    return arr


def compute_velocity_from_position(pos: np.ndarray,
                                   t: np.ndarray) -> np.ndarray:
    if pos.shape[0] < 2:
        return np.zeros_like(pos)
    dt = np.diff(t)
    dt = np.where(dt <= 0, np.nan, dt)
    vel = np.zeros_like(pos)
    vel[1:] = (pos[1:] - pos[:-1]) / dt[:, None]
    vel[0] = vel[1]
    vel = np.nan_to_num(vel, nan=0.0, posinf=0.0, neginf=0.0)
    return vel


def compute_acceleration(vel: np.ndarray, t: np.ndarray) -> np.ndarray:
    if vel.shape[0] < 2:
        return np.zeros_like(vel)
    dt = np.diff(t)
    dt = np.where(dt <= 0, np.nan, dt)
    acc = np.zeros_like(vel)
    acc[1:] = (vel[1:] - vel[:-1]) / dt[:, None]
    acc[0] = acc[1]
    acc = np.nan_to_num(acc, nan=0.0, posinf=0.0, neginf=0.0)
    return acc


def compute_jerk_central(vel: np.ndarray, dt_mean: float) -> np.ndarray:
    """j_i = (v_{i+2} - 2 v_{i+1} + v_i) / dt^2 (plan section 7.2)."""
    if vel.shape[0] < 3 or dt_mean <= 0:
        return np.zeros((max(vel.shape[0] - 2, 0), vel.shape[1]
                         if vel.ndim > 1 else 1))
    if vel.ndim == 1:
        return (vel[2:] - 2.0 * vel[1:-1] + vel[:-2]) / (dt_mean ** 2)
    return (vel[2:] - 2.0 * vel[1:-1] + vel[:-2]) / (dt_mean ** 2)


# ---------------------------------------------------------------------------
# Smoothness + FFT jitter
# ---------------------------------------------------------------------------
def _vec_l2_norm(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return np.abs(arr)
    return np.linalg.norm(arr, axis=1)


def compute_smoothness_metrics(jointstate_df: pd.DataFrame,
                               filter_mode: str, filter_cutoff_hz: float,
                               savgol_window: int,
                               savgol_polyorder: int) -> Dict:
    if jointstate_df.empty or 'joint_vel' not in jointstate_df.columns:
        return {
            'fs_hz_estimate': None,
            'avg_acc_norm': None,
            'avg_jerk_norm': None,
            'max_jerk_norm': None,
            'jerk_norm_series': None,
            'jerk_dt_mean': None,
        }
    t = jointstate_df['t_ros'].fillna(jointstate_df['t_wall']).to_numpy(
        dtype=float)
    order = np.argsort(t)
    t = t[order]
    vel_lists = jointstate_df['joint_vel'].iloc[order].tolist()
    try:
        vel = np.asarray(vel_lists, dtype=float)
    except Exception:
        return {
            'fs_hz_estimate': None,
            'avg_acc_norm': None,
            'avg_jerk_norm': None,
            'max_jerk_norm': None,
            'jerk_norm_series': None,
            'jerk_dt_mean': None,
        }
    if vel.ndim == 1:
        vel = vel.reshape(-1, 1)
    if vel.shape[0] < 5:
        return {
            'fs_hz_estimate': None,
            'avg_acc_norm': None,
            'avg_jerk_norm': None,
            'max_jerk_norm': None,
            'jerk_norm_series': None,
            'jerk_dt_mean': None,
        }
    diffs = np.diff(t)
    dt_mean = float(np.mean(diffs[diffs > 0])) if np.any(diffs > 0) else None
    fs_hz = (1.0 / dt_mean) if dt_mean else None
    if fs_hz is None or fs_hz <= 0:
        return {
            'fs_hz_estimate': None,
            'avg_acc_norm': None,
            'avg_jerk_norm': None,
            'max_jerk_norm': None,
            'jerk_norm_series': None,
            'jerk_dt_mean': None,
        }
    vel_filt = filter_signal(vel, fs_hz, filter_mode,
                             cutoff_hz=filter_cutoff_hz,
                             savgol_window=savgol_window,
                             savgol_polyorder=savgol_polyorder)
    acc = compute_acceleration(vel_filt, t)
    jerk = compute_jerk_central(vel_filt, dt_mean)
    acc_norm = _vec_l2_norm(acc)
    jerk_norm = _vec_l2_norm(jerk)
    return {
        'fs_hz_estimate': fs_hz,
        'avg_acc_norm': float(np.mean(acc_norm)) if acc_norm.size else None,
        'avg_jerk_norm': (float(np.mean(jerk_norm))
                          if jerk_norm.size else None),
        'max_jerk_norm': float(np.max(jerk_norm)) if jerk_norm.size else None,
        'jerk_norm_series': jerk_norm,
        'jerk_dt_mean': dt_mean,
    }


def compute_jitter_fft(jerk_norm_series: Optional[np.ndarray],
                       fs_hz: Optional[float], fc_ratio: float,
                       control_freq_hz: float) -> Dict:
    if (jerk_norm_series is None or jerk_norm_series.size < 8 or fs_hz is None
            or fs_hz <= 0 or control_freq_hz is None
            or control_freq_hz <= 0):
        return {
            'fft_high_freq_energy': None,
            'fft_total_energy': None,
            'fft_normalized_high_freq_energy': None,
            'fft_f_c_hz': None,
        }
    arr = jerk_norm_series - np.mean(jerk_norm_series)
    n = arr.shape[0]
    window = np.hanning(n)
    arr = arr * window
    spectrum = np.fft.rfft(arr)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs_hz)
    power = (np.abs(spectrum) ** 2)
    total = float(np.sum(power))
    fc = fc_ratio * control_freq_hz
    high_mask = freqs > fc
    high = float(np.sum(power[high_mask]))
    j_norm = (high / total) if total > 0 else None
    return {
        'fft_high_freq_energy': high,
        'fft_total_energy': total,
        'fft_normalized_high_freq_energy': j_norm,
        'fft_f_c_hz': fc,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate_episode(episode_dir: str, success: Optional[bool],
                      control_freq_override: Optional[float],
                      filter_mode: str, filter_cutoff_hz: float,
                      savgol_window: int, savgol_polyorder: int,
                      fft_cutoff_ratio: float) -> Dict:
    episode_id = os.path.basename(episode_dir.rstrip('/'))
    events = load_episode_events(episode_dir)
    events_split = split_events(events)
    starts = events_split['episode_start']
    task_id = (starts[0].get('task_id') if starts else None)
    instruction = (starts[0].get('instruction') if starts else None)
    publish_rate_meta = (starts[0].get('publish_rate') if starts else None)
    control_freq_hz = (control_freq_override
                       if control_freq_override is not None else
                       (float(publish_rate_meta)
                        if publish_rate_meta else 30.0))

    rt = compute_realtime_metrics(events_split, control_freq_hz)
    first_resp = compute_first_action_response(events_split)
    completion_s = compute_completion_time(events_split)
    actual_freq = compute_publish_frequency_actual(events_split)

    js_df = load_episode_jointstate(episode_dir)
    smooth = compute_smoothness_metrics(js_df, filter_mode, filter_cutoff_hz,
                                        savgol_window, savgol_polyorder)
    jitter = compute_jitter_fft(smooth.get('jerk_norm_series'),
                                smooth.get('fs_hz_estimate'),
                                fft_cutoff_ratio, control_freq_hz)

    out = {
        'episode_id': episode_id,
        'task_id': task_id,
        'instruction': instruction,
        'success': success,
        'first_action_response_s': first_resp,
        'completion_time_s': completion_s,
        'publish_freq_actual_hz': actual_freq,
    }
    out.update(rt)
    for k in ('avg_acc_norm', 'avg_jerk_norm', 'max_jerk_norm',
              'fs_hz_estimate'):
        out[k] = smooth.get(k)
    for k in ('fft_high_freq_energy', 'fft_total_energy',
              'fft_normalized_high_freq_energy', 'fft_f_c_hz'):
        out[k] = jitter.get(k)

    out['_jerk_norm_series'] = smooth.get('jerk_norm_series')
    out['_jerk_dt_mean'] = smooth.get('jerk_dt_mean')
    out['_jointstate_df'] = js_df
    out['_latency_ms_list'] = [
        ev['latency_ms'] for ev in events_split.get('inference', [])
        if ev.get('latency_ms') is not None
    ]
    return out


def aggregate_overall(per_episode: List[Dict]) -> Dict:
    keys = [
        'mean_latency_ms', 'p50_latency_ms', 'p95_latency_ms', 'p99_latency_ms',
        'mean_inference_elapsed_ms', 'realtime_ratio_R',
        'first_action_response_s', 'completion_time_s',
        'publish_freq_actual_hz', 'avg_acc_norm', 'avg_jerk_norm',
        'max_jerk_norm', 'fft_normalized_high_freq_energy',
    ]
    overall: Dict = {}
    for k in keys:
        vals = [ep.get(k) for ep in per_episode if ep.get(k) is not None]
        vals = [float(v) for v in vals]
        if not vals:
            overall[k] = {'mean': None, 'std': None, 'p95': None, 'p99': None,
                          'n': 0}
        else:
            arr = np.asarray(vals, dtype=float)
            overall[k] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'p95': float(np.percentile(arr, 95)),
                'p99': float(np.percentile(arr, 99)),
                'n': int(arr.size),
            }
    n_total = len(per_episode)
    n_labeled = sum(1 for ep in per_episode if ep.get('success') is not None)
    n_success = sum(1 for ep in per_episode if ep.get('success') is True)
    overall['n_episodes_total'] = n_total
    overall['n_episodes_labeled'] = n_labeled
    overall['n_episodes_success'] = n_success
    overall['success_rate'] = (n_success / n_labeled
                               if n_labeled > 0 else None)
    return overall


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
PER_EP_COLUMNS = [
    'episode_id', 'task_id', 'success', 'n_inferences',
    'mean_latency_ms', 'std_latency_ms', 'p50_latency_ms', 'p95_latency_ms',
    'p99_latency_ms', 'mean_inference_elapsed_ms', 'control_freq_hz',
    'realtime_ratio_R', 'first_action_response_s', 'completion_time_s',
    'publish_freq_actual_hz', 'fs_hz_estimate', 'avg_acc_norm',
    'avg_jerk_norm', 'max_jerk_norm', 'fft_normalized_high_freq_energy',
    'fft_high_freq_energy', 'fft_total_energy', 'fft_f_c_hz',
]


def write_per_episode_csv(per_episode: List[Dict], output_path: str):
    rows = []
    for ep in per_episode:
        row = {k: ep.get(k) for k in PER_EP_COLUMNS}
        rows.append(row)
    df = pd.DataFrame(rows, columns=PER_EP_COLUMNS)
    df.to_csv(output_path, index=False)


def _fmt(x, digits=4):
    if x is None:
        return 'N/A'
    if isinstance(x, float):
        if not np.isfinite(x):
            return 'N/A'
        return f'{x:.{digits}g}'
    return str(x)


def write_summary_markdown(overall: Dict, per_episode: List[Dict],
                           output_path: str):
    lines = []
    lines.append('# FluxVLA Inference Metrics Summary\n')
    lines.append(f'- Total episodes: {overall.get("n_episodes_total")}')
    lines.append(f'- Labeled episodes: {overall.get("n_episodes_labeled")}')
    lines.append(f'- Successful episodes: {overall.get("n_episodes_success")}')
    sr = overall.get('success_rate')
    lines.append(f'- Success rate (labeled): {_fmt(sr)}')
    lines.append('')
    lines.append('## Aggregated metrics across episodes')
    lines.append('')
    lines.append('| metric | mean | std | p95 | p99 | n |')
    lines.append('|---|---|---|---|---|---|')
    for k in [
            'mean_latency_ms', 'p50_latency_ms', 'p95_latency_ms',
            'p99_latency_ms', 'mean_inference_elapsed_ms',
            'realtime_ratio_R', 'first_action_response_s',
            'completion_time_s', 'publish_freq_actual_hz', 'avg_acc_norm',
            'avg_jerk_norm', 'max_jerk_norm',
            'fft_normalized_high_freq_energy']:
        s = overall.get(k, {})
        lines.append(f'| {k} | {_fmt(s.get("mean"))} | {_fmt(s.get("std"))} '
                     f'| {_fmt(s.get("p95"))} | {_fmt(s.get("p99"))} '
                     f'| {s.get("n")} |')
    lines.append('')
    lines.append('## Per-episode breakdown')
    lines.append('')
    head_cols = ['episode_id', 'task_id', 'success', 'mean_latency_ms',
                 'p95_latency_ms', 'p99_latency_ms', 'realtime_ratio_R',
                 'first_action_response_s', 'completion_time_s',
                 'avg_jerk_norm', 'fft_normalized_high_freq_energy']
    lines.append('| ' + ' | '.join(head_cols) + ' |')
    lines.append('|' + '|'.join(['---'] * len(head_cols)) + '|')
    for ep in per_episode:
        cells = [_fmt(ep.get(c)) for c in head_cols]
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines.append('')
    with open(output_path, 'w') as fp:
        fp.write('\n'.join(lines))


def write_summary_json(overall: Dict, output_path: str):
    with open(output_path, 'w') as fp:
        json.dump(overall, fp, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_latency_hist(per_episode: List[Dict], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for ep in per_episode:
        episode_id = ep.get('episode_id', 'unknown')
        latencies = ep.get('_latency_ms_list') or []
        if not latencies:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(latencies, bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Latency (ms)')
        ax.set_ylabel('Count')
        ax.set_title(f'{episode_id}: latency')
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir,
                                 f'{episode_id}__latency_hist.png'),
                    dpi=120)
        plt.close(fig)


def plot_jerk_timeseries(per_episode: List[Dict], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for ep in per_episode:
        series = ep.get('_jerk_norm_series')
        if series is None or series.size == 0:
            continue
        episode_id = ep.get('episode_id', 'unknown')
        dt = ep.get('_jerk_dt_mean')
        if dt and dt > 0:
            t = np.arange(series.size) * dt
            x_label = 'Time (s)'
        else:
            t = np.arange(series.size)
            x_label = 'Sample index'
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(t, series, linewidth=0.8)
        ax.set_xlabel(x_label)
        ax.set_ylabel('||jerk||')
        ax.set_title(f'{episode_id}: jerk L2 norm')
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir,
                                 f'{episode_id}__jerk_ts.png'), dpi=120)
        plt.close(fig)


def plot_fft_spectrum(per_episode: List[Dict], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for ep in per_episode:
        series = ep.get('_jerk_norm_series')
        fs = ep.get('fs_hz_estimate')
        fc = ep.get('fft_f_c_hz')
        if series is None or series.size < 8 or fs is None or fs <= 0:
            continue
        episode_id = ep.get('episode_id', 'unknown')
        arr = series - np.mean(series)
        window = np.hanning(arr.size)
        spec = np.fft.rfft(arr * window)
        freqs = np.fft.rfftfreq(arr.size, d=1.0 / fs)
        power = np.abs(spec) ** 2
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.semilogy(freqs, power + 1e-12)
        if fc is not None:
            ax.axvline(fc, color='red', linestyle='--', linewidth=1,
                       label=f'f_c={fc:.2f}Hz')
            ax.legend()
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power (log)')
        ax.set_title(f'{episode_id}: jerk FFT')
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir,
                                 f'{episode_id}__fft.png'), dpi=120)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Manifest update
# ---------------------------------------------------------------------------
def write_manifest_with_metrics(metrics_root: str, per_episode: List[Dict],
                                output_path: str):
    manifest = load_manifest(os.path.join(metrics_root, MANIFEST_FILENAME))
    by_id = {ep.get('episode_id'): ep for ep in per_episode}
    rows = []
    for entry in manifest:
        ep_id = entry.get('episode_id')
        merged = dict(entry)
        ep = by_id.get(ep_id)
        if ep is not None:
            for k in PER_EP_COLUMNS:
                if k in ('episode_id', 'task_id'):
                    continue
                merged[k] = ep.get(k)
        rows.append(merged)
    with open(output_path, 'w') as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False, allow_nan=False)
                     + '\n')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    metrics_root = os.path.abspath(args.metrics_root)
    output_dir = (os.path.abspath(args.output_dir) if args.output_dir
                  else os.path.join(metrics_root, '_analysis'))
    success_csv = (args.success_csv or os.path.join(metrics_root,
                                                    SUCCESS_CSV_FILENAME))
    os.makedirs(output_dir, exist_ok=True)
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    success_map = load_success(success_csv)
    episode_dirs = list_episode_dirs(metrics_root, args.episodes)
    if not episode_dirs:
        print(f'No episode directories found under {metrics_root}')
        return 0

    per_episode: List[Dict] = []
    for ed in episode_dirs:
        episode_id = os.path.basename(ed.rstrip('/'))
        success = success_map.get(episode_id)
        if args.only_success and success is not True:
            continue
        try:
            ep = aggregate_episode(
                ed, success=success,
                control_freq_override=args.control_freq_hz,
                filter_mode=args.filter,
                filter_cutoff_hz=args.filter_cutoff_hz,
                savgol_window=args.savgol_window,
                savgol_polyorder=args.savgol_polyorder,
                fft_cutoff_ratio=args.fft_cutoff_ratio)
        except Exception as e:
            print(f'[WARN] aggregate_episode({episode_id}) failed: {e}')
            continue
        per_episode.append(ep)

    if not per_episode:
        print('No episodes aggregated.')
        return 0

    overall = aggregate_overall(per_episode)
    write_per_episode_csv(per_episode,
                          os.path.join(output_dir, 'per_episode.csv'))
    write_summary_markdown(overall, per_episode,
                           os.path.join(output_dir, 'summary.md'))
    write_summary_json(overall, os.path.join(output_dir, 'summary.json'))

    plot_latency_hist(per_episode, figures_dir)
    plot_jerk_timeseries(per_episode, figures_dir)
    plot_fft_spectrum(per_episode, figures_dir)

    if args.update_manifest:
        write_manifest_with_metrics(
            metrics_root, per_episode,
            os.path.join(output_dir, 'manifest_with_metrics.jsonl'))

    print(f'Analysis complete. Outputs at: {output_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(parse_args()))
