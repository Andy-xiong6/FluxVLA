"""
Prepare ARX LemonPlate HDF5 files for LeRobot conversion.

The source files use numeric names and local camera names:
  0.hdf5, 1.hdf5, ...
  /observations/images/mid
  /observations/images/right

This script writes conversion-ready files:
  episode_000000.hdf5, episode_000001.hdf5, ...
  /observations/images/cam_high
  /observations/images/cam_left_wrist

By default datasets are written as HDF5 external links to avoid duplicating
large image arrays. Use --copy-data to materialize full copies instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


DATASET_LINKS = {
    '/observations/qpos': '/observations/qpos',
    '/action': '/action',
    '/eef_qpos': '/observations/eepose',
    '/observations/images/mid': '/observations/images/cam_high',
    '/observations/images/right': '/observations/images/cam_left_wrist',
}


def numeric_hdf5_files(raw_dir: Path) -> list[Path]:
    files = []
    for path in raw_dir.glob('*.hdf5'):
        try:
            int(path.stem)
        except ValueError:
            continue
        files.append(path)
    return sorted(files, key=lambda item: int(item.stem))


def require_dataset(src: h5py.File, key: str) -> h5py.Dataset:
    if key not in src:
        raise KeyError(f'Missing required dataset: {key}')
    obj = src[key]
    if not isinstance(obj, h5py.Dataset):
        raise TypeError(f'Expected dataset at {key}, got {type(obj)}')
    return obj


def validate_source(src: h5py.File, path: Path) -> int:
    state = require_dataset(src, '/observations/qpos')
    action = require_dataset(src, '/action')
    eepose = require_dataset(src, '/eef_qpos')
    cam_high = require_dataset(src, '/observations/images/mid')
    cam_wrist = require_dataset(src, '/observations/images/right')

    if state.ndim != 2 or state.shape[1] != 7:
        raise ValueError(
            f'{path}: /observations/qpos must have shape [N, 7], got {state.shape}')
    if action.ndim != 2 or action.shape[1] != 7:
        raise ValueError(
            f'{path}: /action must have shape [N, 7], got {action.shape}')
    if eepose.ndim != 2 or eepose.shape[1] != 7:
        raise ValueError(
            f'{path}: /eef_qpos must have shape [N, 7], got {eepose.shape}')
    if cam_high.ndim != 4 or cam_high.shape[-1] != 3:
        raise ValueError(
            f'{path}: /observations/images/mid must have shape [N, H, W, 3], got {cam_high.shape}')
    if cam_wrist.ndim != 4 or cam_wrist.shape[-1] != 3:
        raise ValueError(
            f'{path}: /observations/images/right must have shape [N, H, W, 3], got {cam_wrist.shape}')

    num_frames = state.shape[0]
    frame_counts = {
        '/action': action.shape[0],
        '/eef_qpos': eepose.shape[0],
        '/observations/images/mid': cam_high.shape[0],
        '/observations/images/right': cam_wrist.shape[0],
    }
    for key, count in frame_counts.items():
        if count != num_frames:
            raise ValueError(
                f'{path}: {key} has {count} frames, expected {num_frames}')
    return num_frames


def ensure_group(dst: h5py.File, dataset_path: str) -> None:
    parent = Path(dataset_path).parent.as_posix()
    if parent != '/':
        dst.require_group(parent.lstrip('/'))


def copy_or_link_dataset(
    src: h5py.File,
    dst: h5py.File,
    src_file: Path,
    src_key: str,
    dst_key: str,
    *,
    copy_data: bool,
) -> None:
    ensure_group(dst, dst_key)
    if copy_data:
        src.copy(src_key, dst, name=dst_key)
    else:
        dst[dst_key] = h5py.ExternalLink(str(src_file.resolve()), src_key)


def prepare_file(
    src_path: Path,
    dst_path: Path,
    *,
    copy_data: bool,
    overwrite: bool,
) -> tuple[int, dict[str, float]]:
    if dst_path.exists():
        if not overwrite:
            raise FileExistsError(f'Output exists: {dst_path}')
        dst_path.unlink()

    with h5py.File(src_path, 'r') as src:
        num_frames = validate_source(src, src_path)
        qpos_gripper = src['/observations/qpos'][:, -1]
        action_gripper = src['/action'][:, -1]

        with h5py.File(dst_path, 'w') as dst:
            for key, value in src.attrs.items():
                dst.attrs[key] = value
            dst.attrs['prepared_from'] = str(src_path.resolve())

            for src_key, dst_key in DATASET_LINKS.items():
                copy_or_link_dataset(
                    src,
                    dst,
                    src_path,
                    src_key,
                    dst_key,
                    copy_data=copy_data,
                )

    stats = {
        'qpos_gripper_min': float(np.min(qpos_gripper)),
        'qpos_gripper_max': float(np.max(qpos_gripper)),
        'action_gripper_min': float(np.min(action_gripper)),
        'action_gripper_max': float(np.max(action_gripper)),
    }
    return num_frames, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Prepare ARX LemonPlate HDF5 files for LeRobot conversion.')
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path('/data/xionghongwei/merged_data'),
        help='Directory containing numeric HDF5 files.')
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('/data/xionghongwei/FluxVLA/datasets/arx_lemon_plate_prepared_hdf5'),
        help='Directory for prepared episode_*.hdf5 files.')
    parser.add_argument(
        '--copy-data',
        action='store_true',
        help='Physically copy datasets instead of using HDF5 external links.')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing prepared episode files.')
    args = parser.parse_args()

    files = numeric_hdf5_files(args.input_dir)
    if not files:
        raise FileNotFoundError(f'No numeric *.hdf5 files found in {args.input_dir}')

    args.output_dir.mkdir(parents=True, exist_ok=True)

    aggregate = {
        'qpos_gripper_min': [],
        'qpos_gripper_max': [],
        'action_gripper_min': [],
        'action_gripper_max': [],
    }
    total_frames = 0

    print(f'Input directory: {args.input_dir}')
    print(f'Output directory: {args.output_dir}')
    print(f'Files found: {len(files)}')
    print(f'Mode: {"copy" if args.copy_data else "external-link"}')

    for index, src_path in enumerate(files):
        dst_path = args.output_dir / f'episode_{index:06d}.hdf5'
        num_frames, stats = prepare_file(
            src_path,
            dst_path,
            copy_data=args.copy_data,
            overwrite=args.overwrite,
        )
        total_frames += num_frames
        for key, value in stats.items():
            aggregate[key].append(value)
        if index < 3 or index == len(files) - 1:
            print(
                f'Prepared {src_path.name} -> {dst_path.name} ({num_frames} frames)')

    print('Done.')
    print(f'Total frames: {total_frames}')
    print(
        'qpos gripper range: '
        f'{min(aggregate["qpos_gripper_min"]):.8f} '
        f'to {max(aggregate["qpos_gripper_max"]):.8f}')
    print(
        'action gripper range: '
        f'{min(aggregate["action_gripper_min"]):.8f} '
        f'to {max(aggregate["action_gripper_max"]):.8f}')


if __name__ == '__main__':
    main()
