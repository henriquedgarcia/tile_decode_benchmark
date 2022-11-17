import json
import pickle
from logging import warning, error
from pathlib import Path
from subprocess import run, STDOUT
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class AutoDict(dict):
    def __missing__(self, key):
        self[key] = type(self)()
        return self[key]


def save_json(data: Union[dict, 'AutoDict', list], filename: Union[str, Path],
              separators=(',', ':'), indent=None):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, separators=separators, indent=indent)


def load_json(filename, object_hook=dict):
    with open(filename, 'r', encoding='utf-8') as f:
        results = json.load(f, object_hook=object_hook)
    return results


def save_pickle(data: object, filename: Union[str, Path]):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, protocol=-1)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    return results


def splitx(string: str) -> tuple[int, ...]:
    """
    Receive a string like "5x6x7" (no spaces) and return a tuple of ints, in
    this case, (5, 6, 7).
    :param string: A string of numbers separated with "x".
    :return: Return a list of int
    """
    return tuple(map(int, string.split('x')))


def run_command(command: str, log_file: Path = None, mode='w'):
    print(command)
    process = run(command, shell=True, stderr=STDOUT, encoding='utf-8')

    if process.returncode != 0:
        error(f'SUBPROCESS ERROR: video {log_file}\n'
                f'    {process.returncode = } - {process.stdout = }. Continuing.')
    if not process.stdout or log_file is None:
        warning(f'LOG FILE ERROR: video {log_file}\n'
                f'    stdout in (None, '') or/and log_file==None. Continuing.')
        return

    log = log_file.read_text() if log_file.exists() and mode == 'a' else ''
    log_file.write_text(log + '\n' + command + '\n' + str(process.stdout))


def cart2hcs(x_y_z: np.ndarray) -> np.ndarray:
    """
    Convert from cartesian system to horizontal coordinate system in radians
    :param x_y_z: 1D ndarray [x, y, z], or 2D array with shape=(N, 3)
    :return: (azimuth, elevation) - in rad
    """
    r = np.sqrt(np.sum(x_y_z ** 2))
    azimuth = np.arctan2(x_y_z[..., 0], x_y_z[..., 2])
    elevation = np.arcsin(-x_y_z[..., 1] / r)
    return np.array([azimuth, elevation]).T


def lin_interpol(t: float, t_f: float, t_i: float, v_f: np.ndarray,
                 v_i: np.ndarray) -> np.ndarray:
    m: np.ndarray = (v_f - v_i) / (t_f - t_i)
    v: np.ndarray = m * (t - t_i) + v_i
    return v


def idx2xy(idx: int, shape: tuple):
    tile_x = idx % shape[1]
    tile_y = idx // shape[1]
    return tile_x, tile_y


def position2trajectory(positions_list, fps=30):
    # in rads
    yaw_state = 0
    pitch_state = 0
    old_yaw = 0
    old_pitch = 0
    yaw_velocity = pd.Series(dtype=float)
    pitch_velocity = pd.Series(dtype=float)
    yaw_trajectory = []
    pitch_trajectory = []
    pi = np.pi

    for frame, position in enumerate(positions_list):
        """
        position: pd.Serie
        position.index == []
        """
        yaw = position[0]
        pitch = position[1]
        roll =  position[2]

        if not frame == 1:
            yaw_diff = yaw - old_yaw
            if yaw_diff > 1.45:
                yaw_state -= 1
            elif yaw_diff < -200:
                yaw_state += 1

            pitch_diff = pitch - old_pitch
            if pitch_diff > 120:
                pitch_state -= 1
            elif pitch_diff < -120:
                pitch_state += 1
            # print(f'Frame {n}, old={old:.3f}°, new={position:.3f}°, diff={diff :.3f}°')  # Want a log?

        new_yaw = yaw + pi * yaw_state
        yaw_trajectory.append(new_yaw)

        new_pitch = pitch + pi/2 * pitch_state
        pitch_trajectory.append(new_pitch)

        if frame == 1:
            yaw_velocity.loc[frame] = 0
            pitch_velocity.loc[frame] = 0
        else:
            yaw_velocity.loc[frame] = (yaw_trajectory[-1] - yaw_trajectory[-2]) * fps
            pitch_velocity.loc[frame] = (pitch_trajectory[-1] - pitch_trajectory[-2]) * fps

        old_yaw = yaw
        old_pitch = pitch

    # Filter
    padded_yaw_velocity = [yaw_velocity.iloc[0]] + list(yaw_velocity) + [yaw_velocity.iloc[-1]]
    yaw_velocity_filtered = [sum(padded_yaw_velocity[idx - 1:idx + 2]) / 3
                             for idx in range(1, len(padded_yaw_velocity) - 1)]

    padded_pitch_velocity = [pitch_velocity.iloc[0]] + list(pitch_velocity) + [pitch_velocity.iloc[-1]]
    pitch_velocity_filtered = [sum(padded_pitch_velocity[idx - 1:idx + 2]) / 3
                               for idx in range(1, len(padded_pitch_velocity) - 1)]

    # Scalar velocity
    yaw_speed = np.abs(yaw_velocity_filtered)
    pitch_speed = np.abs(pitch_velocity_filtered)


