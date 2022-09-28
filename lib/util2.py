import json
from logging import warning
from pathlib import Path
from subprocess import run, STDOUT
from typing import Union
import pickle

import numpy as np


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

    if process.returncode != 0 or process.stdout == '':
        warning(f'SUBPROCESS ERROR: video {log_file}\n'
                f'    {process.returncode = } - {process.stdout = }. Continuing.')

    log = log_file.read_text() if log_file.exists() and mode == 'a' else ''
    log_file.write_text(log + '\n' + command + '\n' + process.stdout)


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
