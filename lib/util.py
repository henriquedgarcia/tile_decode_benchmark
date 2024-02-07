import json
import os
import pickle
from pathlib import Path
from subprocess import run, STDOUT, PIPE
from typing import Union

import numpy as np
import skvideo.io
from matplotlib import pyplot as plt


def save_json(data: Union[dict, list], filename: Union[str, Path], separators=(',', ':'), indent=None):
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


def decode_file(filename, threads=None):
    cmd = (f'bin/ffmpeg -hide_banner -benchmark '
           f'-codec hevc '
           f'{"" if not threads else f"-threads {threads} "}'
           f'-i {filename.as_posix()} '
           f'-f null -')
    if os.name == 'nt':
        cmd = f'bash -c "{cmd}"'

    process = run(cmd, shell=True, stderr=STDOUT, stdout=PIPE, encoding="utf-8")
    return process.stdout


def run_command(command: str):
    print(command)
    process = run(command, shell=True, stderr=STDOUT, stdout=PIPE, encoding="utf-8")
    return process.stdout


def idx2xy(idx: int, shape: tuple):
    tile_x = idx % shape[1]
    tile_y = idx // shape[1]
    return tile_x, tile_y


def splitx(string: str) -> tuple[int, ...]:
    """
    Receive a string like "5x6x7" (no spaces) and return a tuple of ints, in
    this case, (5, 6, 7).
    :param string: A string of numbers separated with "x".
    :return: Return a list of int
    """
    return tuple(map(int, string.split('x')))


def get_times(content):
    times = []
    for line in content:
        if 'utime' in line:
            t = float(line.strip().split(' ')[1].split('=')[1][:-1])
            if t > 0:
                times.append(t)
    return times


def mse2psnr(_mse: float) -> float:
    return 10 * np.log10((255. ** 2 / _mse))


def show(img):
    plt.imshow(img)
    plt.show()


def iter_frame(video_path, gray=True, dtype='float64'):
    vreader = skvideo.io.vreader(f'{video_path}', as_grey=gray)
    frames = []
    for frame in vreader:
        if gray:
            _, height, width, _ = frame.shape
            frame = frame.reshape((height, width)).astype(dtype)
        frames.append(frame)
    return frames
