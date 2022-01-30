import pickle
import json
import os
import subprocess
from logging import debug, error, warning
from pathlib import Path
from subprocess import run
from typing import Any, Dict, Hashable, Iterable, Tuple, Union

import numpy as np
import pandas as pd
import skvideo.io
from scipy import ndimage

from lib.assets import AutoDict, StatsData, PointHCS, Resolution, Point, Pixel

################    QUALITY    ################
def load_sph_file(sph_file: Path, shape: tuple[int, int] = None):
    """
    Load 655362 sample points (elevation, azimuth). Angles in degree.

    :param sph_file:
    :param shape:
    :return:
    """

    iter_file = sph_file.read_text().splitlines()

    sph_points = []
    cart_coord = []
    sph_points_img = []
    sph_points_mask = np.zeros(0)

    if shape is not None:
        sph_points_mask = np.zeros(shape)

    for idx, line in enumerate(iter_file[1:]):
        point = list(map(float, line.strip().split()))
        hcs_point = dict(azimuth=np.deg2rad(point[1]),
                         elevation=np.deg2rad(point[0]))
        sph_points.append(hcs_point)

        ccs_point = dict(x=np.sin(hcs_point['elevation'])
                           * np.cos(hcs_point['azimuth']),
                         y=np.sin(hcs_point['azimuth']),
                         z=-np.cos(hcs_point['elevation'])
                           * np.cos(hcs_point['azimuth']))
        cart_coord.append(ccs_point)

        if shape is not None:
            height, width = shape
            x = np.ceil((hcs_point['azimuth'] + np.pi)
                        * width / (2 * np.pi) - 1)
            y = np.floor((hcs_point['elevation'] - np.pi / 2)
                         * height / np.pi + height)
            if y >= height: y = height - 1
            ics_point = dict(x=int(x), y=int(y))
            sph_points_img.append(ics_point)
            sph_points_mask[int(y), int(x)] = 1

    return sph_points, cart_coord, sph_points_img, sph_points_mask

def mse2psnr(mse: float, pixel_max=255) -> float:
    return 10 * np.log10((pixel_max ** 2 / mse))

################    PROJECTIONS    ################
# ------------------ erp2sph ------------------
def erp2hcs(pixel: Pixel, res: Resolution) -> PointHCS:
    """
    Convert a 2D point of ERP projection coordinates to Horizontal Coordinate
    System in rad. Only ERP Projection
    :param Pixel pixel: A point in ERP projection
    :param Resolution res: The resolution of the projection
    :return: A 3D Point on the sphere
    """
    azimuth = ((pixel[0] + 0.5) / res.shape[1] - 0.5) * 2 * np.pi
    elevation = -((pixel[1] + 0.5) / res.shape[0] - 0.5) * np.pi
    return PointHCS(azimuth, elevation)

def hcs2xyz(hcs_point: PointHCS) -> Point:
    """
    Horizontal Coordinate system to Cartesian coordinates.
    ISO/IEC JTC1/SC29/WG11/N17197l: Algorithm descriptions of projection format conversion and video quality metrics in 360Lib Version 5
    :param PointHCS hcs_point: The coordinates in Horizontal Coordinate System
    :return: A Point3d in cartesian coordinates
    """
    x = np.cos(hcs_point[0]) * np.cos(hcs_point[1])
    y = np.sin(hcs_point[1])
    z = -np.cos(hcs_point[1]) * np.sin(hcs_point[0])

    return Point(x, y, z)

# ------------------ sph2erp ------------------
def xyz2hcs(x, y, z) -> tuple[float, float]:
    """
    Convert from cartesian system to horizontal coordinate system in radians
    :param float x: Coordinate from X axis
    :param float y: Coordinate from Y axis
    :param float z: Coordinate from Z axis
    :return: (azimuth, elevation)
    """
    azimuth = np.arctan2(-z, x)
    elevation = np.arctan2(y, np.sqrt(x * x + y * y + z * z))
    return PointHCS(azimuth, elevation)

def sph2erp(theta, phi, shape: tuple) -> tuple[int, int]:
    return uv2img(*sph2uv(theta, phi), shape)

def sph2uv(theta, phi):
    PI = np.pi
    while True:
        if theta >= PI:
            theta -= 2 * PI
            continue
        elif theta < -PI:
            theta += 2 * PI
            continue
        if phi < -PI / 2:
            phi = -PI - phi
            continue
        elif phi > PI / 2:
            phi = PI - phi
            continue
        break
    u = theta / (2 * PI) + 0.5
    v = -phi / PI + 0.5
    return u, v

def uv2img(u, v, shape: tuple):
    m = round(u * shape[1] - 0.5)
    n = round(v * shape[0] - 0.5)
    return m, n

# ------------------ Util ------------------
def lin_interpol(t: float, t_f: float, t_i: float, v_f: pd.Serie,
                 v_i: pd.Serie) -> pd.Serie:
    m: pd.Serie = (v_f - v_i) / (t_f - t_i)
    v: pd.Serie = m * (t - t_i) + v_i
    return v

def iter_frame(video_path, gray=True, dtype='float32'):
    vreader = skvideo.io.vreader(f'{video_path}', as_grey=gray)
    for frame in vreader:
        if gray:
            _, height, width, _ = frame.shape
            frame = frame.reshape((height, width)).astype(dtype)
        yield frame

def check_video_gop(video_file: Path) -> tuple:
    command = (f'ffprobe -hide_banner -loglevel 0 '
               f'-of default=nk=1:nw=1 '
               f'-show_entries frame=pict_type '
               f'"{video_file}"')
    process = run(command, shell=True, capture_output=True, encoding='utf-8')
    if process.returncode != 0:
        warning(f'FFPROBE ERROR: Return {process.returncode} to video '
                f'{video_file}')
        return 0, []

    output = process.stdout
    gop = []
    max_gop = 0
    len_gop = 0
    for line in output.splitlines():
        line = line.strip()
        if line in ['I', 'B', 'P']:
            if line in 'I':
                len_gop = 1
            else:
                len_gop += 1
            if len_gop > max_gop:
                max_gop = len_gop
            gop.append(line)
    return max_gop, gop

def check_file_size(video_file) -> int:
    debug(f'Checking size of {video_file}')
    if not os.path.isfile(video_file):
        return -1
    filesize = os.path.getsize(video_file)
    if filesize == 0:
        return 0
    debug(f'The size is {filesize}')
    return filesize

def run_command(command: str, log_to_save: Union[str, Path], mode: str = 'w'):
    """
    Run a shell command with subprocess module with realtime output.
    :param command: A command string to run.
    :param log_to_save: A path-like to save the process output.
    :param mode: The write mode: 'w' or 'a'.
    :return: stdout.
    """
    debug(command)

    with open(log_to_save, mode, encoding='utf-8') as f:
        f.write(f'{command}\n')
        p = subprocess.run(command, shell=True, stdout=f,
                           stderr=subprocess.STDOUT, encoding='utf-8')
    if not p.returncode == 0:
        error(f'run error in {command}. Continuing.')

def save_json(data: Union[dict, AutoDict], filename: Union[str, Path],
              separators=(',', ':'),
              indent=None):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, separators=separators, indent=indent)

def load_json(filename, object_hook=AutoDict):
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

def rot_matrix(new_position: tuple):
    """
    Create rotation matrix using Tait–Bryan angles in Z-Y-X order.
    See Wikipedia.
    :param tuple new_position: A new position (yaw, pitch, roll).
    :return:
    """
    yaw, pitch, roll = new_position

    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    cr = np.cos(roll)
    sr = np.sin(roll)

    # pitch
    mat_z = np.array(
        [[cp, -sp, 0],
         [sp, cp, 0],
         [0, 0, 1]])

    # yaw
    mat_y = np.array(
        [[cy, 0, sy],
         [0, 1, 0],
         [-sy, 0, cy]])

    # roll
    mat_x = np.array(
        [[1, 0, 0],
         [0, cr, -sr],
         [0, sr, cr]])

    return mat_y @ mat_z @ mat_x

def idx2xy(idx: int, shape: Resolution):
    tile_x = idx % shape.shape[1]
    tile_y = idx // shape.shape[1]
    return tile_x, tile_y

def xy2idx(coord: Tuple[int, int], shape: Tuple[int, int]):
    x = coord[1]
    y = coord[0]
    idx = x + y * shape[1]
    return idx

def update_dictionary(value, dictionary: AutoDict, key1: Hashable = None,
                      key2: Hashable = None, key3: Hashable = None,
                      key4: Hashable = None, key5: Hashable = None):
    dict_ = dictionary
    if key1:
        if key2:
            dict_ = dict_[key1]
        else:
            dict_[key1] = value
    if key2:
        if key3:
            dict_ = dict_[key2]
        else:
            dict_[key2] = value
    if key3:
        if key4:
            dict_ = dict_[key3]
        else:
            dict_[key3] = value
    if key4:
        if key5:
            dict_ = dict_[key4]
        else:
            dict_[key4] = value
    if key5:
        dict_[key5] = value
    return dictionary

def dishevel_dictionary(dictionary: dict, key1: Hashable = None,
                        key2: Hashable = None, key3: Hashable = None,
                        key4: Hashable = None, key5: Hashable = None) -> Any:
    disheveled_dictionary = dictionary
    if key1: disheveled_dictionary = disheveled_dictionary[key1]
    if key2: disheveled_dictionary = disheveled_dictionary[key2]
    if key3: disheveled_dictionary = disheveled_dictionary[key3]
    if key4: disheveled_dictionary = disheveled_dictionary[key4]
    if key5: disheveled_dictionary = disheveled_dictionary[key5]
    return disheveled_dictionary

def make_menu(options_txt: list) -> tuple:
    options = [str(o) for o in range(len(options_txt))]
    menu_lines = ['Options:']
    menu_lines.extend([f'{o} - {text}'
                       for o, text in zip(options, options_txt)])
    menu_lines.append(':')
    menu_txt = '\n'.join(menu_lines)
    return options, menu_txt

def menu(options_txt: list) -> int:
    options, menu_ = make_menu(options_txt)

    c = None
    while c not in options:
        c = input(menu_)

    return int(c)

def menu2(options_dict: Dict[int, Any]):
    options = []
    text = f'Options:\n'
    for idx in options_dict:
        text += f'{idx} - {options_dict[idx]}\n'
        options.append(idx)
    text += f': '

    c = None
    while c not in options:
        c = input(text)
    return c

def rem_file(file) -> None:
    if os.path.isfile(file):
        os.remove(file)

def calc_stats(data1: Iterable, data2: Iterable) \
        -> Tuple[StatsData, StatsData]:
    # Percentiles & Correlation
    per = [0, 25, 50, 75, 100]
    corr = np.corrcoef((data1, data2))[1][0]

    # Struct statistics results
    percentile_data1 = np.percentile(data1, per).T
    stats_data1 = StatsData(average=np.average(data1),
                            std=float(np.std(data1)),
                            correlation=corr,
                            min=percentile_data1[0],
                            quartile1=percentile_data1[1],
                            median=percentile_data1[2],
                            quartile3=percentile_data1[3],
                            max=percentile_data1[4],
                            )

    percentile_data2 = np.percentile(data2, per).T
    stats_data2 = StatsData(average=np.average(data2),
                            std=float(np.std(data2)),
                            correlation=corr,
                            min=percentile_data2[0],
                            quartile1=percentile_data2[1],
                            median=percentile_data2[2],
                            quartile3=percentile_data2[3],
                            max=percentile_data2[4],
                            )

    return stats_data1, stats_data2

def grouper(iterable, n, fillvalue=None):
    """
    Collect data into fixed-length chunks or blocks.
    Usage: grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx

    :param iterable: The iterable. ;-)
    :param n: The length of blocks.
    :param fillvalue: If "len(iterable)" is not multiple of "n", fill with this.
    :return: A tuple with "n" elements of "iterable".
    """
    from itertools import zip_longest
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def sobel(frame):
    """
    Apply 1st order 2D Sobel filter
    :param frame:
    :return:
    """
    sobx = ndimage.sobel(frame, axis=0)
    soby = ndimage.sobel(frame, axis=1)
    sob = np.hypot(sobx, soby)
    return sob
