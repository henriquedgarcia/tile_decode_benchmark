from math import ceil
from typing import Union

import numpy as np
import pandas as pd


def ______basic_____(): ...


def cart2hcs(x, y, z) -> tuple[float, float]:
    """
    Convert from cartesian system to horizontal coordinate system in radians
    :param float x: Coordinate from X axis
    :param float y: Coordinate from Y axis
    :param float z: Coordinate from Z axis
    :return: (azimuth, elevation) - in rad
    """
    # z-> x,
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    azimuth = np.arctan2(x, z)
    elevation = np.arcsin(-y / r)
    return azimuth, elevation


# todo: Verificar conscistência com arquivo quality.py
#
# def cart2hcs(x_y_z: np.ndarray) -> np.ndarray:
#     """
#     Convert from cartesian system to horizontal coordinate system in radians
#     :param x_y_z: 1D ndarray [x, y, z], or 2D array with shape=(N, 3)
#     :return: (azimuth, elevation) - in rad
#     """
#     r = np.sqrt(np.sum(x_y_z ** 2))
#     azimuth = np.arctan2(x_y_z[..., 0], x_y_z[..., 2])
#     elevation = np.arcsin(-x_y_z[..., 1] / r)
#     return np.array([azimuth, elevation]).T


def hcs2cart(azimuth: Union[np.ndarray, float], elevation: Union[np.ndarray, float]) \
        -> tuple[Union[np.ndarray, float], Union[np.ndarray, float], Union[np.ndarray, float]]:
    """
    Convert from horizontal coordinate system  in radians to cartesian system.
    ISO/IEC JTC1/SC29/WG11/N17197l: Algorithm descriptions of projection format conversion and video quality metrics in 360Lib Version 5
    :param float elevation: Rad
    :param float azimuth: Rad
    :return: (x, y, z)
    """
    z = np.cos(elevation) * np.cos(azimuth)
    y = -np.sin(elevation)
    x = np.cos(elevation) * np.sin(azimuth)
    return x, y, z


def vp2cart(m, n, proj_shape, fov_shape):
    """
    Viewport generation with rectilinear projection

    :param m:
    :param n:
    :param proj_shape: (H, W)
    :param fov_shape: (fov_hor, fov_vert) in degree
    :return:
    """
    proj_h, proj_w = proj_shape
    fov_y, fov_x = map(np.deg2rad, fov_shape)
    half_fov_x, half_fov_y = fov_x / 2, fov_y / 2

    u = (m + 0.5) * 2 * np.tan(half_fov_x) / proj_w
    v = (n + 0.5) * 2 * np.tan(half_fov_y) / proj_h

    x = 1.
    y = -v + np.tan(half_fov_y)
    z = -u + np.tan(half_fov_x)

    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    x = x / r
    y = y / r
    z = z / r

    return x, y, z


def ______erp_____(): ...


def nm2xyv(n_m_coord: np.ndarray, shape: np.ndarray) -> np.ndarray:
    """
    ERP specific.

    :param n_m_coord: [(n, m], ...]
    :param shape: (H, W)
    :return:
    """
    v_u = (n_m_coord + (0.5, 0.5)) / shape
    elevation, azimuth = ((v_u - (0.5, 0.5)) * (-np.pi, 2 * np.pi)).T

    z = np.cos(elevation) * np.cos(azimuth)
    y = -np.sin(elevation)
    x = np.cos(elevation) * np.sin(azimuth)
    return np.array([x, y, z]).T


def erp2cart(n: Union[np.ndarray, int],
             m: Union[np.ndarray, int],
             shape: Union[np.ndarray, tuple[int, int]]) -> np.ndarray:
    """

    :param m: horizontal pixel coordinate
    :param n: vertical pixel coordinate
    :param shape: shape of projection in numpy format: (height, width)
    :return: x, y, z
    """
    azimuth, elevation = erp2hcs(n, m, shape)
    x, y, z = hcs2cart(azimuth, elevation)
    return np.array(x, y, z)


def cart2erp(x, y, z, shape):
    azimuth, elevation = cart2hcs(x, y, z)
    m, n = hcs2erp(azimuth, elevation, shape)
    return m, n


def erp2hcs(n: Union[np.ndarray, int], m: Union[np.ndarray, int], shape: Union[np.ndarray, tuple[int, int]]) -> Union[np.ndarray, tuple[float, float]]:
    """

    :param m: horizontal pixel coordinate
    :param n: vertical pixel coordinate
    :param shape: shape of projection in numpy format: (height, width)
    :return: (azimuth, elevation) - in rad
    """
    proj_h, proj_w = shape
    u = (m + 0.5) / proj_w
    v = (n + 0.5) / proj_h
    azimuth = (u - 0.5) * (2 * np.pi)
    elevation = (0.5 - v) * np.pi
    return azimuth, elevation


def hcs2erp(azimuth: float, elevation: float, shape: tuple) -> tuple[int, int]:
    """

    :param azimuth: in rad
    :param elevation: in rad
    :param shape: shape of projection in numpy format: (height, width)
    :return: (m, n) pixel coord using nearest neighbor
    """
    proj_h, proj_w = shape

    if azimuth >= np.pi or azimuth < -np.pi:
        azimuth = (azimuth + np.pi) % (2 * np.pi)
        azimuth = azimuth - np.pi

    if elevation > np.pi / 2:
        elevation = 2 * np.pi - elevation
    elif elevation < -np.pi / 2:
        elevation = -2 * np.pi - elevation

    u = azimuth / (2 * np.pi) + 0.5
    v = -elevation / np.pi + 0.5
    m = ceil(u * proj_w - 0.5)
    n = ceil(v * (proj_h - 1) - 0.5)
    return m, n


def ______cmp_____(): ...


def hcs2cmp(azimuth: float, elevation: float, shape: tuple) -> tuple[int, int, int]:
    """

    :param azimuth: in rad
    :param elevation: in rad
    :param shape: shape of projection in numpy format: (height, width)
    :return: (m, n) pixel coord using nearest neighbor
    """
    proj_h, proj_w = shape
    u, v, face = None, None, None
    x, y, z = hcs2cart(azimuth, elevation)
    ax, ay, az = np.abs((x, y, z))

    if ax >= ay and ax >= az and x > 0:  # face 0
        face = 0
        u = -z / ax
        v = -y / ax
    if ax >= ay and ax >= az and x < 0:
        face = 1
        u = z / ax
        v = -y / ax

    if ay >= ax and ay >= az and y > 0:
        face = 2
        u = -x / ay
        v = -z / ay
    if ay >= ay >= az and y < 0:
        face = 3
        u = x / ay
        v = -z / ay

    if az >= ax and az >= ax and z > 0:
        face = 4
        u = x / az
        v = -y / az
    if az >= ax and az >= ay and z < 0:
        face = 5
        u = -x / az
        v = -y / az

    a = int(proj_h / 2)  # suppose the face is a square
    m = int((u + 1) * a / 2)
    n = int((v + 1) * a / 2)

    if face == 4:
        m = m
        n = n
    elif face == 0:
        m = m + a
        n = n
    elif face == 5:
        m = m + 2 * a
        n = n
    elif face == 3:
        m = a - n
        n = m + a
    elif face == 1:
        m = 2 * a - n
        n = m + a
    elif face == 2:
        m = 3 * a - n
        n = m + a

    return m, n, face


def cmp2cart(n: int, m: int, shape: tuple[int, int], f: int = 0) -> tuple[float, float, float]:
    x, y, z = None, None, None
    proj_h, proj_w = shape
    face_h = proj_h / 2  # face is a square. u
    face_w = proj_w / 3  # face is a square. v
    u = (m + 0.5) * 2 / face_h - 1
    v = (n + 0.5) * 2 / face_w - 1
    if f == 0:
        x = 1.0
        y = -v
        z = -u
    elif f == 1:
        x = -1.0
        y = -v
        z = u
    elif f == 2:
        x = u
        y = 1.0
        z = v
    elif f == 3:
        x = u
        y = -1.0
        z = -v
    elif f == 4:
        x = u
        y = -v
        z = 1.0
    elif f == 5:
        x = -u
        y = -v
        z = -1.0
    return x, y, z


def ______utils_____(): ...


def rot_matrix(yaw_pitch_roll: Union[np.ndarray, list]) -> np.ndarray:
    """
    Create rotation matrix using Tait–Bryan angles in Z-Y-X order.
    See Wikipedia. Use:
        X axis point to right
        Y axis point to down
        Z axis point to front

    Examples
    --------
    >> x, y, z = point
    >> mat = rot_matrix(yaw, pitch, roll)
    >> mat @ (x, y, z)

    :param yaw_pitch_roll: the rotation (yaw, pitch, roll) in rad.
    :return: A 3x3 matrix of rotation for (z,y,x) vector
    """
    cos_rot = np.cos(yaw_pitch_roll)
    sin_rot = np.sin(yaw_pitch_roll)

    # pitch
    mat_x = np.array(
        [[1, 0, 0],
         [0, cos_rot[1], -sin_rot[1]],
         [0, sin_rot[1], cos_rot[1]]]
    )
    # yaw
    mat_y = np.array(
        [[cos_rot[0], 0, sin_rot[0]],
         [0, 1, 0],
         [-sin_rot[0], 0, cos_rot[0]]]
    )
    # roll
    mat_z = np.array(
        [[cos_rot[2], -sin_rot[2], 0],
         [sin_rot[2], cos_rot[2], 0],
         [0, 0, 1]]
    )

    return mat_y @ mat_x @ mat_z


def check_deg(axis_name: str, value: float) -> float:
    """

    @param axis_name:
    @param value: in rad
    @return:
    """
    n_value = None
    if axis_name == 'azimuth':
        if value >= np.pi or value < -np.pi:
            n_value = (value + np.pi) % (2 * np.pi)
            n_value = n_value - np.pi
        return n_value
    elif axis_name == 'elevation':
        if value > np.pi / 2:
            n_value = 2 * np.pi - value
        elif value < -np.pi / 2:
            n_value = -2 * np.pi - value
        return n_value
    else:
        raise ValueError('"axis_name" not exist.')


def lin_interpol(t: float,
                 t_f: float, t_i: float,
                 v_f: np.ndarray, v_i: np.ndarray) -> np.ndarray:
    m: np.ndarray = (v_f - v_i) / (t_f - t_i)
    v: np.ndarray = m * (t - t_i) + v_i
    return v


def position2trajectory(positions_list, fps=30):
    # in rads: positions_list == (y, p, r)
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
        position: pd.Series
        position.index == []
        """
        yaw = position[0]
        pitch = position[1]

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

        new_pitch = pitch + pi / 2 * pitch_state
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

    # incomplete
    return yaw_speed, pitch_speed
