import numpy as np
from typing import Union

def ______basic_____(): ...

def cart2hcs(x, y, z) -> tuple[float, float]:
    """
    Convert from cartesian system to horizontal coordinate system in radians
    :param float x: Coordinate from X axis
    :param float y: Coordinate from Y axis
    :param float z: Coordinate from Z axis
    :return: (azimuth, elevation) - in rad
    """
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    azimuth = np.arctan2(z, x)
    elevation = np.arcsin(y / r)
    return azimuth, elevation

def hcs2cart(azimuth: np.ndarray, elevation: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert from horizontal coordinate system  in radians to cartesian system.
    ISO/IEC JTC1/SC29/WG11/N17197l: Algorithm descriptions of projection format conversion and video quality metrics in 360Lib Version 5
    :param float elevation: In rad
    :param float azimuth: In rad
    :return: (x, y, z)
    """
    z = np.cos(elevation) * np.cos(azimuth)
    y = -np.sin(elevation)
    x = np.cos(elevation) * np.sin(azimuth)
    return x, y, z

def vp2cart(m, n, shape, fov_shape):
    """
    Viewport generation with rectilinear projection

    :param m:
    :param n:
    :param shape: (H, W)
    :param fov: (fov_hor, fov_vert) in degree
    :return:
    """
    H, W = shape
    fovy, fovx = map(np.deg2rad, fov_shape)
    hfovx, hfovy = fovx/2, fovy/2

    u = (m + 0.5) * 2 * np.tan(hfovx) / W
    v = (n + 0.5) * 2 * np.tan(hfovy) / H

    x = 1.
    y = -v + np.tan(hfovy)
    z = -u + np.tan(hfovx)

    r = np.sqrt(x**2 + y**2 + z**2)
    x = x / r
    y = y / r
    z = z / r

    return x, y, z

def ______erp_____(): ...

def nm2xyv(n_m_coord: np.ndarray, shape: np.ndarray) -> np.ndarray:
    """
    ERP specífic.

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
             m:Union[np.ndarray, int],
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
    H, W = shape
    u = (m + 0.5) / W
    v = (n + 0.5) / H
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
    H, W = shape

    if azimuth >= np.pi or azimuth < -np.pi:
        azimuth = (azimuth + np.pi) % (2 * np.pi)
        azimuth = azimuth - np.pi

    if elevation > np.pi / 2:
        elevation = 2 * np.pi - elevation
    elif elevation < -np.pi / 2:
        elevation = -2 * np.pi - elevation

    u = azimuth / (2 * np.pi) + 0.5
    v = -elevation / np.pi + 0.5
    m = round(u * W - 0.5)
    n = round(v * H - 0.5)
    return m, n

def ______cmp_____(): ...

def cmp2cart(n: int, m: int, shape: tuple[int, int], f: int = 0) -> tuple[float, float, float]:
    H, W = shape
    Ah = H / 2  # face is a square. u
    Aw = W / 3  # face is a square. v
    u = (m + 0.5) * 2 / Ah - 1
    v = (n + 0.5) * 2 / Aw - 1
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

def rot_matrix(yaw_pitch_roll) -> np.array:
    """
    Create rotation matrix using Tait–Bryan angles in Z-Y-X order.
    See Wikipedia. Use:

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
