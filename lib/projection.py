import numpy as np
from collections import namedtuple
from typing import Tuple, List, Dict, NamedTuple
from enum import Enum

PI = np.pi

class Frame(np.array):
    pass

class ImShape(NamedTuple):
    y: int
    x: int

class PointCCS(NamedTuple):
    z: float
    y: float
    x: float

    def __pow__(self, power, modulo=None):
        return PointCCS(self.z**2, self.y**2, self.x**2)

class PointHCS(NamedTuple):
    elevation: float
    azimuth: float

    def __pow__(self, power, modulo=None):
        return PointHCS(self.elevation**2, self.azimuth**2)

class PointBCS(NamedTuple):
    roll: float
    pitch: float
    yaw: float

class ProjFmt(Enum):
    ERP = 0
    CMP = 1

class Projection:
    '''
    # Convention:
    X ain to front
    Y ain to top
    Z ain to right

    # Horizontal coordinate system (attached to sphere)
    yaw == azimuth -> [180, -180[    # MPEG lib360 convention
    pitch == elevation -> [90, -90]

    # Body coordinate system (attached to viewer). Ain on the enter of viewport.
    yaw -> [-180, 180[
    pitch -> [90, 90]
    roll -> [-180, 180[

    '''
    def __init__(self, proj_fmt: ProjFmt, frame: Frame = None, ):
        self.proj_fmt = proj_fmt
        self.frame = frame

        if frame is not None:
            self.shape = ImShape(*frame.shape)

def hcs2cart(elevation: float, azimuth: float):
    """
    Convert from horizontal coordinate system  in radians to cartesian system.
    :param float point: C-order indexing (elevation, azimuth)
    :return: (z, y, x)
    """
    x = np.cos(elevation) * np.cos(azimuth)
    y = np.sin(elevation)
    z = np.cos(elevation) * np.sin(azimuth)
    return z, y, x

def cart2hcs(z: float, y: float, x: float) -> tuple[float, float]:
    """
    Convert from cartesian system to horizontal coordinate system in radians
    :param point: C-order indexing (z, y, x)
    :return: (elevation, azimuth)
    """
    d = np.sqrt(x**2 + y**2 + z**2)
    azimuth = np.arctan2(z, x)
    elevation = np.arcsin(y/ d)
    return elevation, azimuth

class ERP:
    img_in: np.ndarray
    img_out: np.ndarray

    def __init__(self, img_in: np.ndarray):
        self.img_in = img_in

    def map2Dto3D(self, m, n, f=None):
        H, W = self.img_in.shape
        u = (m + 0.5) / W
        v = (n + 0.5) / H
        azimuth = (u - 0.5) * (2 * np.pi)
        elevation  = (0.5 - v) * np.pi
        return hcs2cart(elevation, azimuth)

    def map3Dto2D(self, x, y, z):
        f=0
        elevation, azimuth = cart2hcs(x, y, z)
        u = azimuth / (2 * np.pi) + 0.5
        v = -elevation / np.pi + 0.5
        return f, m, n


class CMP:
    img_in: np.ndarray
    img_out: np.ndarray

    def __init__(self, img_in: np.ndarray):
        self.img_in = img_in

    def map2Dto3D(self, m, n, f=None):
        H, W = self.img_in.shape
        A = H / 2
        u = (m + 0.5) * 2 / A - 1
        v = (n + 0.5) * 2 / A - 1
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

        return z, y, x

    def map3Dto2D(self, x, y, z):
        f=0
        elevation, azimuth = cart2hcs(x, y, z)
        u = azimuth / (2 * np.pi) + 0.5
        v = -elevation / np.pi + 0.5
        return f, m, n



