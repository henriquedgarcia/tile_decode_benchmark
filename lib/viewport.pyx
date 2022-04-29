from typing import Union
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
from .assets import Resolution as Res

def rot_matrix(yaw: float, pitch: float, roll: float) -> np.array:
    """
    Create rotation matrix using Tait–Bryan angles in Z-Y-X order.
    See Wikipedia. Use:

    Examples
    --------
    >> z, x, y = point
    >> mat = rot_matrix(yaw, pitch, roll)
    >> mat @ (z, y, x)

    :param yaw: A new position (yaw, pitch, roll).
    :param pitch: A new position (yaw, pitch, roll).
    :param roll: A new position (yaw, pitch, roll).
    :return: A 3x3 matrix of rotation for (z,y,x) vector
    """

    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    cr = np.cos(roll)
    sr = np.sin(roll)

    # pitch
    mat_z = np.array(
        [[1,  0,   0],
         [0,  cp, -sp],
         [0,  sp,  cp]])

    # yaw
    mat_y = np.array(
        [[ cy, 0, sy],
         [  0, 1,  0],
         [-sy, 0, cy]])

    # roll
    mat_x = np.array(
        [[cr, -sr, 0],
         [sr,  cr, 0],
         [ 0,   0, 1]])

    return mat_y @ mat_z @ mat_x


def pix2cart(n: int, m: int, shape: tuple[int, int], proj: str='erp', f: int=0) -> tuple[float, float, float]:
    """

    :param m: horizontal pixel coordinate
    :param n: vertical pixel coordinate
    :param shape: shape of projection in numpy format: (height, width)
    :param proj: 'erp' or 'cmp'.
    :param f: face of projection.
    :return: z, y, x
    """
    H, W = shape
    z, y, x = 0., 0., 0.

    if proj == 'erp':
        u = (m + 0.5) / W
        v = (n + 0.5) / H
        azimuth = (u - 0.5) * (2 * np.pi)
        elevation = (0.5 - v) * np.pi
        x = np.cos(azimuth) * np.cos(elevation)
        y = np.sin(elevation)
        z = np.cos(elevation) * np.sin(azimuth)

    elif proj == 'cmp':
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

    return z, y, x


class Viewport:
    position: tuple
    projection: np.ndarray
    resolution: Res
    proj: str
    base_normals: list[tuple, tuple, tuple, tuple]
    rotated_normals: list[tuple, tuple, tuple, tuple]

    def __init__(self, fov: str, proj: str ='erp') -> None:
        """
        Viewport Class used to extract view pixels in projections.
        :param fov:
        """
        self.fov = Res(fov)
        self.proj = proj
        self._make_base_normals()

    def _make_base_normals(self):
        H, W = self.fov / (2, 2)
        sin = lambda degree: np.sin(np.deg2rad(degree))
        cos = lambda degree: np.cos(np.deg2rad(degree))
        self.base_normals = [(sin(W + 90), 0, cos(W + 90)),  # z-y-x order
                             (-sin(W + 90), 0, cos(W + 90)),
                             (0, sin(H + 90), cos(H + 90)),
                             (0, -sin(H + 90), cos(H + 90))]

    def rotate(self, yaw, pitch, roll) -> Viewport:
        """
        Set a new position to viewport using aerospace's body coordinate system
        and make the projection. Rotate the normal planes of viewport using matrix of rotation and Tait–Bryan
        angles in Z-Y-X order. Refer to Wikipedia.

        :param yaw: the positions like (yaw, pitch, roll)
        :param pitch: the positions like (yaw, pitch, roll)
        :param roll: the positions like (yaw, pitch, roll)
        :return: self
        """
        self.position = yaw, pitch, roll
        mat = rot_matrix(yaw, pitch, roll)
        self.rotated_normals = []

        # For each plane in view
        self.rotated_normals = [mat @ normal for normal in self.base_normals]
        return self

    def is_viewport(self, z, y, x) -> bool:
        """
        Check if the plane equation is true to viewport
        x1 * m + y1 * n + z1 * z < 0
        If True, the "point" is on the viewport
        Obs: is_in só retorna true se todas as expressões forem verdadeiras

        :param z: A 3D Point in the space (x, y, z).
        :param y: A 3D Point in the space (x, y, z).
        :param x: A 3D Point in the space (x, y, z).
        :return: A boolean
        """
        for normal in self.rotated_normals:
            if (normal[0] * z + normal[1] * y + normal[2] * x) > 0:
                return False
        return True

    def project(self, resolution: Union[str, Res]) -> Viewport:
        """
        Project the sphere using ERP. Where is Viewport the
        :param resolution: The resolution of the Viewport ('WxH')
        :return: a numpy.ndarray with one deep color
        """
        self.resolution = Res(resolution) if isinstance(resolution, str) else resolution
        H, W = self.resolution.shape

        self.projection = np.zeros((H, W), dtype='uint8')+128

        if self.proj == 'erp':
            for n, m in product(range(H), range(W)):
                z, y, x = pix2cart(n, m, (H, W), 'erp', 0)
                if self.is_viewport(z, y, x):
                    self.projection[n, m] = 0

        elif self.proj == 'cmp':
            self.projection = np.ones(self.resolution.shape, dtype=np.uint8) * 255
            face_array = []
            for face_id in range(6):
                f_shape = (H / 2, W / 3)
                f_pos = (face_id // 3, face_id % 2)  # (line, column)
                f_x1 = 0 + f_shape[1] * f_pos[1]
                f_x2 = f_x1 + f_shape[1]
                f_y1 = 0 + f_shape[0] * f_pos[0]
                f_y2 = f_y1 + f_shape[1]
                face_array += [self.projection[f_y1:f_y2, f_x1:f_x2]]

        return self

    def show(self):
        plt.imshow(self.projection, cmap='gray')
        plt.show()

