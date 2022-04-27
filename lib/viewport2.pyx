from typing import Union, Iterator

import matplotlib.pyplot as plt
import numpy as np

def rot_matrix(yaw: float, pitch: float, roll: float):
    """
    Create rotation matrix using Tait–Bryan angles in Z-Y-X order.
    See Wikipedia.
    :param yaw: A new position (yaw, pitch, roll).
    :param pitch: A new position (yaw, pitch, roll).
    :param roll: A new position (yaw, pitch, roll).
    :return:
    """

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


def pix2cart(self, m, n, shape, proj='erp', f=0):
    H, W = shape

    if proj == 'erp':
        u = (m + 0.5) / W
        v = (n + 0.5) / H
        azimuth = (u - 0.5) * (2 * np.pi)
        elevation = (0.5 - v) * np.pi
        x = np.cos(azimuth) * np.cos(elevation)
        y = np.sin(elevation)
        z = -np.cos(elevation) * np.sin(azimuth)

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

    return x, y, z


class Resolution:
    W: int
    H: int

    def __init__(self, resolution: Union[str, tuple]):
        if isinstance(resolution, str):
            w, h = resolution.split('x')
        elif isinstance(resolution, tuple):
            h, w = resolution
        self.shape = h, w

    @property
    def shape(self):
        return self.H, self.W

    @shape.setter
    def shape(self, value: tuple):
        self.H = round(float(value[0]))
        self.W = round(float(value[1]))

    def __iter__(self):
        return iter((self.H, self.W))

    def __str__(self):
        return f'{self.W}x{self.H}'

    def __repr__(self):
        return f'{self.W}x{self.H}'

    def __truediv__(self, shape: tuple):
        if isinstance(shape, tuple) and len(shape) == 2:
            return Resolution((self.H / shape[0], self.W / shape[1]))


class Fov(Resolution):
    ...


class Plane:
    normal: tuple
    relation: str

    def __init__(self, z, y, x, relation='<'):
        """
        @param z: Z component of normal vector
        @param y: Y component of normal vector
        @param x: X component of normal vector
        @param relation: direction of normal.
        """
        self.normal = z, y, x
        self.relation = relation  # direction of normal. Condition to viewport


class View:
    center = (0, 0)

    def __init__(self, fov='0x0'):
        """
        The viewport is the region of sphere created by the intersection of
        four planes that pass by center of sphere and the Field of view angles.
        Each plane split the sphere in two hemispheres and consider the viewport
        overlap.
        Each plane was make using a normal vectors (x1i+y1j+z1k) and the
        equation of the plane (x1x+y2y+z1z=0)
        If we rotate the vectors, so the viewport is roted too.
        :param fov: Field-of-View in degree
        :return: None
        """
        self.fov = Fov(fov)
        fovx = np.deg2rad(self.fov.H)
        fovy = np.deg2rad(self.fov.W)

        s_fovx = np.sin(fovx / 2)
        s_fovy = np.sin(fovy / 2)
        c_fovx = np.cos(fovx / 2)
        c_fovy = np.cos(fovy / 2)

        z, y, x = c_fovy, 0, -s_fovy
        self.p1 = Plane(z, y, x)
        z, y, x = -c_fovy, 0, -s_fovy
        self.p2 = Plane(z, y, x)
        z, y, x = 0, c_fovx, -s_fovx
        self.p3 = Plane(z, y, x)
        z, y, x = 0, -c_fovx, -s_fovx
        self.p4 = Plane(z, y, x)

    def __iter__(self) -> Iterator[Plane]:
        return iter([self.p1, self.p2, self.p3, self.p4])

    def get_planes(self):
        return [self.p1, self.p2, self.p3, self.p4]


class Viewport:
    position: tuple
    projection: np.ndarray
    resolution: Resolution
    proj: str

    def __init__(self, fov: str, proj: str ='erp') -> None:
        """
        Viewport Class used to extract view pixels in projections.
        :param fov:
        """
        self.fov = Fov(fov)
        self.default_view = View(fov)
        self.rotated_view = View(fov)
        self.proj = proj

    def set_rotation(self, yaw, pitch, roll):
        """
        Set a new position to viewport using aerospace's body coordinate system
        and make the projection. Return numpy.ndarray.
        :param yaw: the positions like (yaw, pitch, roll)
        :param pitch: the positions like (yaw, pitch, roll)
        :param roll: the positions like (yaw, pitch, roll)
        :return:
        """
        self.rotated_view = self._rotate(yaw, pitch, roll)
        return self

    def _rotate(self, yaw, pitch, roll):
        """
        Rotate the normal planes of viewport using matrix of rotation and Tait–Bryan
        angles in Z-Y-X order. Refer to Wikipedia.
        :return: A new View object with the new position (yaw, pitch, roll).
        """
        new_view = View(f'{self.fov}')
        self.position = yaw, pitch, roll

        mat = rot_matrix(yaw, pitch, roll)

        # For each plane in view
        for default_plane, new_plane in zip(self.default_view, new_view):
            roted_normal = mat @ tuple(default_plane.normal)
            new_plane.normal = (roted_normal[0], roted_normal[1], roted_normal[2])

        return new_view

    def project(self, resolution: Union[str, Resolution]) -> np.ndarray:
        """
        Project the sphere using ERP. Where is Viewport the
        :param resolution: The resolution of the Viewport
        :return: a numpy.ndarray with one deep color
        """
        if isinstance(resolution, str):
            self.resolution = Resolution(resolution)
        elif isinstance(resolution, Resolution):
            self.resolution = resolution
        else:
            raise TypeError(f'Resolution must be <str> or <Resolution>.')

        self.projection = np.ones(self.resolution.shape, dtype=np.uint8) * 255
        H, W = self.projection.shape

        if self.proj == 'erp':
            for n in range(H):
                for m in range(W):
                    z, y, x = pix2cart(n, m, self.resolution.shape)
                    if self.is_viewport(z, y, x):
                        self.projection[n, m] = 0

        elif self.proj == 'cmp':
            self.projection = np.ones(self.resolution.shape, dtype=np.uint8) * 255
            for face_id in range(6):
                f_shape = (H / 2, W / 3)
                f_pos = (face_id // 3, face_id % 2)
                f_x1 = 0 + f_shape[1] * f_pos[1]
                f_x2 = f_x1 + f_shape[1]
                f_y1 = 0 + f_shape[0] * f_pos[0]
                f_y2 = f_y1 + f_shape[1]
                face_array = self.projection[f_y1:f_y2, f_x1:f_x2]
                #
                # with np.nditer(self.projection, op_flags=['readwrite'],
                #                flags=['multi_index']) as it:
                #     for cell in it:
                #         y, x = it.multi_index
                #         if self.is_viewport(pix2cart(x, y, self.resolution.shape)):
                #             cell[...] = 0

        return self.projection

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
        for plane in self.rotated_view:
            if not (plane.normal[0] * x
                    + plane.normal[1] * y
                    + plane.normal[2] * z) < 0:
                return False
        return True

    def show(self) -> None:
        plt.imshow(self.projection, cmap='gray')
        plt.show()
