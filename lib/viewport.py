from typing import Union, Iterator

import matplotlib.pyplot as plt
import numpy as np

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


class Resolution:
    W: int
    H: int

    def __init__(self, resolution: Union[str, tuple]):
        if isinstance(resolution, str):
            w, h = resolution.split('x')
            self.shape = h, w
        elif isinstance(resolution, tuple):
            self.shape = resolution

    @property
    def shape(self) -> tuple:
        return self.H, self.W

    @shape.setter
    def shape(self, shape: tuple):
        self.H = round(float(shape[0]))
        self.W = round(float(shape[1]))

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

    def __init__(self, normal=(0, 0, 0), relation='<'):
        self.normal = normal
        self.relation = relation  # With viewport


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

        self.p1 = Plane((-np.sin(fovy / 2), 0, np.cos(fovy / 2)))
        self.p2 = Plane((-np.sin(fovy / 2), 0, -np.cos(fovy / 2)))
        self.p3 = Plane((-np.sin(fovx / 2), np.cos(fovx / 2), 0))
        self.p4 = Plane((-np.sin(fovx / 2), -np.cos(fovx / 2), 0))


    def __iter__(self) -> Iterator[Plane]:
        return iter([self.p1, self.p2, self.p3, self.p4])

    def get_planes(self):
        return [self.p1, self.p2, self.p3, self.p4]


class Viewport:
    position: tuple
    projection: np.ndarray
    resolution: Resolution

    def __init__(self, fov: str) -> None:
        """
        Viewport Class used to extract view pixels in projections.
        :param fov:
        """
        self.fov = Fov(fov)
        self.default_view = View(fov)
        self.rotated_view = View(fov)

    def set_rotation(self, position: tuple):
        """
        Set a new position to viewport using aerospace's body coordinate system
        and make the projection. Return numpy.ndarray.
        :param position: the positions like (yaw, pitch, roll)
        :return:
        """
        self.position = position
        self.rotated_view = self._rotate()
        return self

    def _rotate(self):
        """
        Rotate the normal planes of viewport using matrix of rotation and Tait–Bryan
        angles in Z-Y-X order. Refer to Wikipedia.
        :return: A new View object with the new position (yaw, pitch, roll).
        """
        view = self.default_view
        new_view = View(f'{view.fov}')
        mat = rot_matrix(self.position)

        # For each plane in view
        for default_plane, new_plane in zip(view, new_view):
            roted_normal = mat @ tuple(default_plane.normal[::-1])  # change to x-y-z order
            new_plane.normal = (roted_normal[0], roted_normal[1], roted_normal[2])

        return new_view

    def project(self, resolution: Union[str, Resolution]) -> np.ndarray:
        """
        Project the sphere using ERP. Where is Viewport the
        :param resolution: The resolution of the Viewport
        :return: a numpy.ndarray with one deep color
        """
        if isinstance(resolution, str):
            self.resolution = resolution = Resolution(resolution)
        elif isinstance(resolution, Resolution):
            self.resolution = resolution = resolution

        self.projection = np.ones(resolution.shape, dtype=np.uint8) * 255

        with np.nditer(self.projection, op_flags=['readwrite'],
                       flags=['multi_index']) as it:
            for cell in it:
                y, x = it.multi_index
                if self.is_viewport(self.pix2cart(x, y)):
                    cell[...] = 0
        return self.projection

    def pix2cart(self, m, n):
        H, W = self.resolution.shape
        azimuth = ((m + 0.5) / W - 0.5) * 2 * 3.141592653589793
        elevation = -((n + 0.5) / H - 0.5) * 3.141592653589793
        x = np.cos(azimuth) * np.cos(elevation)
        y = np.sin(elevation)
        z = -np.cos(elevation) * np.sin(azimuth)
        return x, y, z

    def is_viewport(self, point: tuple) -> bool:
        """
        Check if the plane equation is true to viewport
        x1 * m + y1 * n + z1 * z < 0
        If True, the "point" is on the viewport
        Obs: is_in só retorna true se todas as expressões forem verdadeiras
        :param point: A 3D Point in the space (x, y, z).
        :return: A boolean
        """
        for plane in self.rotated_view:
            if not (plane.normal[0] * point[0]
                    + plane.normal[1] * point[1]
                    + plane.normal[2] * point[2]) < 0:
                return False
        return True

    def show(self) -> None:
        plt.imshow(self.projection, cmap='gray')
        plt.show()
