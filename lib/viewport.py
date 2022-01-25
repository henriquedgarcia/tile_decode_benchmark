import numpy as np
from lib.util import rot_matrix, erp2hcs, hcs2xyz
from lib.assets import Fov, PointBCS, Resolution, Point, Pixel, PointHCS
import cv2
import matplotlib.pyplot as plt
from typing import Union, Iterator

class Plane:
    normal: Point
    relation: str

    def __init__(self, normal=Point(0, 0, 0), relation='<'):
        self.normal = normal
        self.relation = relation  # With viewport


class View:
    center = PointHCS(0, 0)

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

        self.p1 = Plane(Point(-np.sin(fovy / 2), 0, np.cos(fovy / 2)))
        self.p2 = Plane(Point(-np.sin(fovy / 2), 0, -np.cos(fovy / 2)))
        self.p3 = Plane(Point(-np.sin(fovx / 2), np.cos(fovx / 2), 0))
        self.p4 = Plane(Point(-np.sin(fovx / 2), -np.cos(fovx / 2), 0))

    def __iter__(self) -> Iterator[Plane]:
        return iter([self.p1, self.p2, self.p3, self.p4])

    def get_planes(self):
        return [self.p1, self.p2, self.p3, self.p4]


class Viewport:
    position: PointBCS
    projection: np.ndarray
    resolution: Resolution

    def __init__(self, fov: str) -> None:
        """
        Viewport Class used to extract view pixels in projections.
        :param fov:
        """
        self.fov = Fov(fov)
        self.default_view = View(fov)
        self.new_view = View(fov)

    def set_rotation(self, position: PointBCS):
        """
        Set a new position to viewport using aerospace's body coordinate system
        and make the projection. Return numpy.ndarray.
        :param position:
        :return:
        """
        self.position = position
        self.new_view = self._rotate()
        return self

    def _rotate(self):
        """
        Rotate the normal planes of viewport using matrix of rotation and Tait–Bryan
        angles in Z-Y-X order. Refer to Wikipedia.
        :return: A new View object with the new position.
        """
        view = self.default_view
        new_position = self.position
        new_view = View(f'{view.fov}')
        mat = rot_matrix(new_position)

        # For each plane in view
        for default_plane, new_plane in zip(view, new_view):
            roted_normal = mat @ tuple(default_plane.normal)
            new_plane.normal = Point(roted_normal[0], roted_normal[1], roted_normal[2])

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
                sph_point = erp2hcs(Pixel(x=x, y=y), resolution)
                point_3d = hcs2xyz(sph_point)
                if self.is_viewport(point_3d):
                    cell[...] = 0
        return self.projection

    def is_viewport(self, point: Point) -> bool:
        """
        Check if the plane equation is true to viewport
        x1 * m + y1 * n + z1 * z < 0
        If True, the "point" is on the viewport
        Obs: is_in só retorna true se todas as expressões forem verdadeiras
        :param point: A 3D Point in the space.
        :return: A boolean
        """
        view = self.new_view
        is_in = True
        for plane in view:
            result = (plane.normal.x * point.x
                      + plane.normal.y * point.y
                      + plane.normal.z * point.z)
            test = (result < 0)
            is_in = is_in and test
        return is_in

    def show(self) -> None:
        plt.imshow(self.projection, cmap='gray')
        plt.show()

    def save(self, file_path):
        cv2.imwrite(file_path, self.projection)
        print('save ok')
