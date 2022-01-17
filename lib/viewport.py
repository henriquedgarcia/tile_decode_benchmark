import numpy as np
from util import splitx, rot_matrix, proj2sph, hcs2cart
from assets import Fov, Point_bcs, Point_hcs, Point3d, Resolution, Point2d


class Plane:
    normal: Point3d
    relation: str

    def __init__(self, normal=Point3d(0, 0, 0), relation='<'):
        self.normal = normal
        self.relation = relation  # With viewport


class View:
    center = Point_hcs(1, 0, 0)

    def __init__(self, fov='0x0'):
        """
        The viewport is the a region of sphere created by the intersection of
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
        fovx = np.deg2rad(self.fov.h)
        fovy = np.deg2rad(self.fov.w)

        self.p1 = Plane(Point3d(-np.sin(fovy / 2), 0, np.cos(fovy / 2)))
        self.p2 = Plane(Point3d(-np.sin(fovy / 2), 0, -np.cos(fovy / 2)))
        self.p3 = Plane(Point3d(-np.sin(fovx / 2), np.cos(fovx / 2), 0))
        self.p4 = Plane(Point3d(-np.sin(fovx / 2), -np.cos(fovx / 2), 0))

    def __iter__(self):
        return iter([self.p1, self.p2, self.p3, self.p4])

    def get_planes(self):
        return [self.p1, self.p2, self.p3, self.p4]


class Viewport:
    position: Point_bcs
    projection: np.ndarray

    def __init__(self, fov: str) -> None:
        """
        Viewport Class used to extract view pixels in projections.
        :param fov:
        """
        self.fov = Fov(fov)
        self.default_view = View(fov)
        self.new_view = View(fov)

    def set_position(self, position: Point_bcs):
        """
        Set a new position to viewport using aerospace's body coordinate system
        and make the projection. Return numpy.ndarray.
        :param position:
        :return:
        """
        self.position = position
        self._rotate()
        return self

    def _rotate(self):
        """
        Rotate the normal planes of viewport using matrix of rotation and Tait–Bryan
        angles in Z-Y-X order. Refer to Wikipedia.
        :return: A new View object with the new position.
        """
        view: View = self.default_view
        new_position: Point_bcs = self.position

        new_view = View(f'{view.fov}')
        mat = rot_matrix(new_position)
        new_view.center = Point_hcs(1, new_position[0], new_position[1])

        # For each plane in view
        for default_plane, new_plane in zip(view, new_view):
            normal = default_plane.normal
            roted_normal = mat @ normal
            new_plane.normal = Point3d(roted_normal[0], roted_normal[1], roted_normal[2])

        self.new_view = new_view

    def project(self, scale: str):
        res = Resolution(*splitx(scale))
        self.projection = self._project_viewport(res)
        return self

    def _project_viewport(self, res: Resolution) -> np.ndarray:
        """
        Project the sphere using ERP. Where is Viewport the
        :param res: The resolution of the Viewport
        :return: a numpy.ndarray with one deep color
        """
        projection = np.ones(res.shape, dtype=np.uint8) * 255
        for j, i in res:
            point_hcs = proj2sph(Point2d(i, j), res)
            point_cart = hcs2cart(point_hcs)
            if self.is_viewport(point_cart):
                projection.itemset((j, i), 0)  # by the docs, it is more efficient than projection[j, i] = 0
        return projection

    def is_viewport(self, point: Point3d) -> bool:
        """
        Check if the plane equation is true to viewport
        x1 * m + y1 * n + z1 * z < 0
        If True, the "point" is on the viewport
        :param point: A 3D Point in the space.
        :return: A boolean
        """
        view = self.new_view
        for plane in view.get_planes():
            result = (plane.normal.x * point.x
                      + plane.normal.y * point.y
                      + plane.normal.z * point.z)
            if result >=0:
                return False
        return True

    def show(self) -> None:
        cv2.imshow('imagem', self.projection)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save(self, file_path):
        cv2.imwrite(file_path, self.projection)
        print('save ok')
