from itertools import product
from typing import Union

import numpy as np
from PIL import Image

from .assets import Resolution as Res
from .transform import rot_matrix, erp2cart, cart2erp, vp2cart, cart2hcs


class Viewport:
    position: tuple
    projection: np.ndarray
    resolution: Res
    proj: str
    base_normals: list[tuple, tuple, tuple]
    rotated_normals: list[tuple, tuple, tuple]

    def __init__(self, resolution: Union[str, Res], fov: Union[str, tuple, Res], proj: str = 'erp') -> None:
        """
        Viewport Class used to extract view pixels in projections.
        :param fov:
        """
        self.resolution = Res(resolution)
        self.fov = Res(fov)
        self.proj = proj
        self._make_base_normals()


    def _make_base_normals(self):
        H, W = self.fov / (2, 2)
        sin = lambda degree: np.sin(np.deg2rad(degree))
        cos = lambda degree: np.cos(np.deg2rad(degree))
        self.base_normals = [(cos(W + 90), 0, sin(W + 90)),  # x-y-z order
                             (cos(W + 90), 0, -sin(W + 90)),
                             (cos(H + 90), sin(H + 90), 0),
                             (cos(H + 90), -sin(H + 90), 0)]

    def rotate(self, yaw, pitch, roll) -> 'Viewport':
        """
        Set a new position to viewport using aerospace's body coordinate system
        and make the projection. Rotate the normal planes of viewport using matrix of rotation and Tait–Bryan
        angles in Z-Y-X order. Refer to Wikipedia.

        :param yaw: the positions like (yaw, pitch, roll) in rad
        :param pitch: the positions like (yaw, pitch, roll) in rad
        :param roll: the positions like (yaw, pitch, roll) in rad
        :return: self
        """
        self.position = yaw, pitch, roll
        mat = rot_matrix(yaw, pitch, roll)
        self.rotated_normals = []

        # For each plane in view
        self.rotated_normals = [mat @ normal for normal in self.base_normals]
        return self

    def is_viewport(self, x, y, z) -> bool:
        """
        Check if the plane equation is true to viewport
        x1 * m + y1 * n + z1 * z < 0
        If True, the "point" is on the viewport
        Obs: is_in só retorna true se todas as expressões forem verdadeiras

        :param x: A 3D Point in the space (x, y, z).
        :param y: A 3D Point in the space (x, y, z).
        :param z: A 3D Point in the space (x, y, z).
        :return: A boolean
        """
        for normal in self.rotated_normals:
            if (normal[0] * x + normal[1] * y + normal[2] * z) > 0:
                return False
        return True

    # def project(self, resolution: Union[str, Res]) -> 'Viewport':
    #     """
    #     Project the sphere using ERP. Where is Viewport the
    #     :param resolution: The resolution of the Viewport ('WxH')
    #     :return: a numpy.ndarray with one deep color
    #     """
    #     H, W = self.resolution.shape
    #
    #     self.projection = np.zeros((H, W), dtype='uint8') + 128
    #
    #     if self.proj == 'erp':
    #         for n, m in product(range(H), range(W)):
    #             x, y, z = erp2cart(n, m, (H, W))
    #             if self.is_viewport(x, y, z):
    #                 self.projection[n, m] = 0
    #
    #     elif self.proj == 'cmp':
    #         self.projection = np.ones(self.resolution.shape, dtype=np.uint8) * 255
    #         face_array = []
    #         for face_id in range(6):
    #             f_shape = (H / 2, W / 3)
    #             f_pos = (face_id // 3, face_id % 2)  # (line, column)
    #             f_x1 = 0 + f_shape[1] * f_pos[1]
    #             f_x2 = f_x1 + f_shape[1]
    #             f_y1 = 0 + f_shape[0] * f_pos[0]
    #             f_y2 = f_y1 + f_shape[1]
    #             face_array += [self.projection[f_y1:f_y2, f_x1:f_x2]]
    #
    #     return self

    def get_vp_borders(self):
        # mn -> uv -> xyz -> hcs
        # using the projection...
        # projection dependent
        H, W = self.resolution
        x_i = 0  # first row
        x_f = W  # last row
        y_i = 0  # first line
        y_f = H  # last line

        xi_xf = range(W)
        yi_yf = range(H)

        for x0 in xi_xf:
            n, m = y_i, x0
            x, y, z = vp2cart(m, n, (H, W), self.fov.shape)
            yield (m, n), (x, y, z)
        for x0 in xi_xf:
            n, m = y_f - 1, x0
            x, y, z = vp2cart(m, n, (H, W), self.fov.shape)
            yield (m, n), (x, y, z)
        for y0 in yi_yf:
            n, m = y0, x_i
            x, y, z = vp2cart(m, n, (H, W), self.fov.shape)
            yield (m, n), (x, y, z)
        for y0 in yi_yf:
            n, m = y0, x_f - 1
            x, y, z = vp2cart(m, n, (H, W), self.fov.shape)
            yield (m, n), (x, y, z)

    def show(self):
        frame_img = Image.fromarray(self.projection)
        frame_img.show()


class ERP:
    n_tiles: int
    tiles_position: dict
    projection: np.ndarray

    def __init__(self, tiling: Union[str, Res], proj_res: Union[str, Res], fov: Union[str, Res]):
        self.tiling = Res(tiling)
        self.proj_res = Res(proj_res)
        self.fov = Res(fov)
        self.viewport = Viewport(Res(proj_res), fov, 'erp')
        self.tile_res = self.proj_res / self.tiling
        self.clear_image()

    @property
    def tiles_position(self) -> dict:
        """
        The top-left pixel position of the tile.

        :return: tiles_position = {idx, (m, n), ...}
        """
        position = {}
        for n in range(0, self.proj_res.H, self.tile_res.H):
            for m in range(0, self.proj_res.W, self.tile_res.W):
                idx = len(position)
                position[idx] = (m, n)
        return position

    def clear_image(self):
        self.projection = np.zeros(self.proj_res.shape, dtype='uint8')

    def set_vp(self, yaw, pitch, roll):
        """
        in rad
        :param yaw:
        :param pitch:
        :param roll:
        :return:
        """
        self.viewport.rotate(yaw, pitch, roll)

    def get_tile_borders(self, idx: int):
        # projection agnostic
        m, n = self.tiles_position[idx]
        H, W = self.tile_res.shape

        x_i = m  # first row
        x_f = m + W  # last row
        y_i = n  # first line
        y_f = n + H  # last line

        xi_xf = range(int(x_i), int(x_f))
        yi_yf = range(int(y_i), int(y_f))

        for x0 in xi_xf:
            n, m = y_i, x0
            x, y, z = erp2cart(n, m, self.proj_res.shape)
            yield (m, n), (x, y, z)
        for x0 in xi_xf:
            n, m = y_f - 1, x0
            x, y, z = erp2cart(n, m, self.proj_res.shape)
            yield (m, n), (x, y, z)
        for y0 in yi_yf:
            n, m = y0, x_i
            x, y, z = erp2cart(n, m, self.proj_res.shape)
            yield (m, n), (x, y, z)
        for y0 in yi_yf:
            n, m = y0, x_f - 1
            x, y, z = erp2cart(n, m, self.proj_res.shape)
            yield (m, n), (x, y, z)

    def get_vptiles(self) -> list:
        if f'{self.tiling}' == '1x1':
            return [0]
        tiles = []
        for idx in self.tiles_position:
            for (m, n), (x, y, z) in self.get_tile_borders(idx):
                if self.viewport.is_viewport(x, y, z):
                    tiles.append(int(idx))
                    break
        return tiles

    def draw_vp(self, lum=255):
        """
        Project the sphere using ERP. Where is Viewport the
        :param lum: value to draw line
        :return: a numpy.ndarray with one deep color
        """
        H, W = self.proj_res.shape

        for n, m in product(range(H), range(W)):
            x, y, z = erp2cart(n, m, self.proj_res.shape)
            if self.viewport.is_viewport(x, y, z):
                self.projection[n, m] = lum

    def draw_vp_borders(self, lum=100):
        """
        Project the sphere using ERP. Where is Viewport the
        :param lum: value to draw line
        :return: a numpy.ndarray with one deep color
        """
        H, W = self.proj_res.shape

        for _, (x, y, z) in self.viewport.get_vp_borders():
            m, n = cart2erp(x, y, z, (H, W))
            self.projection[n, m] = lum

    def draw_tiles_borders(self, idx = None, lum=100):
        """
        Project border of tiles of the projection
        :param idx:
        :param lum: Value to draw lines
        :return: a numpy.ndarray with one deep color
        """

        if idx is None:
            for idx in self.tiles_position:
                for (m, n), (x, y, z) in self.get_tile_borders(idx):
                    self.projection[n, m] = lum
        else:
            for (m, n), (x, y, z) in self.get_tile_borders(idx):
                self.projection[n, m] = lum

    def draw_vp_tiles(self, lum=100):
        """
        Project the sphere using ERP. Where is Viewport the
        :param lum: Value to draw lines
        :return: a numpy.ndarray with one deep color
        """
        for tile in self.get_vptiles():
            self.draw_tiles_borders(idx=tile, lum=lum)

    def show(self):
        frame_img = Image.fromarray(self.projection)
        frame_img.show()

class CMP:
    shape: tuple

    def __init__(self, img_in: np.ndarray):
        self.img_in = img_in

    def map2Dto3D(self, m, n, f=0):
        H, W = self.img_in.shape
        A = H / 2
        u = (m + 0.5) * 2 / A - 1
        v = (n + 0.5) * 2 / A - 1
        z, y, x = 0., 0., 0.
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
        shape = self.shape
        f, m, n = 0,0,0
        return f, m, n


