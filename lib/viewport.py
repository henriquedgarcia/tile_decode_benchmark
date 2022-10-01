from math import pi
from typing import Union, Callable

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from .transform import rot_matrix
from .util import splitx


class Viewport:
    rotated_normals: np.ndarray

    def __init__(self, vp_shape: np.ndarray, fov: np.ndarray):
        """ self.tiling, self.fov, 'erp'
        Viewport Class used to extract view pixels in projections.
        :param frame vp_shape: (600, 800) for 800x600px
        :param fov: in rad. Ex: "np.array((pi/2, pi/2))" for (90°x90°)
        """
        self.fov = fov

        self.vp_shape = vp_shape
        self.vp_image = np.zeros(self.vp_shape)

        self.yaw_pitch_roll = np.array([0, 0, 0])
        self.mat = rot_matrix(self.yaw_pitch_roll)

        self._make_base_normals()
        self._make_vp_coord()

    def _make_base_normals(self) -> None:
        """
        Com eixo entrando no observador, rotação horário é negativo e anti-horária
        é positivo. Todos os ângulos são radianos.

        O eixo z aponta para a frente
        O exito x aponta para a direita
        O eixo y aponta para baixo

        Deslocamento para a direita e para cima é positivo.

        O viewport é a região da esfera que faz intersecção com 4 planos que passam pelo
          centro (4 grandes círculos): cima, direita, baixo e esquerda.
        Os planos são definidos tal que suas normais (N) parte do centro e apontam na mesma direção a
          região do viewport. Ex: O plano de cima aponta para cima, etc.
        Todos os píxeis que estiverem abaixo do plano {N(x,y,z) dot P(x,y,z) <= 0}
        O plano de cima possui incinação de FOV_Y / 2.
          Sua normal é x=0,y=sin(FOV_Y/2 + pi/2), z=cos(FOV_Y/2 + pi/2)
        O plano de baixo possui incinação de -FOV_Y / 2.
          Sua normal é x=0,y=sin(-FOV_Y/2 - pi/2), z=cos(-FOV_Y/2 - pi/2)
        O plano da direita possui incinação de FOV_X / 2. (para direita)
          Sua normal é x=sin(FOV_X/2 + pi/2),y=0, z=cos(FOV_X/2 + pi/2)
        O plano da esquerda possui incinação de -FOV_X/2. (para direita)
          Sua normal é x=sin(-FOV_X/2 - pi/2),y=0, z=cos(-FOV_X/2 - pi/2)

        :return:
        """
        fov_y_2, fov_x_2 = self.fov / (2, 2)
        pi_2 = np.pi / 2

        self.base_normals = np.array([[0, -np.sin(fov_y_2 + pi_2), np.cos(fov_y_2 + pi_2)],  # top
                                      [0, -np.sin(-fov_y_2 - pi_2), np.cos(-fov_y_2 - pi_2)],  # bottom
                                      [np.sin(fov_x_2 + pi_2), 0, np.cos(fov_x_2 + pi_2)],  # left
                                      [np.sin(-fov_x_2 - pi_2), 0, np.cos(-fov_x_2 - pi_2)]])  # right
        self.rotated_normals = self.base_normals.copy()

    def _make_vp_coord(self) -> None:
        tan_fov_2 = np.tan(self.fov / 2)

        vp_coord_x, vp_coord_y = np.meshgrid(np.linspace(-tan_fov_2[1], tan_fov_2[1], self.vp_shape[1], endpoint=False),
                                             np.linspace(-tan_fov_2[0], tan_fov_2[0], self.vp_shape[0], endpoint=True))
        vp_coord_z = np.ones(self.vp_shape)
        vp_coord_xyz_ = np.array([vp_coord_x, vp_coord_y, vp_coord_z]).transpose((1, 2, 0))

        r = np.sqrt(np.sum(vp_coord_xyz_ ** 2, axis=2, keepdims=True))

        vp_coord_xyz = vp_coord_xyz_ / r
        self.vp_coord_xyz = vp_coord_xyz.reshape(-1, 3).T  # shape==(H,W,3)

        self.vp_rotated_xyz = vp_coord_xyz.copy()

    def rotate(self, yaw_pitch_roll: np.ndarray) -> None:
        """
        Set a new position to viewport using aerospace's body coordinate system
        and make the projection. Rotate the normal planes of viewport using matrix of rotation and Tait–Bryan
        angles in Z-Y-X order. Refer to Wikipedia.

        :param yaw_pitch_roll: the positions like (yaw, pitch, roll) in rad
        :return:
        """
        self.yaw_pitch_roll = yaw_pitch_roll
        self.mat = rot_matrix(yaw_pitch_roll)

        self.rotated_normals = (self.mat @ self.base_normals.T).T

    def is_viewport(self, x_y_z) -> bool:
        """
        Check if the plane equation is true to viewport
        x1 * m + y1 * n + z1 * z < 0
        If True, the "point" is on the viewport
        Obs: is_in só retorna true se todas as expressões forem verdadeiras

        :param x_y_z: A 3D Point in the space [(x, y, z)].
        :return: A boolean         belong = np.all(inner_product <= 0, axis=0).reshape(self.shape)

        """
        inner_prod = np.dot(self.rotated_normals, x_y_z.reshape((-1, 3)).T)
        px_in_vp = np.all(inner_prod <= 0, axis=0)
        return np.any(px_in_vp)

    def get_vp_borders_xyz(self, thickness=1, yaw_pitch_roll: np.ndarray = None) -> np.ndarray:
        """

        :param thickness: in pixels
        :param yaw_pitch_roll: a new head position. shape==(3,). (optional)
        :return: np.ndarray (shape == (1,HxW,3)
        """
        self._rotate_vp(yaw_pitch_roll)
        return get_borders(frame=self.vp_rotated_xyz, thickness=thickness)

    def _rotate_vp(self, yaw_pitch_roll: np.ndarray = None) -> None:
        """

        :param yaw_pitch_roll: a new head position. shape==(3,). (optional)
        :return: None
        """

        H, W = self.vp_shape[:2]
        vp_rotated_xyz_list = (self.mat @ self.vp_coord_xyz).T
        self.vp_rotated_xyz = vp_rotated_xyz_list.reshape((H, W, 3))

    def get_vp(self, frame: np.ndarray, xyz2nm: Callable, yaw_pitch_roll: np.ndarray = None) -> np.ndarray:
        """

        :param frame: The projection image.
        :param xyz2nm: A function from 3D to projection.
        :param yaw_pitch_roll: a new head position. (optional)
        :return: The viewport image (RGB)
        """
        if yaw_pitch_roll:
            self.rotate(yaw_pitch_roll)

        H, W = self.vp_shape[:2]
        vp_rotated_xyz_list = (self.mat @ self.vp_coord_xyz).T
        self.vp_rotated_xyz = vp_rotated_xyz_list.reshape((H, W, 3))

        nm_coord = xyz2nm(self.vp_rotated_xyz, frame.shape)

        out = cv2.remap(frame,
                        map1=nm_coord[..., 1:2].astype(np.float32),
                        map2=nm_coord[..., 0:1].astype(np.float32),
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_WRAP)
        # show2(out)
        return out


class ERP:
    vp_image: np.ndarray  # A RGB image

    def __init__(self, tiling: str, proj_res: str, fov: str, vp_shape: np.ndarray = None):
        self.fov = np.deg2rad(splitx(fov)[::-1])
        self.shape = np.array(splitx(proj_res)[::-1], dtype=int)
        self.vp_shape = vp_shape
        if vp_shape is None:
            self.vp_shape = np.round(self.fov * self.shape / (pi, 2 * pi)).astype('int')
        self.tiling = np.array(splitx(tiling)[::-1], dtype=int)
        self.n_tiles = self.tiling[0] * self.tiling[1]

        self.viewport = Viewport(self.vp_shape, self.fov)

        self.projection = np.zeros(self.shape, dtype='uint8')
        self.nm_coord = np.mgrid[range(self.shape[0]), range(self.shape[1])].transpose((1, 2, 0))
        self.xyz_coord = self.nm2xyz(self.nm_coord)

        self.tile_res = (self.shape / self.tiling).astype(int)
        self.border_base = get_borders(shape=self.tile_res)
        self.tiles_position = np.array([(n, m)
                                        for n in range(0, self.shape[0], self.tile_res[0])
                                        for m in range(0, self.shape[1], self.tile_res[1])])

    def get_vptiles(self) -> list:
        if tuple(self.tiling) == (1, 1):
            return [0]

        tiles = []
        for tile in range(self.n_tiles):
            borders = self.get_tile_borders(tile)
            borders_3d = self.nm2xyz(borders, shape=self.shape)
            is_vp = self.viewport.is_viewport(borders_3d)
            if is_vp:
                tiles.append(str(tile))
        return tiles

    def get_viewport(self, frame: np.ndarray, yaw_pitch_roll: np.ndarray = None) -> np.ndarray:
        self.vp_image = self.viewport.get_vp(frame=frame, xyz2nm=self.xyz2nm, yaw_pitch_roll=yaw_pitch_roll)
        return self.vp_image

    def draw_all_tiles_borders(self, lum=100):
        for tile in range(self.n_tiles):
            self.draw_tile_border(tile, lum)

    def draw_vp_tiles(self, lum=100):
        for tile in self.get_vptiles():
            self.draw_tile_border(idx=tile, lum=lum)

    def draw_vp(self, lum=255):
        """
        Project the sphere using ERP. Where is Viewport the
        :param lum: value to draw line
        :return: a numpy.ndarray with one deep color
        """
        inner_product = self.viewport.rotated_normals @ self.xyz_coord.reshape(-1, 3).T
        belong = np.all(inner_product <= 0, axis=0).reshape(self.shape)
        self.projection[belong] = lum

    def draw_vp_borders(self, lum=100, thickness=1):
        """
        Project the sphere using ERP. Where is Viewport the
        :param lum: value to draw line
        :param thickness: in pixel.
        :return: a numpy.ndarray with one deep color
        """
        vp_borders_xyz = self.viewport.get_vp_borders_xyz(thickness=thickness)
        nm_coord = self.xyz2nm(vp_borders_xyz, shape=self.shape, round_nm=True).astype(int)
        self.projection[nm_coord[0, :, 0], nm_coord[0, :, 1]] = lum

    def get_tile_borders(self, idx: int):
        # projection agnostic

        border_n_m = self.border_base + self.tiles_position[idx]
        return border_n_m

    def draw_tile_border(self, idx, lum=100):
        borders = self.get_tile_borders(idx)
        n, m = borders.T
        self.projection[n, m] = lum

    def show(self):
        show1(self.projection)

    def clear_projection(self):
        # self.projection = np.zeros(self.proj_res.shape, dtype='uint8')
        self.projection = np.zeros(self.shape, dtype='uint8')

    @staticmethod
    def nm2xyz(nm_coord, shape=None):
        """
        ERP specific.

        :param nm_coord: [[(0, 0), ...], [(n, m), ...], ..., [..., (N-1, M-1)]
        :param shape: (N, M)
        :return:
        """
        if shape is None:
            shape = nm_coord.shape[:2]

        normalize = (nm_coord + (0.5, 0.5)) / shape
        shift = normalize - (0.5, 0.5)
        elevation_azimuth = shift * (-np.pi, 2 * np.pi)

        z = np.cos(elevation_azimuth[:, :, 0]) * np.cos(elevation_azimuth[:, :, 1])
        y = -np.sin(elevation_azimuth[:, :, 0])
        x = np.cos(elevation_azimuth[:, :, 0]) * np.sin(elevation_azimuth[:, :, 1])
        xyz_coord = np.array([x, y, z]).transpose((1, 2, 0))
        return xyz_coord

    @staticmethod
    def xyz2nm(xyz_coord: np.ndarray, shape: np.ndarray = None, round_nm: bool = False):
        """
        ERP specific.

        :param xyz_coord: [[[x, y, z], ..., M], ..., N] (shape == (N,M,3))
        :param shape: the shape of projection that cover all sphere
        :param round_nm: round the coords? is not needed.
        :return:
        """
        if shape is None:
            shape = xyz_coord.shape[:2]

        N, M = shape[:2]

        r = np.sqrt(np.sum(xyz_coord ** 2, axis=2))

        elevation = np.arcsin(xyz_coord[..., 1] / r)
        azimuth = np.arctan2(xyz_coord[..., 0], xyz_coord[..., 2])

        v = elevation / pi + 0.5
        u = azimuth / (2 * pi) + 0.5

        n = v * N - 0.5
        m = u * M - 0.5

        if round_nm:
            n = np.mod(np.round(n), N)
            m = np.mod(np.round(m), M)

        return np.array([n, m]).transpose((1, 2, 0))


# class CMP:
#     shape: tuple
#
#     def __init__(self, img_in: np.ndarray):
#         self.img_in = img_in
#
#     def map2Dto3D(self, m, n, f=0):
#         H, W = self.img_in.shape
#         A = H / 2
#         u = (m + 0.5) * 2 / A - 1
#         v = (n + 0.5) * 2 / A - 1
#         z, y, x = 0., 0., 0.
#         if f == 0:
#             x = 1.0
#             y = -v
#             z = -u
#         elif f == 1:
#             x = -1.0
#             y = -v
#             z = u
#         elif f == 2:
#             x = u
#             y = 1.0
#             z = v
#         elif f == 3:
#             x = u
#             y = -1.0
#             z = -v
#         elif f == 4:
#             x = u
#             y = -v
#             z = 1.0
#         elif f == 5:
#             x = -u
#             y = -v
#             z = -1.0
#
#         return z, y, x
#
#     def map3Dto2D(self, x, y, z):
#         f = 0
#         elevation, azimuth = cart2hcs(x, y, z)
#         u = azimuth / (2 * np.pi) + 0.5
#         v = -elevation / np.pi + 0.5
#         shape = self.shape
#         f, m, n = 0, 0, 0
#         return f, m, n
#
#     def project(self, resolution: Union[str, Res]) -> 'Viewport':
#         """
#         Project the sphere using ERP. Where is Viewport the
#         :param resolution: The resolution of the Viewport ('WxH')
#         :return: a numpy.ndarray with one deep color
#         """
#         H, W = self.resolution.shape
#
#         self.projection = np.zeros((H, W), dtype='uint8') + 128
#
#         if self.proj == 'erp':
#             for n, m in product(range(H), range(W)):
#                 x, y, z = erp2cart(n, m, (H, W))
#                 if self.is_viewport(x, y, z):
#                     self.projection[n, m] = 0
#
#         elif self.proj == 'cmp':
#             self.projection = np.ones(self.resolution.shape, dtype=np.uint8) * 255
#             face_array = []
#             for face_id in range(6):
#                 f_shape = (H / 2, W / 3)
#                 f_pos = (face_id // 3, face_id % 2)  # (line, column)
#                 f_x1 = 0 + f_shape[1] * f_pos[1]
#                 f_x2 = f_x1 + f_shape[1]
#                 f_y1 = 0 + f_shape[0] * f_pos[0]
#                 f_y2 = f_y1 + f_shape[1]
#                 face_array += [self.projection[f_y1:f_y2, f_x1:f_x2]]
#
#         return self


def show1(projection: np.ndarray):
    plt.imshow(projection)
    plt.show()


def show2(projection: np.ndarray):
    frame_img = Image.fromarray(projection)
    frame_img.show()


def get_borders(*, frame: Union[tuple, np.ndarray] = None, shape=None, thickness=1):
    if frame is None:
        frame = np.mgrid[range(shape[0]), range(shape[1])].transpose((1, 2, 0))
    shape = frame.shape
    left = frame[:, 0:thickness, :].reshape((-1, shape[2]))
    right = frame[:, :- 1 - thickness:-1, :].reshape((-1, shape[2]))
    top = frame[0:thickness, :, :].reshape((-1, shape[2]))
    bottom = frame[:- 1 - thickness:-1, :, :].reshape((-1, shape[2]))

    return np.r_[top, right, bottom, left].reshape((1, -1, shape[2]))


def main():
    # erp '144x72', '288x144','432x216','576x288'
    # cmp '144x96', '288x192','432x288','576x384'
    erp = ERP('6x4', '576x288', '90x90')
    height, width = erp.shape

    frame_img: Union[np.ndarray, Image.Image] = Image.open('lib/images/erp1.jpg')
    frame_img = frame_img.resize((width, height))

    yaw_pitch_roll = np.deg2rad((-30, 20, -10))

    erp.viewport.rotate(yaw_pitch_roll)
    vp_image = erp.get_viewport(np.asarray(frame_img))

    # Draw all tiles border
    erp.clear_projection()
    erp.draw_all_tiles_borders(lum=255)
    cover = Image.new("RGB", (width, height), (0, 0, 0))
    frame_img = Image.composite(cover, frame_img, mask=Image.fromarray(erp.projection))

    # Draw tiles in viewport
    erp.clear_projection()
    erp.draw_vp_tiles(lum=255)
    cover = Image.new("RGB", (width, height), (0, 255, 0))
    frame_img = Image.composite(cover, frame_img, mask=Image.fromarray(erp.projection))

    # Draw viewport
    erp.clear_projection()
    erp.draw_vp(lum=200)
    cover = Image.new("RGB", (width, height), (200, 200, 200))
    frame_img = Image.composite(cover, frame_img, mask=Image.fromarray(erp.projection))

    # Draw viewport borders
    erp.clear_projection()
    erp.draw_vp_borders(lum=255)
    cover = Image.new("RGB", (width, height), (0, 0, 255))
    frame_img = Image.composite(cover, frame_img, mask=Image.fromarray(erp.projection))

    tiles = erp.get_vptiles()
    print(f'The tiles {tiles} have pixel in viewport.')

    width_vp = int(np.round(height * vp_image.shape[1]/vp_image.shape[0]))
    vp_image = Image.fromarray(vp_image).resize(( width_vp, height))

    new_im = Image.new('RGB', (width+width_vp+2, height), (255,255,255))
    new_im.paste(frame_img, (0,0))
    new_im.paste(vp_image, (width+2,0))

    new_im.show()


if __name__ == '__main__':
    main()
