from math import pi
from typing import Union, Callable

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

try:
    from .transform import rot_matrix
    from .util import splitx
except ImportError:
    from transform import rot_matrix
    from util import splitx


class Viewport:
    base_normals: np.ndarray
    fov: np.ndarray
    updated_array: set
    vp_coord_xyz: np.ndarray
    vp_shape: np.ndarray

    _yaw_pitch_roll: np.ndarray
    _mat_rot: np.ndarray
    _rotated_normals: np.ndarray
    _vp_rotated_xyz: np.ndarray
    _vp_borders_xyz: np.ndarray
    _is_in_vp: bool
    _out: np.ndarray

    def __init__(self, vp_shape: np.ndarray, fov: np.ndarray):
        """
        Viewport Class used to extract view pixels in projections.
        The vp is an image as numpy array with shape (H, M, 3).
        That can be RGB (matplotlib, pillow, etc) or BGR (opencv).

        :param frame vp_shape: (600, 800) for 800x600px
        :param fov: in rad. Ex: "np.array((pi/2, pi/2))" for (90°x90°)
        """
        self.fov = fov
        self.vp_shape = vp_shape
        self.updated_array = set()
        self._make_base_normals()
        self._make_base_vp_coord()

        self._yaw_pitch_roll = np.array([0, 0, 0])

    def is_viewport(self, x_y_z: np.ndarray) -> bool:
        """
        Check if the plane equation is true to viewport
        x1 * m + y1 * n + z1 * z < 0
        If True, the "point" is on the viewport
        Obs: is_in só retorna true se todas as expressões forem verdadeiras

        :param x_y_z: A 3D Point list in the space [(x, y, z), ...].T, shape == (3, ...)
        :return: A boolean         belong = np.all(inner_product <= 0, axis=0).reshape(self.shape)

        """
        inner_prod = self.rotated_normals.T @ x_y_z
        px_in_vp = np.all(inner_prod <= 0, axis=0)
        self._is_in_vp = np.any(px_in_vp)
        return self._is_in_vp

    def rotate_vp(self) -> np.ndarray:
        if "rotate_vp" not in self.updated_array:
            self._vp_rotated_xyz = np.tensordot(self.mat_rot, self.vp_coord_xyz, axes=1)
            self.updated_array.update(["rotate_vp"])
        return self._vp_rotated_xyz

    def get_vp(self, frame: np.ndarray, xyz2nm: Callable) -> np.ndarray:
        """

        :param frame: The projection image.
        :param xyz2nm: A function from 3D to projection.
        :return: The viewport image (RGB)
        """
        self.rotate_vp()
        nm_coord: np.ndarray = xyz2nm(self._vp_rotated_xyz, frame.shape)
        nm_coord = nm_coord.transpose((1, 2, 0))
        self._out = cv2.remap(frame,
                              map1=nm_coord[..., 1:2].astype(np.float32),
                              map2=nm_coord[..., 0:1].astype(np.float32),
                              interpolation=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_WRAP)
        # show2(self._out)
        return self._out

    def get_vp_borders_xyz(self, thickness: int = 1) -> np.ndarray:
        """

        :param thickness: in pixels
        :return: np.ndarray (shape == (1,HxW,3)
        """
        self.rotate_vp()
        if "get_vp_borders_xyz" not in self.updated_array:
            self._vp_borders_xyz = get_borders(coord_nm=self._vp_rotated_xyz, thickness=thickness)
            self.updated_array.update(["get_vp_borders_xyz"])
        return self._vp_borders_xyz

    def _make_base_normals(self) -> None:
        """
        Com eixo entrando no observador, rotação horário é negativo e anti-horária
        é positivo. Todos os ângulos são radianos.

        O exito x aponta para a direita
        O eixo y aponta para baixo
        O eixo z aponta para a frente

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
                                      [np.sin(-fov_x_2 - pi_2), 0, np.cos(-fov_x_2 - pi_2)]]).T  # right

    def _make_base_vp_coord(self) -> None:
        """
        The VP projection is based in rectilinear projection.

        In the sphere domain, in te cartesian system, the center of a plain touch the sphere
        on the point (x=0,y=0,z=1).
        The plain sizes are based on the tangent of fov.
        The resolution (number of samples) of viewport is defined by the constructor.
        The proj_coord_xyz.shape is (3,H,W). The dim 0 are x, y z coordinates.
        :return:
        """
        tan_fov_2 = np.tan(self.fov / 2)
        x_coord = np.linspace(-tan_fov_2[1], tan_fov_2[1], self.vp_shape[1], endpoint=False)
        y_coord = np.linspace(-tan_fov_2[0], tan_fov_2[0], self.vp_shape[0], endpoint=True)

        (vp_coord_x, vp_coord_y), vp_coord_z = np.meshgrid(x_coord, y_coord), np.ones(self.vp_shape)
        vp_coord_xyz_ = np.array([vp_coord_x, vp_coord_y, vp_coord_z])

        r = np.sqrt(np.sum(vp_coord_xyz_ ** 2, axis=0, keepdims=True))

        self.vp_coord_xyz = vp_coord_xyz_ / r  # normalize. final shape==(3,H,W)

    @property
    def yaw_pitch_roll(self) -> np.ndarray:
        return self._yaw_pitch_roll

    @yaw_pitch_roll.setter
    def yaw_pitch_roll(self, value: np.ndarray):
        """
        Set a new position to viewport using aerospace's body coordinate system
        and rotate the normals. Rotate the normal planes of viewport using matrix of rotation and Tait–Bryan
        angles in Y-X-Z order. Refer to Wikipedia.

        :param value: the positions like array(yaw, pitch, roll) in rad
        """
        if not np.all(self.yaw_pitch_roll == value):
            self._yaw_pitch_roll = value
            self.updated_array = set()

    @property
    def mat_rot(self) -> np.ndarray:
        if "mat_rot" not in self.updated_array:
            self._mat_rot = rot_matrix(self.yaw_pitch_roll)
            self.updated_array.update(["mat_rot"])
        return self._mat_rot

    @property
    def rotated_normals(self) -> np.ndarray:
        if "rotated_normals" not in self.updated_array:
            self._rotated_normals = self.mat_rot @ self.base_normals
            self.updated_array.update(["rotated_normals"])
        return self._rotated_normals


class ERP:
    vp_image: np.ndarray  # A RGB image
    vptiles: list

    def __init__(self, tiling: str, proj_res: str, fov: str, vp_shape: np.ndarray = None):
        # Create the projection
        self.shape = np.array(splitx(proj_res)[::-1], dtype=int)
        self.projection = np.zeros(self.shape, dtype='uint8')
        self.proj_coord_nm = np.mgrid[range(self.shape[0]), range(self.shape[1])]
        self.proj_coord_xyz = self.nm2xyz(self.proj_coord_nm, self.shape)

        # Create the viewport
        self.fov = np.deg2rad(splitx(fov)[::-1])
        if vp_shape is None:
            self.vp_shape = np.round(self.fov * self.shape / (pi, 2 * pi)).astype('int')
        else:
            self.vp_shape = vp_shape
        self.viewport = Viewport(self.vp_shape, self.fov)

        # Create the tiling
        self.tiling = np.array(splitx(tiling)[::-1], dtype=int)
        self.n_tiles = self.tiling[0] * self.tiling[1]
        self.tile_res = (self.shape / self.tiling).astype(int)
        self.tile_border_base = get_borders(shape=self.tile_res)
        self.tile_position_list = np.array([(n, m)
                                            for n in range(0, self.shape[0], self.tile_res[0])
                                            for m in range(0, self.shape[1], self.tile_res[1])])
        self.borders_xyz = []
        for tile in range(self.n_tiles):
            borders_nm = self.get_tile_borders(tile)
            self.borders_xyz.append(self.nm2xyz(nm_coord=borders_nm, shape=self.shape))

    def get_vptiles(self, yaw_pitch_roll) -> list[str]:
        """

        :param yaw_pitch_roll: The coordinate of center of VP.
        :return:
        """
        if tuple(self.tiling) == (1, 1): return ['0']

        self.yaw_pitch_roll = yaw_pitch_roll
        self.vptiles = [str(tile) for tile in range(self.n_tiles)
                        if self.viewport.is_viewport(self.borders_xyz[tile])]
        return self.vptiles

    def get_viewport(self, frame: np.ndarray, yaw_pitch_roll: np.ndarray) -> np.ndarray:
        self.yaw_pitch_roll = yaw_pitch_roll
        self.vp_image = self.viewport.get_vp(frame=frame, xyz2nm=self.xyz2nm)
        return self.vp_image

    def get_tile_borders(self, idx: int):
        # projection agnostic
        return self.tile_border_base + self.tile_position_list[idx].reshape(2, -1)

    ##############################################
    def draw_all_tiles_borders(self, lum=255):
        self.clear_projection()
        for tile in range(self.n_tiles):
            self._draw_tile_border(idx=int(tile), lum=lum)
        return self.projection

    def draw_vp_tiles(self, yaw_pitch_roll, lum=255):
        self.clear_projection()
        for tile in self.get_vptiles(yaw_pitch_roll):
            self._draw_tile_border(idx=int(tile), lum=lum)
        return self.projection

    def draw_vp_mask(self, yaw_pitch_roll, lum=255) -> np.ndarray:
        """
        Project the sphere using ERP. Where is Viewport the
        :param yaw_pitch_roll:
        :param lum: value to draw line
        :return: a numpy.ndarray with one deep color
        """
        self.clear_projection()
        self.yaw_pitch_roll = yaw_pitch_roll
        rotated_normals = self.viewport.rotated_normals.T
        inner_product = np.tensordot(rotated_normals, self.proj_coord_xyz, axes=1)
        belong = np.all(inner_product <= 0, axis=0)
        self.projection[belong] = lum
        return self.projection

    def draw_vp_borders(self, yaw_pitch_roll: np.ndarray, lum=255, thickness=1):
        """
        Project the sphere using ERP. Where is Viewport the
        :param yaw_pitch_roll:
        :param lum: value to draw line
        :param thickness: in pixel.
        :return: a numpy.ndarray with one deep color
        """
        self.clear_projection()
        self.yaw_pitch_roll = yaw_pitch_roll

        vp_borders_xyz = self.viewport.get_vp_borders_xyz(thickness=thickness)
        nm_coord = self.xyz2nm(vp_borders_xyz, shape=self.shape, round_nm=True).astype(int)
        self.projection[nm_coord[0, ...], nm_coord[1, ...]] = lum
        return self.projection

    @property
    def yaw_pitch_roll(self):
        return self.viewport.yaw_pitch_roll

    @yaw_pitch_roll.setter
    def yaw_pitch_roll(self, value: np.ndarray):
        self.viewport.yaw_pitch_roll = value

    def _draw_tile_border(self, idx, lum=255):
        borders = self.get_tile_borders(idx)
        n, m = borders
        self.projection[n, m] = lum

    def show(self):
        show1(self.projection)

    def clear_projection(self):
        # self.projection = np.zeros(self.proj_res.shape, dtype='uint8')
        self.projection = np.zeros(self.shape, dtype='uint8')

    @staticmethod
    def nm2xyz(nm_coord: np.ndarray, shape: np.ndarray):
        """
        ERP specific.

        :param nm_coord: shape==(2,...)
        :param shape: (N, M)
        :return:
        """
        azimuth = ((nm_coord[1] + 0.5) / shape[1] - 0.5) * 2 * np.pi
        elevation = ((nm_coord[0] + 0.5) / shape[0] - 0.5) * -np.pi

        z = np.cos(elevation) * np.cos(azimuth)
        y = -np.sin(elevation)
        x = np.cos(elevation) * np.sin(azimuth)

        xyz_coord = np.array([x, y, z])
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

        r = np.sqrt(np.sum(xyz_coord ** 2, axis=0))

        elevation = np.arcsin(xyz_coord[1] / r)
        azimuth = np.arctan2(xyz_coord[0], xyz_coord[2])

        v = elevation / pi + 0.5
        u = azimuth / (2 * pi) + 0.5

        n = v * N - 0.5
        m = u * M - 0.5

        if round_nm:
            n = np.mod(np.round(n), N)
            m = np.mod(np.round(m), M)

        return np.array([n, m])


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


def get_borders(*, coord_nm: Union[tuple, np.ndarray] = None, shape=None, thickness=1):
    """
    frame must be shape==(C, N, M)
    :param coord_nm:
    :param shape:
    :param thickness:
    :return: shape==(C, thickness*(2N+2M))
    """
    if coord_nm is None:
        assert shape is not None
        coord_nm = np.mgrid[range(shape[0]), range(shape[1])]
        C = 2
    else:
        C = coord_nm.shape[0]

    left = coord_nm[:, :, 0:thickness].reshape((C, -1))
    right = coord_nm[:, :, :- 1 - thickness:-1].reshape((C, -1))
    top = coord_nm[:, 0:thickness, :].reshape((C, -1))
    bottom = coord_nm[:, :- 1 - thickness:-1, :].reshape((C, -1))

    return np.c_[top, right, bottom, left]


def main():
    # erp '144x72', '288x144','432x216','576x288'
    # cmp '144x96', '288x192','432x288','576x384'
    yaw_pitch_roll = np.deg2rad((70, 0, 0))
    height, width = 288, 576

    cover_red = Image.new("RGB", (width, height), (255, 0, 0))
    cover_green = Image.new("RGB", (width, height), (0, 255, 0))
    cover_gray = Image.new("RGB", (width, height), (200, 200, 200))
    cover_blue = Image.new("RGB", (width, height), (0, 0, 255))

    ########################################
    # Open Image
    frame_img: Union[Image, list] = Image.open('images/erp1.jpg')
    frame_img = frame_img.resize((width, height))
    frame_array = np.asarray(frame_img)

    erp = ERP('6x4', f'{width}x{height}', '100x90')
    tiles = erp.get_vptiles(yaw_pitch_roll)

    viewport_array = erp.get_viewport(frame_array, yaw_pitch_roll)
    vp_image = Image.fromarray(viewport_array)
    width_vp = int(np.round(height * vp_image.width / vp_image.height))
    vp_image_resized = vp_image.resize((width_vp, height))

    # Get masks
    mask_all_tiles_borders = Image.fromarray(erp.draw_all_tiles_borders())
    mask_vp_tiles = Image.fromarray(erp.draw_vp_tiles(yaw_pitch_roll))
    mask_vp = Image.fromarray(erp.draw_vp_mask(yaw_pitch_roll, lum=200))
    mask_vp_borders = Image.fromarray(erp.draw_vp_borders(yaw_pitch_roll))

    # Composite mask with projection
    frame_img = Image.composite(cover_red, frame_img, mask=mask_all_tiles_borders)
    frame_img = Image.composite(cover_green, frame_img, mask=mask_vp_tiles)
    frame_img = Image.composite(cover_gray, frame_img, mask=mask_vp)
    frame_img = Image.composite(cover_blue, frame_img, mask=mask_vp_borders)

    new_im = Image.new('RGB', (width + width_vp + 2, height), (255, 255, 255))
    new_im.paste(frame_img, (0, 0))
    new_im.paste(vp_image_resized, (width + 2, 0))
    new_im.show()
    print(f'The viewport touch the tiles {tiles}.')


if __name__ == '__main__':
    main()
