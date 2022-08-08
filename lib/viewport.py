import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from .transform import rot_matrix, cart2hcs
from .util import splitx


class Viewport:
    position: tuple
    projection: np.ndarray
    mat: np.ndarray
    resolution: np.ndarray
    base_normals: np.ndarray
    rotated_normals: np.ndarray

    def __init__(self, resolution: np.ndarray, fov: np.ndarray, proj: str = 'erp'):
        """ self.tiling, self.fov, 'erp'
        Viewport Class used to extract view pixels in projections.
        :param frame resolution: (600, 800) for 800x600px
        :param fov: in rad. Ex: "np.array((pi/2, pi/2))" for (90°x90°)
        :param proj: 'erp' is default
        """
        # self.resolution = Res(resolution)
        # self.fov = Res(fov)
        self.frame_shape = resolution
        self.fov = fov
        self.proj = proj
        self._make_base_normals()
        # self.frame_borders = self.get_borders(self.frame_shape)  # [(y,x),...]

        # self.y = self.frame_borders.T[0]
        # self.x = self.frame_borders.T[1]
        #
        # self.rou = np.sqrt(self.x ** 2 + self.y ** 2)
        # atan_rou = np.arctan(self.rou)
        # self.sin_atan_rou = np.sin(atan_rou)
        # self.cos_atan_rou = np.cos(atan_rou)
        # self.y_sin_atan_rou = self.y * self.sin_atan_rou
        # self.x_sin_atan_rou = self.x * self.sin_atan_rou
        # self.rou_cos_atan_rou = self.rou * self.cos_atan_rou

    def _make_base_normals(self):
        """
        com eixo entrando no observador, rotação horário é negativo e anti-horária
        é positivo. Todos os ângulos são radianos.

        O eixo z aponta pra frente
        O exito x aponta pra direita
        O eixo y aponta pra baixo

        deslocamento pra direita e para cima é positivo.

        o viewport é a região da esfera que faz intersecção com 4 planos que passam pelo
          centro (4 grandes circulos): cima, direita, baixo e esquerda.
        Os planos são definidos tal que suas normais (N) parte do centro e apontam na mesma direção a
          região do viewport. ex: O plano de cima aponta para cima, etc.
        Todos os pixels que estiverem abaixo do plano {N(x,y,z) dot P(x,y,z) <= 0}
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
        fov_y, fov_x = self.fov / (2, 2)
        pi_2 = np.pi * 0.5

        self.base_normals = np.array([[0, -np.sin(fov_y + pi_2), np.cos(fov_y + pi_2)],  # top
                                      [0, -np.sin(-fov_y - pi_2), np.cos(-fov_y - pi_2)],  # botton
                                      [np.sin(fov_x + pi_2), 0, np.cos(fov_x + pi_2)],  # left
                                      [np.sin(-fov_x - pi_2), 0, np.cos(-fov_x - pi_2)]])  # right

    def rotate(self, yaw, pitch, roll):
        """
        Set a new position to viewport using aerospace's body coordinate system
        and make the projection. Rotate the normal planes of viewport using matrix of rotation and Tait–Bryan
        angles in Z-Y-X order. Refer to Wikipedia.

        :param yaw: the positions like (yaw, pitch, roll) in rad
        :param pitch: the positions like (yaw, pitch, roll) in rad
        :param roll: the positions like (yaw, pitch, roll) in rad
        :return: self
        """
        # self.position = yaw, pitch, roll
        # self.mat = rot_matrix(yaw, pitch, roll)
        # self.rotated_normals = []

        # For each plane in view
        # self.rotated_normals = [self.mat @ normal for normal in self.base_normals]
        self.cp = yaw, pitch
        self.mat = rot_matrix((yaw, pitch, roll))
        self.rotated_normals = (self.mat @ self.base_normals.T).T

    # def is_viewport(self, x, y, z) -> bool:
    def is_viewport(self, x_y_z) -> bool:
        """
        Check if the plane equation is true to viewport
        x1 * m + y1 * n + z1 * z < 0
        If True, the "point" is on the viewport
        Obs: is_in só retorna true se todas as expressões forem verdadeiras

        :param x: A 3D Point in the space (x, y, z).
        :param y: A 3D Point in the space (x, y, z).
        :param z: A 3D Point in the space (x, y, z).
        :return: A boolean         belong = np.all(inner_product <= 0, axis=0).reshape(self.shape)

        """
        inner_prod = np.dot(self.rotated_normals, x_y_z.T)
        px_in_vp = np.all(inner_prod <= 0, axis=0)
        return np.any(px_in_vp)

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

    @staticmethod
    def get_borders(shape):
        border = [[0, x] for x in range(shape[1])]
        border += [[y, 0] for y in range(shape[0])]
        border += [[shape[0] - 1, x] for x in range(shape[1])]
        border += [[y, shape[1] - 1] for y in range(shape[0])]

        border = np.array(border)

        # border = np.array([((0, x), (y, shape[1] - 1),
        #                     (shape[0] - 1, shape[1] - 1 - x), (shape[0] - 1 - y, 0))
        #                    for y, x in zip(range(shape[0]), range(shape[1]))])
        return border.reshape(-1, 2)

    def get_vp_borders(self, shape):
        shape = self.frame_shape

        hfovy, hfovx = np.tan(self.fov / 2)
        x_y_z = np.ones((shape[0] * shape[1], 3))

        border = self.get_borders()

        v_u = (border + 0.5) * 2 * (np.tan(hfovy) / shape[0], np.tan(hfovx) / shape[1])
        y_z = -v_u + (np.tan(hfovy), np.tan(hfovx))
        x_y_z[:, 1:] = y_z
        r = np.sqrt(np.sum(x_y_z ** 2, axis=1))
        x_y_z_norm = x_y_z / r.reshape(shape[0] * shape[1], 1)

        r = np.ones(len(x_y_z_norm))
        azimuth = np.arctan2(x_y_z_norm[:, 2], x_y_z_norm[:, 0])
        elevation = np.arcsin(x_y_z_norm[:, 1])

        if np.any(azimuth >= np.pi) or np.any(azimuth < -np.pi):
            azimuth = (azimuth + np.pi) % (2 * np.pi)
            azimuth = azimuth - np.pi

        if np.any(elevation > np.pi / 2):
            elevation = 2 * np.pi - elevation
        elif np.any(elevation < -np.pi / 2):
            elevation = -2 * np.pi - elevation

        u = azimuth / (2 * np.pi) + 0.5
        v = -elevation / np.pi + 0.5
        m = np.round(u * shape[1] - 0.5).astype(int)
        n = np.round(v * shape[0] - 0.5).astype(int)
        m_n = np.unique(np.array([m, n]).T, axis=0)

        return m_n

        # mn -> uv -> xyz -> hcs
        # using the projection...
        # projection dependent

        # H, W = self.resolution
        # x_i = 0  # first row
        # x_f = W  # last row
        # y_i = 0  # first line
        # y_f = H  # last line
        #
        # xi_xf = range(W)
        # yi_yf = range(H)
        # for x0 in xi_xf:
        #     n, m = y_i, x0
        #     # x, y, z = vp2cart(m, n, (H, W), self.fov.shape)
        #     x, y, z = vp2cart(m, n, (H, W), self.fov)
        #     x, y, z = self.mat @ (x, y, z)
        #     yield (m, n), (x, y, z)
        # for x0 in xi_xf:
        #     n, m = y_f - 1, x0
        #     # x, y, z = vp2cart(m, n, (H, W), self.fov.shape)
        #     x, y, z = vp2cart(m, n, (H, W), self.fov)
        #     x, y, z = self.mat @ (x, y, z)
        #     yield (m, n), (x, y, z)
        # for y0 in yi_yf:
        #     n, m = y0, x_i
        #     # x, y, z = vp2cart(m, n, (H, W), self.fov.shape)
        #     x, y, z = vp2cart(m, n, (H, W), self.fov)
        #     x, y, z = self.mat @ (x, y, z)
        #     yield (m, n), (x, y, z)
        # for y0 in yi_yf:
        #     n, m = y0, x_f - 1
        #     # x, y, z = vp2cart(m, n, (H, W), self.fov)
        #     x, y, z = vp2cart(m, n, (H, W), self.fov)
        #     x, y, z = self.mat @ (x, y, z)
        #     yield (m, n), (x, y, z)

    def show(self):
        frame_img = Image.fromarray(self.projection)
        frame_img.show()


class ERP:
    n_tiles: int
    projection: np.ndarray

    # def __init__(self, tiling: Union[str, Res], proj_res: Union[str, Res], fov: Union[str, Res]):
    # self.tiling = Res(tiling)
    # self.proj_res = Res(proj_res)
    # self.fov = Res(fov)
    # self.viewport = Viewport(Res(proj_res), fov, 'erp')
    # self.tile_res = self.proj_res / self.tiling

    def __init__(self, tiling: str, proj_res: str, fov: str):
        self.tiling = np.array(splitx(tiling)[::-1], dtype=int)
        self.shape = np.array(splitx(proj_res)[::-1], dtype=int)
        self.fov = np.deg2rad(splitx(fov)[::-1])
        self.viewport = Viewport(self.shape, self.fov, 'erp')
        self.tile_res = (self.shape / self.tiling).astype(int)
        self.n_tiles = self.tiling[0] * self.tiling[1]

        self.clear_image()

        n_m_coord = np.array([[n, m] for n in range(self.shape[0]) for m in range(self.shape[1])])

        self.proj_coord = self.nm2xyv(n_m_coord, self.shape)

        self.tiles_position = np.array([(n, m)
                                        for n in range(0, self.shape[0], self.tile_res[0])
                                        for m in range(0, self.shape[1], self.tile_res[1])])

        self.border_base = Viewport.get_borders(self.tile_res)

    def clear_image(self):
        # self.projection = np.zeros(self.proj_res.shape, dtype='uint8')
        self.projection = np.zeros(self.shape, dtype='uint8')

    @staticmethod
    def nm2xyv(n_m_coord, shape):
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

    def rotate(self, yaw, pitch, roll):
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
        # m, n = self.tiles_position[idx]
        # H, W = self.tile_res.shape
        #
        # x_i = m  # first row
        # x_f = m + W  # last row
        # y_i = n  # first line
        # y_f = n + H  # last line
        #
        # xi_xf = range(int(x_i), int(x_f))
        # yi_yf = range(int(y_i), int(y_f))
        #
        # for x0 in xi_xf:
        #     n, m = y_i, x0
        #     x, y, z = erp2cart(n, m, self.proj_res.shape)
        #     yield (m, n), (x, y, z)
        # for x0 in xi_xf:
        #     n, m = y_f - 1, x0
        #     x, y, z = erp2cart(n, m, self.proj_res.shape)
        #     yield (m, n), (x, y, z)
        # for y0 in yi_yf:
        #     n, m = y0, x_i
        #     x, y, z = erp2cart(n, m, self.proj_res.shape)
        #     yield (m, n), (x, y, z)
        # for y0 in yi_yf:
        #     n, m = y0, x_f - 1
        #     x, y, z = erp2cart(n, m, self.proj_res.shape)
        #     yield (m, n), (x, y, z)
        border_n_m = self.border_base + self.tiles_position[idx]
        return border_n_m

    def get_vptiles(self) -> list:
        if f'{self.tiling}' == '1x1':
            return [0]

        tiles = []
        for tile in range(self.n_tiles):
            border_n_m = self.get_tile_borders(tile)
            border_xyz = self.nm2xyv(border_n_m, self.shape)
            if self.viewport.is_viewport(border_xyz):
                tiles.append(tile)
        return tiles

    def draw_vp(self, lum=255):
        """
        Project the sphere using ERP. Where is Viewport the
        :param lum: value to draw line
        :return: a numpy.ndarray with one deep color
        """
        inner_product = self.viewport.rotated_normals @ self.proj_coord.T
        belong = np.all(inner_product <= 0, axis=0).reshape(self.shape)
        self.projection[belong] = lum

    def draw_all_tiles_borders(self, lum=100):
        for tile in range(self.n_tiles):
            self.draw_tile_border(tile, lum)

    def draw_vp_tiles(self, lum=100):
        for tile in self.get_vptiles():
            self.draw_tile_border(idx=tile, lum=lum)

    def draw_tile_border(self, idx, lum=100):
        borders = self.get_tile_borders(idx)
        n, m = borders.T
        self.projection[n, m] = lum

    def draw_vp_borders(self, lum=100):
        """
        Project the sphere using ERP. Where is Viewport the
        :param lum: value to draw line
        :return: a numpy.ndarray with one deep color
        """
        # H, W = self.proj_res.shape
        vp_borders = self.viewport.get_vp_borders()

        # self.projection[n, m] = lum

    def show(self):
        show1(self.projection)


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
        f = 0
        elevation, azimuth = cart2hcs(x, y, z)
        u = azimuth / (2 * np.pi) + 0.5
        v = -elevation / np.pi + 0.5
        shape = self.shape
        f, m, n = 0, 0, 0
        return f, m, n


def show1(projection):
    plt.imshow(projection)
    plt.show()


def show2(projection):
    frame_img = Image.fromarray(projection)
    frame_img.show()


if __name__ == '__main__':
    # erp '144x72', '288x144','432x216','576x288'
    # cmp '144x96', '288x192','432x288','576x384'

    erp = ERP('6x4', '576x288', '110x90')
    height, width = erp.shape
    frame_img = Image.new("RGB", (width, height), (0, 0, 0))

    yaw, pitch, roll = -30, 20, -10
    yaw, pitch, roll = np.deg2rad((yaw, pitch, roll))
    erp.rotate(yaw, pitch, roll)

    tiles = erp.get_vptiles()

    # # Draw all tiles border
    erp.clear_image()
    erp.draw_all_tiles_borders(lum=200)
    cover = Image.new("RGB", (width, height), (255, 0, 0))
    frame_img = Image.composite(cover, frame_img, mask=Image.fromarray(erp.projection))

    # Draw tiles in viewport
    erp.clear_image()
    erp.draw_vp_tiles(lum=255)
    cover = Image.new("RGB", (width, height), (0, 255, 0))
    frame_img = Image.composite(cover, frame_img, mask=Image.fromarray(erp.projection))
    # frame_img.show()

    # Draw viewport borders
    erp.clear_image()
    erp.draw_vp(lum=255)
    cover = Image.new("RGB", (width, height), (200, 200, 200))
    frame_img = Image.composite(cover, frame_img, mask=Image.fromarray(erp.projection))
    frame_img.show()
