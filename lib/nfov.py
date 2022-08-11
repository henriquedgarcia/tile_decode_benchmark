# Copyright 2017 Nitish Mutha (nitishmutha.com)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from math import pi

import cv2
import numpy as np


class NFOV():
    frame: np.ndarray

    def __init__(self, proj_shape, FOV=(110, 90)):
        self.FOV = np.deg2rad(FOV)[::-1]
        self.FOV_norm = self.FOV / (pi, 2*pi)
        self.PI = pi
        self.PI_2 = pi * 0.5
        self.PI2 = pi * 2.0
        self.proj_shape = proj_shape
        self.vp_shape = np.round(self.FOV_norm * proj_shape)
        self.frame_height, self.frame_width = self.proj_shape
        self.vp_height, self.vp_width = self.vp_shape

        xx, yy = np.meshgrid(np.linspace(-pi, pi, self.vp_width, dtype=np.float32),
                             np.linspace(-pi/2, pi/2, self.vp_height, dtype=np.float32))

        self.x = xx * self.FOV_norm[0]
        self.y = yy * self.FOV_norm[1]

        self.rou = np.sqrt(self.x ** 2 + self.y ** 2)
        atan_rou = np.arctan(self.rou)
        self.sin_atan_rou = np.sin(atan_rou)
        self.cos_atan_rou = np.cos(atan_rou)
        self.y_sin_atan_rou = self.y * self.sin_atan_rou
        self.x_sin_atan_rou = self.x * self.sin_atan_rou
        self.rou_cos_atan_rou = self.rou * self.cos_atan_rou

    def spherical2gnomonic(self):
        lat = np.arcsin(self.cos_atan_rou * np.sin(-self.cp[1])
                        + (self.y_sin_atan_rou * np.cos(-self.cp[1]))
                        / self.rou)
        lon = self.cp[0] + np.arctan2(self.x_sin_atan_rou,
                                      self.rou_cos_atan_rou * np.cos(-self.cp[1])
                                      - self.y_sin_atan_rou * np.sin(-self.cp[1]))

        map_x = (lon / self.PI2 + 0.5 )  * self.frame_width
        map_y = (lat / self.PI + 0.5 )  * self.frame_height

        out = cv2.remap(self.frame,
                        map_x,
                        map_y,
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_WRAP)

        # Border
        top = np.round(np.mod([map_y[:10, :], map_x[:10, :]], self.frame_width - 1)).astype(int)
        botton = np.round(np.mod([map_y[-10:, :], map_x[-10:, :]], self.frame_width - 1)).astype(int)
        left = np.round(np.mod([map_y[:, :10], map_x[:, :10]], self.frame_width - 1)).astype(int)
        right = np.round(np.mod([map_y[:, -10:], map_x[:, -10:]], self.frame_width - 1)).astype(int)
        self.frame[top[0], top[1], :] = 0
        self.frame[botton[0], botton[1], :] = 0
        self.frame[left[0], left[1], :] = 0
        self.frame[right[0], right[1], :] = 0
        return out

    def spherical2rectilinear(self, m, n):
        tan_fov_2 = np.tan(self.FOV/2)
        image = np.zeros(self.vp_shape)

        vp_coord_x, vp_coord_y = np.meshgrid(np.linspace(-tan_fov_2[1], tan_fov_2[1], self.vp_width, endpoint=False),
                                             np.linspace(tan_fov_2[0], -tan_fov_2[0], self.vp_height, endpoint=True))
        vp_coord_z = np.zeros(self.vp_shape)
        vp_coord_xyz_ =np.array(vp_coord_x,vp_coord_y,vp_coord_z)

        sqr = np.sum(vp_coord_xyz_ * vp_coord_xyz_, axis=1)
        r = np.sqrt(sqr)

        self.vp_coord_xyz = vp_coord_xyz_ / r

    def get_viewport(self, frame: np.ndarray, center_point: np.ndarray):
        self.frame = frame
        self.frame_height = frame.shape[0]
        self.frame_width = frame.shape[1]
        self.frame_channel = frame.shape[2]

        self.cp = center_point

        return self.spherical2gnomonic()

def main(cp=(0,0), fov=(110,90)):
    # cp = (longitude, latitude) in rad
    # fov = (horizontal, vertical) in rad
    cp = np.deg2rad(cp)
    cp = np.array(cp)  # camera center point (valid range [0,1])

    # img = im.imread('lib/images/360.jpg')
    # img = im.imread("lib\\images\\erp1.jpg")
    img = im.imread("lib\\images\\erp_mili.jpg")

    height, width,_ = img.shape

    start = time.time()
    result = NFOV(proj_shape=(height, width), FOV=fov)
    result = result.get_viewport(img, cp)
    print(f'total time {time.time() - start} s')
    print(f'proj shape "{img.shape}"')
    print(f'vp shape "{result.shape}"')

    plt.imshow(result)
    # plt.imshow(img)
    plt.show()

# test the class
if __name__ == '__main__':
    import imageio as im
    import matplotlib.pyplot as plt
    import time

    main()