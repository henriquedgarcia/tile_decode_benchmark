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
import numpy as np

class NFOV():
    def __init__(self, frame_width, frame_height, FOV = (110/180*pi, 90/180*pi)):
        self.FOV = FOV / np.array([2*pi, pi])
        self.PI = pi
        self.PI_2 = pi * 0.5
        self.PI2 = pi * 2.0
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.width = round(self.FOV[0] * frame_width)
        self.height = round(self.FOV[1] * frame_height)
        self.screen_points = self._get_screen_img()
        self.sphericalScreenCoord = ((self.screen_points * 2 - 1)
                                     * np.array([self.PI, self.PI_2])
                                     * (np.ones(self.screen_points.shape) * self.FOV))

        self.x = self.sphericalScreenCoord.T[0]
        self.y = self.sphericalScreenCoord.T[1]

        self.rou = np.sqrt(self.x ** 2 + self.y ** 2)
        atan_rou = np.arctan(self.rou)
        self.sin_c = np.sin(atan_rou)
        self.cos_c = np.cos(atan_rou)

    def _get_coord_rad(self, isCenterPt, center_point=None):
        return (center_point * 2 - 1) * np.array([self.PI, self.PI_2]) \
            if isCenterPt \
            else \
            (self.screen_points * 2 - 1) * np.array([self.PI, self.PI_2]) * (
                np.ones(self.screen_points.shape) * self.FOV)

    def _get_screen_img(self):
        xx, yy = np.meshgrid(np.linspace(0, 1, self.width), np.linspace(0, 1, self.height))
        return np.array([xx.ravel(), yy.ravel()]).T

    def _calcSphericaltoGnomonic(self, convertedScreenCoord):
        x = convertedScreenCoord.T[0]
        y = convertedScreenCoord.T[1]

        # rou = np.sqrt(x ** 2 + y ** 2)
        # c = np.arctan(rou)
        # sin_c = np.sin(c)
        # cos_c = np.cos(c)

        lat = np.arcsin(self.cos_c * np.sin(-self.cp[1]) + (y * self.sin_c * np.cos(-self.cp[1])) / self.rou)
        lon = self.cp[0] + np.arctan2(x * self.sin_c, self.rou * np.cos(-self.cp[1]) * self.cos_c - y * np.sin(-self.cp[1]) * self.sin_c)

        lat = (lat / self.PI_2 + 1.) * 0.5
        lon = (lon / self.PI + 1.) * 0.5

        return np.array([lon, lat]).T

    def _bilinear_interpolation(self, screen_coord):
        uf = np.mod(screen_coord.T[0],1) * self.frame_width  # long - width
        vf = np.mod(screen_coord.T[1],1) * self.frame_height  # lat - height

        x0 = np.floor(uf).astype(int)  # coord of pixel to bottom left
        y0 = np.floor(vf).astype(int)
        x2 = np.add(x0, np.ones(uf.shape).astype(int))  # coords of pixel to top right
        y2 = np.add(y0, np.ones(vf.shape).astype(int))

        base_y0 = np.multiply(y0, self.frame_width)
        base_y2 = np.multiply(y2, self.frame_width)

        A_idx = np.add(base_y0, x0)
        B_idx = np.add(base_y2, x0)
        C_idx = np.add(base_y0, x2)
        D_idx = np.add(base_y2, x2)

        flat_img = np.reshape(self.frame, [-1, self.frame_channel])

        A = np.take(flat_img, A_idx, axis=0)
        B_idx=np.array([item if item < len(flat_img) else len(flat_img)-1 for item in B_idx])
        B = np.take(flat_img, B_idx, axis=0)
        C = np.take(flat_img, C_idx, axis=0)
        D_idx=np.array([item if item < len(flat_img) else len(flat_img)-1 for item in D_idx])
        D = np.take(flat_img, D_idx, axis=0)

        wa = np.multiply(x2 - uf, y2 - vf)
        wb = np.multiply(x2 - uf, vf - y0)
        wc = np.multiply(uf - x0, y2 - vf)
        wd = np.multiply(uf - x0, vf - y0)

        # interpolate
        AA = np.multiply(A, np.array([wa, wa, wa]).T)
        BB = np.multiply(B, np.array([wb, wb, wb]).T)
        CC = np.multiply(C, np.array([wc, wc, wc]).T)
        DD = np.multiply(D, np.array([wd, wd, wd]).T)
        nfov = np.reshape(np.round(AA + BB + CC + DD).astype(np.uint8), [self.height, self.width, 3])

        return nfov

    def toNFOV(self, frame: np.ndarray, center_point: np.ndarray, frame_width=None, frame_height=None):
        self.frame = frame
        self.frame_height = frame.shape[0]
        self.frame_width = frame.shape[1]
        self.frame_channel = frame.shape[2]

        self.cp = center_point

        gnomonicCoord = self._calcSphericaltoGnomonic(self.sphericalScreenCoord)
        return self._bilinear_interpolation(gnomonicCoord)


# test the class
if __name__ == '__main__':
    import imageio as im
    import matplotlib.pyplot as plt
    import time

    center_point = np.array([0,pi/2])  # camera center point (valid range [0,1])

    img = im.imread('images/360.jpg')
    height, width,_ = img.shape
    result = NFOV(frame_width=width, frame_height=height)

    start = time.time()
    result = result.toNFOV(img, center_point)
    print(f'total time {time.time() - start} s')

    plt.imshow(result)
    # plt.imshow(img)
    plt.show()
