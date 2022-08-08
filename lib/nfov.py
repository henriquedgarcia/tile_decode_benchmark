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
import cv2
class NFOV():
    frame: np.ndarray

    def __init__(self, proj_width, proj_height, FOV=(110, 90)):
        self.FOV_norm = FOV / np.array([360, 180])
        self.PI = pi
        self.PI_2 = pi * 0.5
        self.PI2 = pi * 2.0
        self.frame_width = proj_width
        self.frame_height = proj_height
        self.vp_width = round(self.FOV_norm[0] * proj_width)
        self.vp_height = round(self.FOV_norm[1] * proj_height)

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

        top = np.round(np.mod([map_y[:10, :], map_x[:10, :]], self.frame_width - 1)).astype(int)
        botton = np.round(np.mod([map_y[-10:, :], map_x[-10:, :]], self.frame_width - 1)).astype(int)
        left = np.round(np.mod([map_y[:, :10], map_x[:, :10]], self.frame_width - 1)).astype(int)
        right = np.round(np.mod([map_y[:, -10:], map_x[:, -10:]], self.frame_width - 1)).astype(int)
        self.frame[top[0], top[1], :] = 0
        self.frame[botton[0], botton[1], :] = 0
        self.frame[left[0], left[1], :] = 0
        self.frame[right[0], right[1], :] = 0
        return out

    def get_viewport(self, frame: np.ndarray, center_point: np.ndarray):
        self.frame = frame
        self.frame_height = frame.shape[0]
        self.frame_width = frame.shape[1]
        self.frame_channel = frame.shape[2]

        self.cp = center_point

        return self.spherical2gnomonic()


# test the class
if __name__ == '__main__':
    import imageio as im
    import matplotlib.pyplot as plt
    import time

    center_point = np.array([0,pi/2])  # camera center point (valid range [0,1])

    img = im.imread('images/360.jpg')
    height, width,_ = img.shape
    result = NFOV(proj_width=width, proj_height=height)

    start = time.time()
    result = result.get_viewport(img, center_point)
    print(f'total time {time.time() - start} s')

    plt.imshow(result)
    # plt.imshow(img)
    plt.show()
