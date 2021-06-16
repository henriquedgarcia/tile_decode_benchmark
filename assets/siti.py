import collections
import json
import os
import subprocess
from typing import List, Union

import matplotlib.axes as axes
import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skvideo.io

from assets.util import splitx, sobel

class SiTi:
    def __init__(self, filename, scale, plot_siti=False, folder='siti'):
        self.filename = filename
        self.scale = scale
        self.width, self.height = splitx(scale)

        self.si = []
        self.ti = []
        self.previous_frame = None
        self.frame_counter = 0
        self.stats = dict(si_0q=[], si_1q=[], si_2q=[], si_3q=[], si_4q=[],
                          si_average=[], si_std=[],
                          ti_0q=[], ti_1q=[], ti_2q=[], ti_3q=[], ti_4q=[],
                          ti_average=[], ti_std=[])

        os.makedirs(folder, exist_ok=True)
        self.jump_siti = False
        self.folder = folder
        self.plot_siti = plot_siti
        self.fig: Union[figure.Figure, None] = None
        self.ax: Union[List[List[axes.Axes]], None] = None

    def _calc_si(self, frame: np.ndarray) -> (float, np.ndarray):
        """
        Calcule Spatial Information for a video frame.
        :param frame: A luma video frame in numpy ndarray format.
        :return: spatial information and sobel frame.
        """
        sob = sobel(frame)
        si = sob.std()
        self.si.append(si)
        return si, sob

    def _calc_ti(self, frame: np.ndarray) -> (float, np.ndarray):
        """
        Calcule Temporal Information for a video frame. If is a first frame,
        the information is zero.
        :param frame: A luma video frame in numpy ndarray format.
        :return: Temporal information and diference frame. If first frame the
        diference is zero array on same shape of frame.
        """
        if self.previous_frame:
            difference = frame - self.previous_frame
            ti = difference.std()
        else:
            difference = np.zeros(frame.shape)
            ti = 0.0

        self.ti.append(ti)
        self.previous_frame = frame

        return ti, difference

    def calc_siti(self, verbose=False):
        vreader = skvideo.io.vreader(fname=self.filename, as_grey=True)
        jump_debug = True if os.path.isfile(
                f'{self.folder}/siti.mp4') else False
        self.jump_siti = True if os.path.isfile(
                f'{self.folder}/siti.csv') else False
        for self.frame_counter, frame in enumerate(vreader, 1):
            if self.jump_siti: break
            width = frame.shape[1]
            height = frame.shape[2]
            frame = frame.reshape((width, height)).astype('float32')
            value_si, sobel = self._calc_si(frame)
            value_ti, difference = self._calc_ti(frame)
            if verbose:
                print(f"{self.frame_counter:04}, "
                      f"si={value_si:05.3f}, ti={value_ti:05.3f}")
            else:
                print('.', end='', flush=True)

            '''For Debug'''
            if self.plot_siti and not jump_debug:
                plt.close()
                self.fig, self.ax = plt.subplots(2, 2, figsize=(12, 6))
                self.fig.tight_layout()

                '''Show frame'''
                self.ax[0][0].imshow(frame, cmap='gray')
                self.ax[0][0].set_xlabel('Frame luma')
                self.ax[0][0].get_xaxis().set_ticks([])
                self.ax[0][0].get_yaxis().set_ticks([])

                '''Show sobel'''
                self.ax[1][0].imshow(sobel, cmap='gray')
                self.ax[1][0].set_xlabel('Sobel result')
                self.ax[1][0].get_xaxis().set_ticks([])
                self.ax[1][0].get_yaxis().set_ticks([])

                '''Show difference'''
                if self.previous_frame is not None:
                    self.ax[1][1].imshow(np.abs(difference), cmap='gray')
                self.ax[1][1].set_xlabel('Diff result')
                self.ax[1][1].get_xaxis().set_ticks([])
                self.ax[1][1].get_yaxis().set_ticks([])

                '''Show a moving si/ti graph'''
                samples = 300
                val_si = self.si
                val_ti = self.ti
                rotation = -samples
                if len(self.si) < samples:
                    val_si = self.si + [0] * (samples - len(self.si))
                    val_ti = val_ti + [0] * (samples - len(self.ti))
                    rotation = -self.frame_counter
                v_si = collections.deque(val_si[-samples:])
                v_ti = collections.deque(val_ti[-samples:])
                v_si.rotate(rotation)
                v_ti.rotate(rotation)

                self.ax[0][1].plot(v_si, 'b', label=f'SI={value_si:05.3f}')
                self.ax[0][1].plot(v_ti, 'r', label=f'TI={value_ti:05.3f}')
                self.ax[0][1].set_xlabel('SI/TI')
                self.ax[0][1].legend(loc='upper left',
                                     bbox_to_anchor=(1.01, 0.99))
                self.ax[0][1].set_ylim(bottom=0)
                self.ax[0][1].set_xlim(left=0)

                '''Saving'''
                plt.savefig(f'{self.folder}/frame_{self.frame_counter}.jpg',
                            dpi=150)
                # plt.show()
        if self.plot_siti and not jump_debug:
            subprocess.run(f'ffmpeg '
                           f'-y -r 30 '
                           f'-i {self.folder}/frame_%d.jpg '
                           f'-c:v libx264 '
                           f'-vf fps=30 '
                           f'-pix_fmt yuv420p '
                           f'{self.folder}/siti.mp4',
                           shell=True, encoding='utf-8')

    def save_siti(self, overwrite=False):
        if self.jump_siti and not overwrite:
            return
        df = pd.DataFrame({'si': self.si, 'ti': self.ti})
        df.to_csv(f'{self.folder}/siti.csv', index_label='frame')

    def save_stats(self, overwrite=False):
        if self.jump_siti and not overwrite:
            return
        si = self.si
        ti = self.ti
        stats = dict(si_average=f'{np.average(si):05.3f}',
                     si_std=f'{np.std(si):05.3f}',
                     si_0q=f'{np.quantile(si, 0.00):05.3f}',
                     si_1q=f'{np.quantile(si, 0.25):05.3f}',
                     si_2q=f'{np.quantile(si, 0.50):05.3f}',
                     si_3q=f'{np.quantile(si, 0.75):05.3f}',
                     si_4q=f'{np.quantile(si, 1.00):05.3f}',
                     ti_average=f'{np.average(ti):05.3f}',
                     ti_std=f'{np.std(ti):05.3f}',
                     ti_0q=f'{np.quantile(ti, 0.00):05.3f}',
                     ti_1q=f'{np.quantile(ti, 0.25):05.3f}',
                     ti_2q=f'{np.quantile(ti, 0.50):05.3f}',
                     ti_3q=f'{np.quantile(ti, 0.75):05.3f}',
                     ti_4q=f'{np.quantile(ti, 1.00):05.3f}')

        with open(f'{self.folder}/stats.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f, separators=(',', ':'))
