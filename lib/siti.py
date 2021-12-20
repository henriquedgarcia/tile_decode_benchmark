import collections
import json
from logging import info, debug, warning
from typing import List, Union, Tuple

import matplotlib.axes as axes
import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skvideo.io

from lib.util import splitx, sobel
from lib.video_state import VideoContext


class SiTi:
    def __init__(self, state: VideoContext):
        info('Stating SiTi')
        self.state = state
        self.filename = state.compressed_file
        self.frame = state.frame
        self.folder = state.siti_folder

        self.si = []
        self.ti = []
        self.previous_frame = None
        self.stats = dict(si_0q=[], si_1q=[], si_2q=[], si_3q=[], si_4q=[],
                          si_average=[], si_std=[],
                          ti_0q=[], ti_1q=[], ti_2q=[], ti_3q=[], ti_4q=[],
                          ti_average=[], ti_std=[])

        self.fig: Union[figure.Figure, None] = None
        self.ax: Union[List[List[axes.Axes]], None] = None
        self.writer = skvideo.io.FFmpegWriter(f'{state.siti_movie}')

    @staticmethod
    def _calc_si(frame: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Calcule Spatial Information for a video frame.
        :param frame: A luma video frame in numpy ndarray format.
        :return: spatial information and sobel frame.
        """
        sob = sobel(frame)
        si = sob.std()
        return si, sob

    def _calc_ti(self, frame: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Calcule Temporal Information for a video frame. If is a first frame,
        the information is zero.
        :param frame: A luma video frame in numpy ndarray format.
        :return: Temporal information and diference frame. If first frame the
        diference is zero array on same shape of frame.
        """
        if self.previous_frame is not None:
            difference = frame - self.previous_frame
            ti = difference.std()
        else:
            difference = np.zeros(frame.shape)
            ti = 0.0

        self.previous_frame = frame
        return ti, difference

    def run(self, animate_graph=False, overwrite=False, save=True):
        info('Running SiTi')
        movie_exist = self.state.siti_movie.exists()

        if self.state.siti_results.exists() and not overwrite:
            warning(f'The file {self.state.siti_results} exist. Skipping.')
            df = pd.read_csv(f'{self.state.siti_results}', index_col=0)
            self.si = df['si'].tolist()
            self.ti = df['ti'].tolist()
            self.save_stats(overwrite=overwrite)
            return

        if not self.filename.exists():
            warning(f'The file {self.filename} not exist. Skipping.')
            return

        vreader = skvideo.io.vreader(fname=str(self.filename), as_grey=True)

        for frame_counter, frame in enumerate(vreader, 1):
            info(f'Calculating metrics. Frame {frame_counter}.')
            width = frame.shape[1]
            height = frame.shape[2]
            frame = frame.reshape((width, height)).astype('float32')

            value_si, sob = self._calc_si(frame)
            value_ti, difference = self._calc_ti(frame)
            self.si.append(value_si)
            self.ti.append(value_ti)
            debug(f"{self.state.name} - {frame_counter:05}, si={value_si:05.3f}, ti={value_ti:05.3f}")

            if not animate_graph or (movie_exist and not overwrite):
                continue
            info('Animate Graph')

            debug(f'Add frame {frame_counter:05} to siti video.')
            data = self.make_frame_graph(frame_counter, frame, sob, difference)
            self.writer.writeFrame(data)

        if animate_graph and (not movie_exist or overwrite):
            self.writer.close()

        if save:
            self.save_siti(overwrite=overwrite)
            self.save_stats(overwrite=overwrite)

    def make_frame_graph(self, frame_counter, frame, sob, difference):
        plt.close()
        fig, ax = plt.subplots(2, 2, figsize=(12, 6))
        fig.tight_layout()

        '''Show frame'''
        debug('Making main frame.')
        ax[0][0].imshow(frame, cmap='gray')
        ax[0][0].set_xlabel('Frame luma')
        ax[0][0].get_xaxis().set_ticks([])
        ax[0][0].get_yaxis().set_ticks([])

        '''Show sobel'''
        debug('Making sobel frame.')
        ax[1][0].imshow(sob, cmap='gray')
        ax[1][0].set_xlabel('Sobel result')
        ax[1][0].get_xaxis().set_ticks([])
        ax[1][0].get_yaxis().set_ticks([])

        '''Show difference'''
        debug('Making differences frame.')
        if self.previous_frame is not None:
            ax[1][1].imshow(np.abs(difference), cmap='gray')
        ax[1][1].set_xlabel('Diff result')
        ax[1][1].get_xaxis().set_ticks([])
        ax[1][1].get_yaxis().set_ticks([])

        '''A moving si/ti graph'''
        debug('Making a queue.')
        samples = 300
        val_si = self.si
        val_ti = self.ti
        rotation = -samples
        if len(val_si) < samples:
            val_si = val_si + [0] * (samples - len(val_si))
            val_ti = val_ti + [0] * (samples - len(val_ti))
            rotation = -frame_counter
        v_si = collections.deque(val_si[-samples:])
        v_ti = collections.deque(val_ti[-samples:])
        v_si.rotate(rotation)
        v_ti.rotate(rotation)

        debug('Making moving queue.')
        ax[0][1].plot(v_si, 'b', label=f'SI={self.si[-1]:05.3f}')
        ax[0][1].plot(v_ti, 'r', label=f'TI={self.ti[-1]:05.3f}')
        ax[0][1].set_xlabel('SI/TI')
        ax[0][1].legend(loc='upper left', bbox_to_anchor=(1.01, 0.99))
        ax[0][1].set_ylim(bottom=0)
        ax[0][1].set_xlim(left=0)

        '''Saving'''
        # path = self.folder / self.state.video.name
        # path.mkdir(parents=True, exist_ok=True)
        # frame_filename = path / f'frame_{frame_counter}.jpg'
        # plt.savefig(str(frame_filename), dpi=150)
        # plt.show()

        debug('Convert Matplotlib to Numpy and save.')
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data

    def save_siti(self, overwrite=False):
        if not self.state.siti_results.exists() or overwrite:
            df = pd.DataFrame({'si': self.si, 'ti': self.ti})
            df.to_csv(f'{self.state.siti_results}', index_label='frame')

    def save_stats(self, overwrite=False):
        si = self.si
        ti = self.ti
        stats = {'si_average': f'{np.average(si):05.3f}',
                 'si_std': f'{np.std(si):05.3f}',
                 'si_0q': f'{np.quantile(si, 0.00):05.3f}',
                 'si_1q': f'{np.quantile(si, 0.25):05.3f}',
                 'si_2q': f'{np.quantile(si, 0.50):05.3f}',
                 'si_3q': f'{np.quantile(si, 0.75):05.3f}',
                 'si_4q': f'{np.quantile(si, 1.00):05.3f}',
                 'ti_average': f'{np.average(ti):05.3f}',
                 'ti_std': f'{np.std(ti):05.3f}',
                 'ti_0q': f'{np.quantile(ti, 0.00):05.3f}',
                 'ti_1q': f'{np.quantile(ti, 0.25):05.3f}',
                 'ti_2q': f'{np.quantile(ti, 0.50):05.3f}',
                 'ti_3q': f'{np.quantile(ti, 0.75):05.3f}',
                 'ti_4q': f'{np.quantile(ti, 1.00):05.3f}'}
        print(f'{self.state.name}:')
        print(json.dumps(stats, indent=2))

        if self.state.siti_stats.exists() and not overwrite:
            return
        json_stats = json.dumps(stats, separators=(',', ':'))
        self.state.siti_stats.write_text(json_stats, encoding='utf-8')
