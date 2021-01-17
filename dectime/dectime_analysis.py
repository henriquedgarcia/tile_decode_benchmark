import pickle
from enum import Enum
from cycler import cycler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import matplotlib.figure as figure
from typing import Union, List
import os
import pandas as pd
from dectime.video_state import Pattern, Tile, Video, Config, VideoState
# from dectime.util import AutoDict
import json
import fitter.main


class DectimeHandler:
    """Classe responsável por gerenciar os dados brutos e processados,
    assim como calcular as estatísticas.
    - Esta classe é capaz de minerar os dados, armazenar em um AutoDict
    e salvar como um Json.
    - Depois será capaz de ler o Json, selecionar só os dados pertinentes
    e calcular as estatísticas necessárias."""

    # Struct statistics results
    percentile = {0: 0.0, 25: 0.0, 50: 0.0, 75: 0.0, 100: 0.0}

    stats = {'average_t': 0.0, 'std_t': 0.0, 'percentile_t': percentile.copy(),
             'average_r': 0.0, 'std_r': 0.0, 'percentile_r': percentile.copy(),
             'corr': 0.0, 'fitter': None}

    def __init__(self, config_file, stats_folder='stats',
                 error_metric='sse', bins: Union[str, int] = "auto"):
        self.cfg = Config(config_file)
        self.video_state = VideoState(self.cfg)
        self.dectime_filename = self.video_state.result_file

        self.stats_folder = f'results/{self.cfg.project}/{stats_folder}'
        os.makedirs(self.stats_folder, exist_ok=True)

        self.raw_dectime = {}
        self.data = {}
        self.time = []
        self.rate = []

        # Fitter attributes
        self.fit_file: Union[str, None] = None
        self.error_metric = error_metric
        self.bins = bins
        self.fit: Union[fitter.Fitter, None] = None

        self.load_dectime()

    def load_dectime(self):
        with open(self.dectime_filename) as f:
            self.raw_dectime = json.load(f)

    def make_fit(self, time_or_rate, overwrite=False):
        data_serie = eval(f'self.{time_or_rate}')

        if self.fit_file is None:
            self.fit_file = f'{self.stats_folder}/' \
                            f'fitter_pickle_{len(self.time)}.pickle'

        # Calcule fit
        if os.path.exists(self.fit_file) and not overwrite:
            with open(self.fit_file, 'rb') as f:
                self.fit = pickle.load(f)
        else:
            self.fit = fitter.Fitter(data_serie, bins=self.bins, timeout=30,
                                     distributions=self.cfg.distributions)
            self.fit.fit()

        bins = len(self.fit.x)
        if self.error_metric in 'sse':
            pass
        if self.error_metric in 'rmse':
            f.df_errors = np.sqrt(f.df_errors / bins)
        elif self.error_metric in 'nrmse':
            data_range = (f.df_errors.max() - f.df_errors.min())
            f.df_errors = (np.sqrt(f.df_errors / bins) / data_range)

    def calc_stats(self):
        if self.time or self.rate is None:
            raise ValueError('Atributos time ou rate não definido')

        self.make_fit('time')

        # Calculate percentiles
        per = [0, 25, 50, 75, 100]
        percentile = np.percentile([self.time, self.rate], per)
        percentile_t = {key: value for key, value in
                        zip(per, percentile[:, 0])}
        percentile_r = {key: value for key, value in
                        zip(per, percentile[:, 1])}

        # Struct statistics results
        self.stats['average_t'] = np.average(self.time)
        self.stats['std_t'] = float(np.std(self.time))
        self.stats['average_r'] = np.average(self.rate)
        self.stats['std_r'] = float(np.std(self.rate))
        self.stats['corr'] = np.corrcoef((self.time, self.rate))[1][0]
        self.stats['percentile_t'] = percentile_t
        self.stats['percentile_r'] = percentile_r

    def _get_data(self, groups: Union[List[int], None] = None,
                  videos_list: Union[List[Video], None] = None,
                  pattern_list: Union[List[Pattern], None] = None,
                  quality_list: Union[List[int], None] = None,
                  tiles_list: Union[List[Tile], None] = None,
                  chunks_list: Union[range, None] = None):
        """
        Pega os dados do json
        :param groups:
        :param videos_list:
        :param pattern_list:
        :param quality_list:
        :param tiles_list:
        :param chunks_list:
        :return: (pd.DataFrame, pd.DataFrame, double)
        """
        if videos_list is None:
            videos_list = self.cfg.videos_list
        if pattern_list is None:
            pattern_list = self.cfg.pattern_list
        if quality_list is None:
            quality_list = self.cfg.quality_list
        if tiles_list is None:
            tiles_list = "pattern.tiles_list"
        else:
            tiles_list = f'{tiles_list}'
        if chunks_list is None:
            chunks_list = "video.chunks"
        else:
            chunks_list = f'{chunks_list}'

        time = []
        rate = []
        for video in videos_list:
            if groups is not None and video.group in groups: continue
            for pattern in pattern_list:
                for quality in quality_list:
                    for tile in eval(tiles_list):
                        for chunk in eval(chunks_list):
                            time.append(self.raw_dectime[
                                            video.name][
                                            pattern.pattern][
                                            quality][
                                            tile.idx][
                                            chunk]['time'])

                            rate.append(self.raw_dectime[
                                            video.name][
                                            pattern.pattern][
                                            quality][
                                            tile.idx][
                                            chunk]['rate'])
        return {'time': time, 'rate': rate}

    def get_data_by_pattern(self, pattern_list=None):
        if pattern_list is None:
            pattern_list = self.cfg.pattern_list

        for pattern in pattern_list:
            self.data[pattern.pattern] = self._get_data(pattern_list=[pattern])

    def get_data_by_pattern_quality(self, pattern_list=None, quality_list=None):
        if pattern_list is None:
            pattern_list = self.cfg.pattern_list
        if quality_list is None:
            quality_list = self.cfg.quality_list

        for pattern in pattern_list:
            for quality in quality_list:
                time, rate = self._get_data(pattern_list=[pattern],
                                            quality_list=[quality])
                key = f'{pattern.pattern}_{quality}'
                self.data[key] = dict(time=time,
                                      rate=rate)

    def get_data_by_video_pattern(self, videos_list=None, pattern_list=None):
        if videos_list is None:
            videos_list = self.cfg.videos_list
        if pattern_list is None:
            pattern_list = self.cfg.pattern_list

        for video in videos_list:
            for pattern in pattern_list:
                time, rate = self._get_data(videos_list=[video],
                                            pattern_list=[pattern])
                key = f'{video.name}_{pattern.pattern}'
                self.data[key] = dict(time=time,
                                      rate=rate)

    def get_data_by_video_pattern_quality(self, videos_list=None,
                                          pattern_list=None, quality_list=None):
        if videos_list is None:
            videos_list = self.cfg.videos_list
        if pattern_list is None:
            pattern_list = self.cfg.pattern_list
        if quality_list is None:
            quality_list = self.cfg.quality_list

        for video in videos_list:
            for pattern in pattern_list:
                for quality in quality_list:
                    time, rate = self._get_data(videos_list=[video],
                                                pattern_list=[pattern],
                                                quality_list=[quality])
                    key = f'{video.name}_{pattern.pattern}_{quality}'
                    self.data[key] = dict(time=time,
                                          rate=rate)


class PaperPlots:
    class Role(Enum):
        BY_PATTERN = 0

    def __init__(self, config_file: str,
                 graph_folder='graphs'):
        self.cfg = Config(config_file)
        self.workfolder = f'results/{self.cfg.project}/{graph_folder}'
        os.makedirs(self.workfolder, exist_ok=True)

        self.bins: Union[str, int] = 'auto'
        self.fig = []
        self.axes = []

        self.video_state = VideoState(self.cfg)
        # self.data = DectimeHandler(config_file)

        self.role: Union[None, PaperPlots.Role] = None
        self.rc()

    @staticmethod
    def rc():
        # plt.rc('errorbar', capsize=10)
        plt.rc('figure', figsize=(6.4, 4.8), dpi=300)
        plt.rc('lines', linewidth=0.5, markersize=3)
        plt.rc('axes', linewidth=0.5,
               prop_cycle=cycler(color=[plt.get_cmap('tab20')(i)
                                        for i in range(20)]))

    def hist_by_pattern(self):
        """Usado no SVR e Eletronic Imaging"""
        # self.role = self.role('BY_PATTERN')
        #
        # # Coletando e processando dados
        # self.data.get_data_by_pattern()
        # self.data.calc_stats()
        # self.make_dataframe()
        #
        # self.make_plots(show=True)

        # name = f'hist_by_pattern_{self.bins}bins_{self.cfg.factor}'
        # self.save_plots(name)

    def make_dataframe(self):
        pass

    def make_plots(self):
        pass

    def save_plots(self):
        pass

    def plot_siti(self, one_plot=True):
        """

        :return:
        """
        siti_folder = f'{self.video_state.project}/siti'

        dataframe = []
        si = []
        ti = []

        fig_s: figure.Figure
        fig_t: figure.Figure
        ax_s: axes.Axes
        ax_t: axes.Axes
        ax: List[List[axes.Axes]]

        plt.close()
        # noinspection PyTypeChecker
        fig_s, ax_s = \
            plt.subplots(1, 1, sharey=True,
                         figsize=(9.6, 6.8),
                         dpi=300,
                         tight_layout=True,
                         subplot_kw={'xlabel': 'Frame',
                                     'ylabel': 'Spatial',
                                     'title': 'SI - all videos'}
                         )
        # noinspection PyTypeChecker
        fig_t, ax_t = \
            plt.subplots(1, 1, sharey=True,
                         figsize=(9.6, 6.8),
                         dpi=300,
                         tight_layout=True,
                         subplot_kw={'xlabel': 'Frame',
                                     'ylabel': 'Temporal',
                                     'title': 'TI - all videos'})
        # noinspection PyTypeChecker
        fig_scatter, ax_scatter = \
            plt.subplots(1, 1, sharey=True,
                         figsize=(9.6, 6.8),
                         dpi=300,
                         tight_layout=True,
                         subplot_kw={'xlabel': 'SI',
                                     'ylabel': 'TI',
                                     'title': 'SI/TI - Median Values'})

        for self.video_state.video in self.cfg.videos_list:
            self.video_state.quality = 28
            self.video_state.pattern = Pattern('1x1', self.cfg.frame)
            self.video_state.tile = self.video_state.pattern.tiles_list[0]

            name = self.video_state.video.name
            filename = self.video_state.compressed_file
            folder, _ = os.path.split(filename)
            _, tail = os.path.split(folder)

            siti = pd.read_csv(f'{siti_folder}/{tail}/siti.csv', index_col=0)
            with open(f'{siti_folder}/{tail}/stats.json', 'r',
                      encoding='utf-8') as f:
                stats = json.load(f)

            if one_plot:
                print(f'name={name},\n'
                      f'si={si}\n'
                      f'ti={ti}\n'
                      )
                stats['name'] = name
                dataframe.append(stats)

                si = float(stats["si_2q"])
                ti = float(stats["ti_2q"])

                xerr = np.array([[float(stats["si_1q"])],
                                 [float(stats["si_3q"])]])
                yerr = np.array([[float(stats["ti_1q"])],
                                 [float(stats["ti_3q"])]])

                ax_scatter.errorbar(x=si, y=ti, label=name, fmt='o',
                                    yerr=np.abs(yerr - ti),
                                    xerr=np.abs(xerr - si))
                ax_s.plot(siti['si'], label=name)
                ax_t.plot(siti['ti'], label=name)

            else:
                plt.close()
                fig, ax = plt.subplots(2, 2, tight_layout=True,
                                       figsize=(9.6, 6.8), dpi=300, )
                '''plot si'''
                if True:
                    ax[0][0].plot(siti['si'], label=name)
                    ax[0][0].set_ylim(bottom=0)
                    ax[0][0].set_title(f'{name}')
                    ax[0][0].set_xlabel('frame')
                    ax[0][0].set_ylabel('Spatial')
                    ax[0][0].legend([f'si_average={stats["si_average"]}\n'
                                     f'si_std={stats["si_std"]}\n'
                                     f'si_min={stats["si_0q"]}\n'
                                     f'si_1q={stats["si_1q"]}\n'
                                     f'si_med={stats["si_2q"]}\n'
                                     f'si_3q={stats["si_3q"]}\n'
                                     f'si_max={stats["si_4q"]}'],
                                    loc='upper right')

                '''plot ti'''
                if True:
                    ax[0][1].plot(siti['ti'], label=name)
                    ax[0][1].set_ylim(bottom=0)
                    ax[0][1].set_title(f'{name}')
                    ax[0][1].set_xlabel('frame')
                    ax[0][1].set_ylabel('Temporal')
                    ax[0][1].legend([f'ti_average={stats["ti_average"]}\n'
                                     f'ti_std={stats["ti_std"]}\n'
                                     f'ti_min={stats["ti_0q"]}\n'
                                     f'ti_1q={stats["ti_1q"]}\n'
                                     f'ti_med={stats["ti_2q"]}\n'
                                     f'ti_3q={stats["ti_3q"]}\n'
                                     f'ti_max={stats["ti_4q"]}'],
                                    loc='upper right')

                '''histogram si and ti'''
                if True:
                    label1 = (f'avg={stats["si_average"]}\n'
                              f'std={stats["si_std"]}')
                    label2 = (f'avg={stats["ti_average"]}\n'
                              f'std={stats["ti_std"]}')
                    ax[1][0].hist(siti['si'], histtype='bar', label=label1,
                                  density=True)
                    ax[1][1].hist(siti['ti'], histtype='bar', label=label2,
                                  density=True)
                    ax[1][0].set_xlabel('Spatial Information')
                    ax[1][0].set_ylabel("Frequency")
                    ax[1][1].set_xlabel('Temporal Information')
                    ax[1][1].set_ylabel("Frequency")

                    fig.savefig(f'{siti_folder}/{name}_siti')

        if one_plot:
            ax_scatter.set_ylim(bottom=0)
            ax_scatter.set_xlim(left=0)
            ax_scatter.legend(loc='upper left', bbox_to_anchor=(1.01, 1.01))
            ax_s.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))
            ax_t.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))

            fig_s.savefig(f'{siti_folder}/si_all')
            fig_t.savefig(f'{siti_folder}/ti_all')
            fig_scatter.savefig(f'{siti_folder}/scatter')
            pd.DataFrame(dataframe).to_csv(f'{siti_folder}/stats_all.csv')
