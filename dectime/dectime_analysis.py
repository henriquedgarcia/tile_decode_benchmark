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

    class Data(NamedTuple):
        time: list = []
        rate: list = []

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




