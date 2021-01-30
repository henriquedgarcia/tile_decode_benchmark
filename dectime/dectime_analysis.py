import pickle
from enum import Enum
from cycler import cycler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.axes as axes
import matplotlib.figure as figure
from typing import Union, List
import os
import pandas as pd
from dectime.video_state import Tiling, Tile, Video, Config, VideoState
from dectime.util import AutoDict
import json
from fitter.fitter import Fitter
from typing import NamedTuple, Dict, Tuple


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

    fit: Fitter
    data: Data
    dectime: dict

    def __init__(self, config: Config, error_metric='sse',
                 bins: Union[str, int] = "auto"):
        self.config = config
        self.video_state = VideoState(self.config)
        self.dectime_filename = self.video_state.dectime_raw_json

        self.work_folder = (f'results/{self.config.project}/'
                            f'{self.config.stats_folder}')
        os.makedirs(self.work_folder, exist_ok=True)

        # Fitter attributes
        self.error_metric = error_metric
        self.bins = bins

        self.load_dectime(self.dectime_filename)

    def load_dectime(self, dectime_filename):
        with open(dectime_filename) as f:
            self.dectime = json.load(f)

    def get_data(self, groups: Union[List[int], None] = None,
                 videos_list: Union[List[Video], None] = None,
                 tiling_list: Union[List[Tiling], None] = None,
                 quality_list: Union[List[int], None] = None,
                 tiles_list: Union[List[Tile], None] = None,
                 chunks_list: Union[range, None] = None) -> Data:
        """
        Pega os dados do json
        :param groups:
        :param videos_list:
        :param tiling_list:
        :param quality_list:
        :param tiles_list:
        :param chunks_list:
        :return: (pd.DataFrame, pd.DataFrame, double)
        """
        if videos_list is None:
            videos_list = self.config.videos_list
        if tiling_list is None:
            tiling_list = self.config.tiling_list
        if quality_list is None:
            quality_list = self.config.quality_list
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
            for pattern in tiling_list:
                for quality in quality_list:
                    for tile in eval(tiles_list):
                        for chunk in eval(chunks_list):
                            time_ = self.dectime[
                                video.name][
                                pattern.pattern][
                                str(quality)][
                                str(tile.id)][
                                str(chunk)]['time']

                            rate_ = self.dectime[
                                video.name][
                                pattern.pattern][
                                str(quality)][
                                str(tile.id)][
                                str(chunk)]['rate']

                            try:
                                time_ = time_['avg']
                            except KeyError:
                                pass

                            time.append(time_)
                            rate.append(rate_)

        return DectimeHandler.Data(time, rate)

class ErrorMetric(Enum):
    RMSE = 0
    NRMSE = 1
    SSE = 2


class BasePlot:
    fitter = AutoDict()
    data = AutoDict()
    df_error = AutoDict()
    colormap = [plt.get_cmap('tab20')(i) for i in range(20)]
    color_list = ("#000000", "#ff8888", "#687964", "#11cc22", "#0f2080",
                  "#ff9910", "#ffc428", "#990099", "#f5793a", "#c9bd9e",
                  "#85c0f9")

    fitted_dataframe: pd.DataFrame
    stats_dataframe: pd.DataFrame
    paper_dataframe: pd.DataFrame
    config: Config

    def __init__(self, config: str, folder: str, figsize: Tuple[float, float],
                 bins: Union[str, int] = 'auto',
                 error_metric: ErrorMetric = ErrorMetric.RMSE):
        self.error_metric = error_metric
        self.bins = bins
        self.config = Config(config)
        self.project: str = self.config.project
        self.data_handler: DectimeHandler = DectimeHandler(self.config)
        self.tiling_list: List[Tiling] = self.config.tiling_list
        self.quality_list: List[int] = self.config.quality_list
        self.factor: str = self.config.factor
        self.dist: List[str] = self.config.distributions
        os.makedirs(f'{self.workfolder}/data', exist_ok=True)
        self.workfolder: str = f'results/{self.project}/graphs/{folder}'

        mpl.rc('figure', figsize=figsize, dpi=300, autolayout=True)
        mpl.rc('lines', linewidth=0.5, markersize=3)
        mpl.rc('axes', linewidth=0.5, prop_cycle=cycler(color=self.colormap))

    @staticmethod
    def _calc_stats(data1, data2):
        # Percentiles & Correlation
        per = [0, 25, 50, 75, 100]
        corr = np.corrcoef((data1, data2))[1][0]

        # Struct statistics results
        percentile_data1 = np.percentile(data1, per).T
        stats_data1 = dict(average=np.average(data1),
                           std=float(np.std(data1)),
                           correlation=corr,
                           min=percentile_data1[0],
                           quartile1=percentile_data1[1],
                           median=percentile_data1[2],
                           quartile3=percentile_data1[3],
                           max=percentile_data1[4],
                           )

        percentile_data2 = np.percentile(data2, per).T
        stats_data2 = dict(average=np.average(data2),
                           std=float(np.std(data2)),
                           correlation=corr,
                           min=percentile_data2[0],
                           quartile1=percentile_data2[1],
                           median=percentile_data2[2],
                           quartile3=percentile_data2[3],
                           max=percentile_data2[4],
                           )

        return stats_data1, stats_data2

