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


class ErrorMetric(Enum):
    RMSE = 0
    NRMSE = 1
    SSE = 2


class DectimeData(NamedTuple):
    time: list = []
    rate: list = []


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
                 chunks_list: Union[range, None] = None) -> DectimeData:
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

        return DectimeData(time, rate)


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


class HistByPattern(BasePlot):
    def create(self):
        self.get_data()
        self.make_fit()
        self.make_dataframe()
        self.hist_by_pattern()

    def get_data(self):
        get_data = self.data_handler.get_data
        data = {}

        for tilling in self.tiling_list:
            data[tilling.pattern] = get_data(tiling_list=[tilling])

        self.data = data

    def make_fit(self, overwrite=False):
        for tiling in self.tiling_list:
            pattern = tiling.pattern
            name = f'fitter_time_{pattern}_{self.factor}_{self.bins}bins.pickle'
            out_file = f'{self.workfolder}/data/{name}'

            if not (os.path.exists(out_file) and not overwrite):
                print(f'Making fit for {pattern}')
                fitter = Fitter(self.data[pattern].time,
                                bins=self.bins,
                                distributions=self.dist)
                fitter.fit()

                # Saving fit
                with open(out_file, 'wb') as fd:
                    pickle.dump(fitter, fd, pickle.HIGHEST_PROTOCOL)

    def _load_fit(self, pattern: str, out_file=None):
        name = f'fitter_time_{pattern}_{self.factor}_{self.bins}bins.pickle'
        out_file = out_file if out_file else f'{self.workfolder}/data/{name}'

        print(f'Loading pickle from {out_file}.')
        with open(out_file, 'rb') as fd:
            fitter = pickle.load(fd)
        self.fitter[pattern] = fitter

        # Calculate error
        bins = len(fitter.x)
        df_error = fitter.df_errors['sumsquare_error']

        if self.error_metric is ErrorMetric.RMSE:
            self.df_error[pattern] = np.sqrt(df_error / bins)
        elif self.error_metric is ErrorMetric.SSE:
            self.df_error[pattern] = df_error
        elif self.error_metric is ErrorMetric.NRMSE:
            rmse = np.sqrt(fitter.df_errors / bins)
            err_rg = (fitter.df_errors.max() - fitter.df_errors.min())
            self.df_error[pattern] = (rmse / err_rg)

    def make_dataframe(self, overwrite=False):
        exist = (os.path.exists(f'{self.workfolder}/fitted_dataframe.csv') and
                 os.path.exists(f'{self.workfolder}/stats_dataframe.csv') and
                 os.path.exists(f'{self.workfolder}/paper_dataframe.csv'))
        if not (exist and not overwrite):
            error_metric = self.error_metric.name
            fitted_dataframe = {
                'Format': [],
                'Distribution': [],
                error_metric: [],
                'Parameters': [],
                'Loc': [],
                'Scale': []}
            stats_dataframe = {}
            paper_dataframe = {
                'Pattern': [],
                'Mean Time': [],
                'Deviation Time': [],
                'Mean Rate': [],
                'Deviation Rate': [],
                'Correlation': []}

            for tiling in self.tiling_list:
                pattern = tiling.pattern
                self._load_fit(pattern)

                # Register statistics
                time_stats_, rate_stats_ = self._calc_stats(
                    self.data[pattern].time, self.data[pattern].rate)

                paper_dataframe['Pattern'].append(pattern)
                paper_dataframe['Mean Time'].append(time_stats_['average'])
                paper_dataframe['Deviation Time'].append(time_stats_['std'])
                paper_dataframe['Mean Rate'].append(rate_stats_['average'])
                paper_dataframe['Deviation Rate'].append(rate_stats_['std'])
                paper_dataframe['Correlation'].append(
                    time_stats_['correlation'])

                time_stats = {f'time_{key}': time_stats_[key]
                              for key in time_stats_}
                rate_stats = {f'rate_{key}': rate_stats_[key]
                              for key in rate_stats_}

                stats_dataframe[pattern] = time_stats.copy()
                stats_dataframe[pattern].update(rate_stats)

                # Register distributions, parameters and fit errors
                sorted_errors = (self.df_error[pattern]
                                 .sort_values())
                shorted_dist = sorted_errors.index

                for dist in shorted_dist:
                    try:
                        params = self.fitter[pattern].fitted_param[dist]
                    except KeyError:
                        continue

                    fitted_dataframe['Format'].append(pattern)
                    if dist in 'burr12':
                        fitted_dataframe['Distribution'].append('Burr Type XII')
                        fitted_dataframe[error_metric].append(
                            self.df_error[pattern][dist])
                        fitted_dataframe['Parameters'].append(
                            f'c={params[0]}, d={params[1]}')
                        fitted_dataframe['Loc'].append(params[2])
                        fitted_dataframe['Scale'].append(params[3])
                    elif dist in 'fatiguelife':
                        fitted_dataframe['Distribution'].append(
                            'Birnbaum-Saunders')
                        fitted_dataframe[error_metric].append(
                            self.df_error[pattern][dist])
                        fitted_dataframe['Parameters'].append(f'c={params[0]}')
                        fitted_dataframe['Loc'].append(params[1])
                        fitted_dataframe['Scale'].append(params[2])
                    elif dist in 'gamma':
                        fitted_dataframe['Distribution'].append('Gama')
                        fitted_dataframe[error_metric].append(
                            self.df_error[pattern][dist])
                        fitted_dataframe['Parameters'].append(f'a={params[0]}')
                        fitted_dataframe['Loc'].append(params[1])
                        fitted_dataframe['Scale'].append(params[2])
                    elif dist in 'invgauss':
                        fitted_dataframe['Distribution'].append(
                            'Inverse Gaussian')
                        fitted_dataframe[error_metric].append(
                            self.df_error[pattern][dist])
                        fitted_dataframe['Parameters'].append(f'mu={params[0]}')
                        fitted_dataframe['Loc'].append(params[1])
                        fitted_dataframe['Scale'].append(params[2])
                    elif dist in 'rayleigh':
                        fitted_dataframe['Distribution'].append('Rayleigh')
                        fitted_dataframe[error_metric].append(
                            self.df_error[pattern][dist])
                        fitted_dataframe['Parameters'].append(f'')
                        fitted_dataframe['Loc'].append(params[0])
                        fitted_dataframe['Scale'].append(params[1])
                    elif dist in 'lognorm':
                        fitted_dataframe['Distribution'].append('Log Normal')
                        fitted_dataframe[error_metric].append(
                            self.df_error[pattern][dist])
                        fitted_dataframe['Parameters'].append(f's={params[0]}')
                        fitted_dataframe['Loc'].append(params[1])
                        fitted_dataframe['Scale'].append(params[2])
                    elif dist in 'genpareto':
                        fitted_dataframe['Distribution'].append(
                            'Generalized Pareto')
                        fitted_dataframe[error_metric].append(
                            self.df_error[pattern][dist])
                        fitted_dataframe['Parameters'].append(f'c={params[0]}')
                        fitted_dataframe['Loc'].append(params[1])
                        fitted_dataframe['Scale'].append(params[2])
                    elif dist in 'pareto':
                        fitted_dataframe['Distribution'].append(
                            'Pareto Distribution')
                        fitted_dataframe[error_metric].append(
                            self.df_error[pattern][dist])
                        fitted_dataframe['Parameters'].append(f'b={params[0]}')
                        fitted_dataframe['Loc'].append(params[1])
                        fitted_dataframe['Scale'].append(params[2])
                    elif dist in 'halfnorm':
                        fitted_dataframe['Distribution'].append('Half-Normal')
                        fitted_dataframe[error_metric].append(
                            self.df_error[pattern][dist])
                        fitted_dataframe['Parameters'].append(f'')
                        fitted_dataframe['Loc'].append(params[0])
                        fitted_dataframe['Scale'].append(params[1])
                    elif dist in 'expon':
                        fitted_dataframe['Distribution'].append('Exponential')
                        fitted_dataframe[error_metric].append(
                            self.df_error[pattern][dist])
                        fitted_dataframe['Parameters'].append(f'')
                        fitted_dataframe['Loc'].append(params[0])
                        fitted_dataframe['Scale'].append(params[1])
                    else:
                        fitted_dataframe['Distribution'].append(
                            f'erro: {dist} not found')
                        fitted_dataframe[error_metric].append(0)
                        fitted_dataframe['Parameters'].append(f'')
                        fitted_dataframe['Loc'].append(0)
                        fitted_dataframe['Scale'].append(0)

            # Create Pandas Dataframe
            self.fitted_dataframe = pd.DataFrame(fitted_dataframe)
            self.stats_dataframe = pd.DataFrame(stats_dataframe).T
            self.paper_dataframe = pd.DataFrame(paper_dataframe)
            self.paper_dataframe.to_csv(
                f'{self.workfolder}/paper_dataframe.csv', index=False)
            self.fitted_dataframe.to_csv(
                f'{self.workfolder}/fitted_dataframe.csv', index=False)
            self.stats_dataframe.to_csv(
                f'{self.workfolder}/stats_dataframe.csv', index_label='pattern')

    def _load_dataframes(self):
        self.fitted_dataframe = pd.read_csv(
            f'{self.workfolder}/fitted_dataframe.csv')
        self.stats_dataframe = pd.read_csv(
            f'{self.workfolder}/stats_dataframe.csv', index_col=0)

    def _make_hist(self, ax, pattern, n_dist=3):
        ax.set_title(f'{pattern}')
        ax.set_xlabel('Decoding Time (s)')

        # ax.hist(self.data[pattern].time,
        #         bins=self.bins,
        #         histtype='bar',
        #         density=True,
        #         label="Empirical",
        #         color='#3297c9')
        #
        ax.bar(self.fitter[pattern].x, self.fitter[pattern].y,
               width=self.fitter[pattern].x[1] - self.fitter[pattern].x[0],
               label='empirical', color='#3297c9', edgecolor='black',
               linewidth=0.5)

        dist_error = self.df_error[pattern]
        errors_sorted = dist_error.sort_values()[0:n_dist]
        c = iter(self.color_list)

        for dist_name in errors_sorted.index:
            label = f'{dist_name}'
            ax.plot(self.fitter[pattern].x,
                    self.fitter[pattern].fitted_pdf[dist_name],
                    label=label,
                    color=next(c),
                    lw=1., )

        ax.legend(loc='upper right')

    def hist_by_pattern(self, name=None, overwrite=False):
        """Usado no SVR e Electronic Imaging"""
        name = name if name else f'hist_by_pattern_{self.bins}bins_{self.factor}'
        filename = f'{self.workfolder}/{name}.png'
        exist = os.path.exists(filename)
        if not (exist and not overwrite):
            self._load_dataframes()

            fig = figure.Figure()

            for n, tilling in enumerate(self.tiling_list, 1):
                pattern = tilling.pattern
                self._load_fit(pattern)

                if n > 2 * 4: break

                ax: axes.Axes = fig.add_subplot(2, 4, n)
                self._make_hist(ax, pattern)

            print(f'Salvando a figura')
            # fig.show()
            fig.savefig(filename)


class HistByPatternByQuality(BasePlot):
    def create(self):
        self.get_data()
        self.make_fit()
        self.make_dataframe()
        self.hist_by_pattern_by_quality()
        self.hist_by_quality_by_pattern()
        self.bar_by_pattern_by_quality()

    def get_data(self):
        get_data = self.data_handler.get_data
        data = {}

        for tilling in self.tiling_list:
            for quality in self.quality_list:
                data[tilling.pattern][quality] = get_data(tiling_list=[tilling],
                                                          quality_list=[
                                                              quality])

        self.data = data

    def make_fit(self, overwrite=False):
        for tiling in self.tiling_list:
            for quality in self.quality_list:
                pattern = tiling.pattern
                name = (f'fitter_time_{pattern}_{self.factor}{quality}'
                        f'_{self.bins}bins.pickle')
                out_file = f'{self.workfolder}/data/{name}'

                if not (os.path.exists(out_file) and not overwrite):
                    print(f'Making fit for {pattern}_{self.factor}{quality}')
                    fitter = Fitter(self.data[pattern][quality].time,
                                    bins=self.bins,
                                    distributions=self.dist)
                    fitter.fit()

                    # Saving fit
                    with open(out_file, 'wb') as fd:
                        pickle.dump(fitter, fd, pickle.HIGHEST_PROTOCOL)

    def _load_fit(self, pattern: str, quality: int, out_file=None):
        name = (f'fitter_time_{pattern}_{self.factor}{quality}'
                f'_{self.bins}bins.pickle')
        out_file = out_file if out_file else f'{self.workfolder}/data/{name}'

        print(f'Loading pickle from {out_file}.')
        with open(out_file, 'rb') as fd:
            fitter = pickle.load(fd)
        self.fitter[pattern][quality] = fitter

        # Calculate error
        bins = len(fitter.x)
        df_error = fitter.df_errors['sumsquare_error']

        if self.error_metric is ErrorMetric.RMSE:
            self.df_error[pattern][quality] = np.sqrt(df_error / bins)
        elif self.error_metric is ErrorMetric.SSE:
            self.df_error[pattern][quality] = df_error
        elif self.error_metric is ErrorMetric.NRMSE:
            rmse = np.sqrt(fitter.df_errors / bins)
            err_rg = (fitter.df_errors.max() - fitter.df_errors.min())
            self.df_error[pattern][quality] = (rmse / err_rg)

    def make_dataframe(self, overwrite=False):
        exist = (os.path.exists(f'{self.workfolder}/fitted_dataframe.csv') and
                 os.path.exists(f'{self.workfolder}/stats_dataframe.csv') and
                 os.path.exists(f'{self.workfolder}/paper_dataframe.csv'))
        if not (exist and not overwrite):
            error_metric = self.error_metric.name
            fitted_dataframe = {
                'Format': [],
                'Distribution': [],
                error_metric: [],
                'Parameters': [],
                'Loc': [],
                'Scale': []}
            stats_dataframe = {}
            paper_dataframe = {
                'Pattern': [],
                'Mean Time': [],
                'Deviation Time': [],
                'Mean Rate': [],
                'Deviation Rate': [],
                'Correlation': []}

            for tiling in self.tiling_list:
                pattern = tiling.pattern
                self._load_fit(pattern)

                # Register statistics
                time_stats_, rate_stats_ = self._calc_stats(
                    self.data[pattern].time, self.data[pattern].rate)

                paper_dataframe['Pattern'].append(pattern)
                paper_dataframe['Mean Time'].append(time_stats_['average'])
                paper_dataframe['Deviation Time'].append(time_stats_['std'])
                paper_dataframe['Mean Rate'].append(rate_stats_['average'])
                paper_dataframe['Deviation Rate'].append(rate_stats_['std'])
                paper_dataframe['Correlation'].append(
                    time_stats_['correlation'])

                time_stats = {f'time_{key}': time_stats_[key]
                              for key in time_stats_}
                rate_stats = {f'rate_{key}': rate_stats_[key]
                              for key in rate_stats_}

                stats_dataframe[pattern] = time_stats.copy()
                stats_dataframe[pattern].update(rate_stats)

                # Register distributions, parameters and fit errors
                sorted_errors = (self.df_error[pattern]
                                 .sort_values())
                shorted_dist = sorted_errors.index

                for dist in shorted_dist:
                    try:
                        params = self.fitter[pattern].fitted_param[dist]
                    except KeyError:
                        continue

                    fitted_dataframe['Format'].append(pattern)
                    if dist in 'burr12':
                        fitted_dataframe['Distribution'].append('Burr Type XII')
                        fitted_dataframe[error_metric].append(
                            self.df_error[pattern][dist])
                        fitted_dataframe['Parameters'].append(
                            f'c={params[0]}, d={params[1]}')
                        fitted_dataframe['Loc'].append(params[2])
                        fitted_dataframe['Scale'].append(params[3])
                    elif dist in 'fatiguelife':
                        fitted_dataframe['Distribution'].append(
                            'Birnbaum-Saunders')
                        fitted_dataframe[error_metric].append(
                            self.df_error[pattern][dist])
                        fitted_dataframe['Parameters'].append(f'c={params[0]}')
                        fitted_dataframe['Loc'].append(params[1])
                        fitted_dataframe['Scale'].append(params[2])
                    elif dist in 'gamma':
                        fitted_dataframe['Distribution'].append('Gama')
                        fitted_dataframe[error_metric].append(
                            self.df_error[pattern][dist])
                        fitted_dataframe['Parameters'].append(f'a={params[0]}')
                        fitted_dataframe['Loc'].append(params[1])
                        fitted_dataframe['Scale'].append(params[2])
                    elif dist in 'invgauss':
                        fitted_dataframe['Distribution'].append(
                            'Inverse Gaussian')
                        fitted_dataframe[error_metric].append(
                            self.df_error[pattern][dist])
                        fitted_dataframe['Parameters'].append(f'mu={params[0]}')
                        fitted_dataframe['Loc'].append(params[1])
                        fitted_dataframe['Scale'].append(params[2])
                    elif dist in 'rayleigh':
                        fitted_dataframe['Distribution'].append('Rayleigh')
                        fitted_dataframe[error_metric].append(
                            self.df_error[pattern][dist])
                        fitted_dataframe['Parameters'].append(f'')
                        fitted_dataframe['Loc'].append(params[0])
                        fitted_dataframe['Scale'].append(params[1])
                    elif dist in 'lognorm':
                        fitted_dataframe['Distribution'].append('Log Normal')
                        fitted_dataframe[error_metric].append(
                            self.df_error[pattern][dist])
                        fitted_dataframe['Parameters'].append(f's={params[0]}')
                        fitted_dataframe['Loc'].append(params[1])
                        fitted_dataframe['Scale'].append(params[2])
                    elif dist in 'genpareto':
                        fitted_dataframe['Distribution'].append(
                            'Generalized Pareto')
                        fitted_dataframe[error_metric].append(
                            self.df_error[pattern][dist])
                        fitted_dataframe['Parameters'].append(f'c={params[0]}')
                        fitted_dataframe['Loc'].append(params[1])
                        fitted_dataframe['Scale'].append(params[2])
                    elif dist in 'pareto':
                        fitted_dataframe['Distribution'].append(
                            'Pareto Distribution')
                        fitted_dataframe[error_metric].append(
                            self.df_error[pattern][dist])
                        fitted_dataframe['Parameters'].append(f'b={params[0]}')
                        fitted_dataframe['Loc'].append(params[1])
                        fitted_dataframe['Scale'].append(params[2])
                    elif dist in 'halfnorm':
                        fitted_dataframe['Distribution'].append('Half-Normal')
                        fitted_dataframe[error_metric].append(
                            self.df_error[pattern][dist])
                        fitted_dataframe['Parameters'].append(f'')
                        fitted_dataframe['Loc'].append(params[0])
                        fitted_dataframe['Scale'].append(params[1])
                    elif dist in 'expon':
                        fitted_dataframe['Distribution'].append('Exponential')
                        fitted_dataframe[error_metric].append(
                            self.df_error[pattern][dist])
                        fitted_dataframe['Parameters'].append(f'')
                        fitted_dataframe['Loc'].append(params[0])
                        fitted_dataframe['Scale'].append(params[1])
                    else:
                        fitted_dataframe['Distribution'].append(
                            f'erro: {dist} not found')
                        fitted_dataframe[error_metric].append(0)
                        fitted_dataframe['Parameters'].append(f'')
                        fitted_dataframe['Loc'].append(0)
                        fitted_dataframe['Scale'].append(0)

            # Create Pandas Dataframe
            self.fitted_dataframe = pd.DataFrame(fitted_dataframe)
            self.stats_dataframe = pd.DataFrame(stats_dataframe).T
            self.paper_dataframe = pd.DataFrame(paper_dataframe)
            self.paper_dataframe.to_csv(
                f'{self.workfolder}/paper_dataframe.csv', index=False)
            self.fitted_dataframe.to_csv(
                f'{self.workfolder}/fitted_dataframe.csv', index=False)
            self.stats_dataframe.to_csv(
                f'{self.workfolder}/stats_dataframe.csv', index_label='pattern')

    def _load_dataframes(self):
        self.fitted_dataframe = pd.read_csv(
            f'{self.workfolder}/fitted_dataframe.csv')
        self.stats_dataframe = pd.read_csv(
            f'{self.workfolder}/stats_dataframe.csv', index_col=0)

    def _make_hist(self, ax, pattern, quality, n_dist=3):
        ax.set_title(f'{pattern}')
        ax.set_xlabel('Decoding Time (s)')

        # ax.hist(self.data[pattern].time,
        #         bins=self.bins,
        #         histtype='bar',
        #         density=True,
        #         label="Empirical",
        #         color='#3297c9')
        #
        ax.bar(self.fitter[pattern].x, self.fitter[pattern].y,
               width=self.fitter[pattern].x[1] - self.fitter[pattern].x[0],
               label='empirical', color='#3297c9', edgecolor='black',
               linewidth=0.5)

        dist_error = self.df_error[pattern]
        errors_sorted = dist_error.sort_values()[0:n_dist]
        c = iter(self.color_list)

        for dist_name in errors_sorted.index:
            label = f'{dist_name}'
            ax.plot(self.fitter[pattern].x,
                    self.fitter[pattern].fitted_pdf[dist_name],
                    label=label,
                    color=next(c),
                    lw=1., )

        ax.legend(loc='upper right')

    def hist_by_pattern_by_quality(self, name=None, overwrite=False):
        """Usado no SVR e Electronic Imaging"""
        name = name if name else f'hist_by_pattern_{self.bins}bins_{self.factor}'
        filename = f'{self.workfolder}/{name}.png'
        exist = os.path.exists(filename)
        if not (exist and not overwrite):
            self._load_dataframes()

            fig = figure.Figure()

            for n, tilling in enumerate(self.tiling_list, 1):
                pattern = tilling.pattern
                self._load_fit(pattern)

                if n > 2 * 4: break

                ax: axes.Axes = fig.add_subplot(2, 4, n)
                self._make_hist(ax, pattern)

            print(f'Salvando a figura')
            fig.savefig(filename)

    def bar_by_pattern_by_quality(self, name=None, overwrite=False):
        pass

    def hist_by_quality_by_pattern(self):
        pass
