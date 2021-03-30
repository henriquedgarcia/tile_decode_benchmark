import csv
import json
import os
import pickle
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from enum import Enum
from os.path import exists
from typing import Any, Dict, Iterable, List, NamedTuple, Tuple, Union

import matplotlib as mpl
import matplotlib.axes as axes
import matplotlib.figure as figure
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from fitter.fitter import Fitter

from assets.config import Config
from assets.util import (
    AutoDict, calc_stats, dishevel_dictionary,
    update_dictionary,
    )
from assets.video_state import Tile, Tiling, Video, VideoState


class DectimeFactors:
    # Usado no Dectime Analysis
    _config = None
    _name = None
    rate_control = None

    video_name = Union[str, None]
    pattern = Union[str, None]
    quality = Union[int, None]
    tile = Union[int, None]
    chunk = Union[int, None]

    def __init__(self, rate_control):
        self.rate_control = rate_control

    def clear(self):
        self.video_name = None
        self.pattern = None
        self.quality = None
        self.tile = None
        self.chunk = None

    def name(self, base_name: Union[str, None] = None,
             ext: Union[str, None] = None,
             other: Any = None,
             separator='_'):
        name = f'{base_name}' if base_name else None
        if self.video_name:
            name = (f'{name}{separator}{self.video_name}'
                    if name else f'{self.video_name}')
        if self.pattern:
            name = (f'{name}{separator}{self.pattern}'
                    if name else f'{self.pattern}')
        if self.quality:
            name = (f'{name}{separator}{self.rate_control}{self.quality}'
                    if name else f'{self.rate_control}{self.quality}')
        if self.tile:
            name = (f'{name}{separator}tile{self.tile}'
                    if name else f'tile{self.tile}')
        if self.chunk:
            name = (f'{name}{separator}chunk{self.chunk}'
                    if name else f'chunk{self.chunk}')
        if other:
            name = (f'{name}{separator}{other}'
                    if name else f'{other}')
        if ext:
            name = (f'{name}.{ext}'
                    if name else f'{ext}')

        self._name = name
        return name


class DectimeData(NamedTuple):
    time: Union[list, float] = []
    rate: Union[list, float] = []


class ErrorMetric(Enum):
    RMSE = 0
    NRMSE = 1
    SSE = 2


class Dataframes(Enum):
    STATS_DATAFRAME = 'df_stats'
    FITTED_DATAFRAME = 'df_dist'
    PAPER_DATAFRAME = 'df_paper'
    DATA_DATAFRAME = 'df_data'


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
             'corr'     : 0.0, 'fitter': None}

    fit: Fitter
    data: DectimeData
    dectime: dict

    def __init__(self, config: Config, error_metric='sse',
                 bins: Union[str, int] = "auto"):
        self.config = config
        self.video_state = VideoState(self.config)
        self.dectime_filename = self.video_state.dectime_raw_json

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
                 chunks_list: Union[Iterable, None] = None)\
            -> DectimeData:
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
        if not videos_list:
            videos_list = self.config.videos_list
        if not tiling_list:
            tiling_list = self.config.tiling_list
        if not quality_list:
            quality_list = self.config.quality_list
        if not tiles_list:
            tiles_list = "tiling.tiles_list"
        else:
            tiles_list = f'{tiles_list}'
        if not chunks_list:
            chunks_list = "video.chunks"
        else:
            chunks_list = f'{chunks_list}'

        time = []
        rate = []
        for video in videos_list:
            if video.group and str(video.group) in groups: continue
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


class BasePlot(ABC):
    fitter = AutoDict()
    data = AutoDict()
    fit_errors = AutoDict()

    df_data = defaultdict(list)
    df_stats = defaultdict(list)
    df_dist = defaultdict(list)
    df_paper: Union[Dict[Union[int, str], list],
                    pd.DataFrame] = defaultdict(list)

    stats = ()
    colormap = [plt.get_cmap('tab20')(i) for i in range(20)]
    color_list = ("#000000", "#ff8888", "#687964", "#11cc22", "#0f2080",
                  "#ff9910", "#ffc428", "#990099", "#f5793a", "#c9bd9e",
                  "#85c0f9")

    def __init__(self, config: str, folder: str,
                 figsize: Tuple[float, float] = (6.4, 4.8),
                 bins: Union[str, int] = 'auto',
                 error_metric: ErrorMetric = ErrorMetric.RMSE):
        self.context = DectimeFactors(config)
        self.error_metric = error_metric
        self.bins = bins
        self.config = Config(config)

        self.project: str = self.config.project
        self.data_handler: DectimeHandler = DectimeHandler(self.config)
        self.video_list: List[Video] = self.config.videos_list
        self.tiling_list: List[Tiling] = self.config.tiling_list
        self.pattern_list: List[str] = [tiling.pattern
                                        for tiling in self.tiling_list]
        self.quality_list: List[int] = self.config.quality_list
        self.qfactor: str = self.config.factor
        self.dist: List[str] = self.config.distributions

        self.workfolder: str = f'results/{self.project}/graphs/{folder}'
        os.makedirs(f'{self.workfolder}/data', exist_ok=True)

        mpl.rc('figure', figsize=figsize, dpi=300, autolayout=True)
        mpl.rc('lines', linewidth=0.5, markersize=3)
        mpl.rc('errorbar', capsize=10)
        mpl.rc('patch', linewidth=0.5, edgecolor='black', facecolor='#3297c9')
        mpl.rc('axes', linewidth=0.5, prop_cycle=cycler(color=self.colormap))

    def _get_data(self) -> DectimeData:
        assign = lambda ctx: [getattr(self.context, ctx)] if ctx else None
        data = self.data_handler.get_data(
                videos_list=assign(self.context.video_name),
                tiling_list=assign(self.context.pattern),
                quality_list=assign(self.context.quality),
                tiles_list=assign(self.context.tile),
                chunks_list=assign(self.context.chunk))
        return data

    def _make_fit(self, overwrite=False):
        """

        :param overwrite:
        :return:
        """
        name = self.context.name(base_name='fitter_time', ext='pickle',
                                 other=f'{self.bins}bins')
        fitter_pickle_file = f'{self.workfolder}/data/{name}'

        if not overwrite and self._load_fit():
            return

        print(f'Making fit for {name}')
        data: DectimeData = self.get_dict(self.data)
        fitter = Fitter(data.time, bins=self.bins, distributions=self.dist)
        fitter.fit()
        self.update_dict(fitter, self.fitter)

        # Saving fit
        with open(fitter_pickle_file, 'wb') as fd:
            pickle.dump(fitter, fd, pickle.HIGHEST_PROTOCOL)

    def _load_fit(self) -> bool:
        """
        Load fitter from pickle.
        :return: Success or not.
        """
        name = self.context.name(base_name='fitter_time', ext='pickle',
                                 other=f'{self.bins}bins')
        print(f'Loading pickle - {name}.')
        try:
            with open(f'{self.workfolder}/data/{name}', 'rb') as fd:
                fitter = pickle.load(fd)
        except FileNotFoundError:
            return False

        self.update_dict(fitter, self.fitter)
        return True

    def _make_dataframes(self, overwrite):
        # Calculate statistics
        self._calc_stats()
        self._calc_errors()

        # Statistics for df_paper
        self._make_df_paper(overwrite)

        # Update stats dict to df_stats
        self._make_df_stats(overwrite)

        # Register distributions, parameters and fit errors
        self._make_df_dist(overwrite)

        # Make a df from all data.
        self._make_df_data(overwrite)

    def _calc_stats(self):
        data: DectimeData = self.get_dict(self.data)
        self.stats = calc_stats(data.time, data.rate)

    def _calc_errors(self):
        fitter = self.get_dict(self.fitter)

        # Calculate error
        bins = len(fitter.x)
        df_error = fitter.df_errors['sumsquare_error']

        if self.error_metric is ErrorMetric.RMSE:
            errors = np.sqrt(df_error / bins)
        elif self.error_metric is ErrorMetric.SSE:
            errors = df_error
        elif self.error_metric is ErrorMetric.NRMSE:
            rmse = np.sqrt(fitter.df_errors / bins)
            err_range = (fitter.df_errors.max() - fitter.df_errors.min())
            errors = (rmse / err_range)
        else:
            errors = np.NAN

        self.update_dict(errors, self.fit_errors)

    def _make_df_paper(self, overwrite):
        if not overwrite and self._load_dataframes(Dataframes.STATS_DATAFRAME):
            return

        ctx = self.context
        if ctx.video_name: self.df_paper['video_name'].append(ctx.video_name)
        if ctx.pattern: self.df_paper['tiling'].append(ctx.pattern)
        if ctx.quality: self.df_paper['quality'].append(ctx.quality)
        if ctx.tile: self.df_paper['tile'].append(ctx.tile)
        if ctx.chunk: self.df_paper['chunk'].append(ctx.chunk)

        time_stats, rate_stats = self.stats

        self.df_paper['Mean Time'].append(time_stats.average)
        self.df_paper['Deviation Time'].append(time_stats.std)
        self.df_paper['Mean Rate'].append(rate_stats.average)
        self.df_paper['Deviation Rate'].append(rate_stats.std)
        self.df_paper['Correlation'].append(time_stats.correlation)

    def _make_df_stats(self, overwrite):
        if not overwrite and self._load_dataframes(Dataframes.STATS_DATAFRAME):
            return

        ctx = self.context
        if ctx.video_name: self.df_stats['video_name'].append(ctx.video_name)
        if ctx.pattern: self.df_stats['tiling'].append(ctx.pattern)
        if ctx.quality: self.df_stats['quality'].append(ctx.quality)
        if ctx.tile: self.df_stats['tile'].append(ctx.tile)
        if ctx.chunk: self.df_stats['chunk'].append(ctx.chunk)

        time_stats, rate_stats = self.stats

        time_stats = {f'time_{key}': getattr(time_stats, key) for key in
                      time_stats}
        rate_stats = {f'rate_{key}': getattr(rate_stats, key) for key in
                      rate_stats}
        time_stats.update(rate_stats)

        update = lambda key: self.df_stats[key].append(time_stats[key])
        _ = [update(key) for key in time_stats]

    def _make_df_dist(self, overwrite):
        if not overwrite and self._load_dataframes(Dataframes.FITTED_DATAFRAME):
            return

        error = self.get_dict(self.fit_errors)
        dist_list = error.sort_values().index
        df_error = self.get_dict(self.fit_errors)

        for dist in dist_list:
            fitter: Fitter = self.get_dict(self.fitter)
            params = fitter.fitted_param[dist]

            if dist in 'burr12':
                name = 'Burr Type XII'
                parameters = f'c={params[0]}, d={params[1]}'
                loc = params[2]
                scale = params[3]
            elif dist in 'fatiguelife':
                name = 'Birnbaum-Saunders'
                parameters = f'c={params[0]}'
                loc = params[1]
                scale = params[2]
            elif dist in 'gamma':
                name = 'Gamma'
                parameters = f'a={params[0]}'
                loc = params[1]
                scale = params[2]
            elif dist in 'invgauss':
                name = 'Inverse Gaussian'
                parameters = f'mu={params[0]}'
                loc = params[1]
                scale = params[2]
            elif dist in 'rayleigh':
                name = 'Rayleigh'
                parameters = f''
                loc = params[0]
                scale = params[1]
            elif dist in 'lognorm':
                name = 'Log Normal'
                parameters = f's={params[0]}'
                loc = params[1]
                scale = params[2]
            elif dist in 'genpareto':
                name = 'Generalized Pareto'
                parameters = f'c={params[0]}'
                loc = params[1]
                scale = params[2]
            elif dist in 'pareto':
                name = 'Pareto Distribution'
                parameters = f'b={params[0]}'
                loc = params[1]
                scale = params[2]
            elif dist in 'halfnorm':
                name = 'Half-Normal'
                parameters = f''
                loc = params[0]
                scale = params[1]
            elif dist in 'expon':
                name = 'Exponential'
                parameters = f''
                loc = params[0]
                scale = params[1]
            else:
                name = f'erro: {dist} not found'
                parameters = f''
                loc = 0
                scale = 0

            ctx = self.context
            if ctx.video_name: self.df_dist['Video Name'].append(ctx.video_name)
            if ctx.pattern: self.df_dist['Format'].append(ctx.pattern)
            if ctx.quality: self.df_dist['Quality'].append(ctx.quality)
            if ctx.tile: self.df_dist['Tile'].append(ctx.tile)
            if ctx.chunk: self.df_dist['Chunk'].append(ctx.chunk)

            self.df_dist['Distribution'].append(name)
            self.df_dist[self.error_metric.name].append(df_error[dist])
            self.df_dist['Parameters'].append(parameters)
            self.df_dist['Loc'].append(loc)
            self.df_dist['Frame'].append(scale)

    def _make_df_data(self, overwrite):
        if not overwrite and self._load_dataframes(Dataframes.DATA_DATAFRAME):
            return

        time_key = self.context.name('time')
        rate_key = self.context.name('rate')
        data = self.get_dict(self.data)
        self.df_data[time_key] = data.time
        self.df_data[rate_key] = data.rate

    def _save_dataframes(self, overwrite):
        for dataframe in Dataframes:
            path = f'{self.workfolder}/{dataframe.name.lower()}.csv'
            if exists(path) and not overwrite: continue

            if dataframe is Dataframes.DATA_DATAFRAME:
                csv_rows = ([key] + [self.df_data[key]] for key in self.df_data)
                with open(path, 'w', newline='') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerows(csv_rows)

            else:
                df = getattr(self, dataframe.value)
                self.df_stats = pd.DataFrame(df)
                self.df_stats.to_csv(path, index=False)

    def _load_dataframes(self, dataframe_name: Dataframes) -> bool:
        path = f'{self.workfolder}/{dataframe_name.name.lower()}.csv'
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            return False
        setattr(self, dataframe_name.value, df)
        return True

    def _make_hist(self, ax: axes.Axes, n_dist=3, label='empirical'):
        data: DectimeData = self.get_dict(self.data)
        ax.hist(data.time,
                bins=self.bins,
                histtype='bar',
                density=True,
                label=label,
                color='#3297c9')

        fitter: Fitter = self.get_dict(self.fitter)
        dist_error: pd.Series = self.get_dict(self.fit_errors)
        dists = dist_error.sort_values()[0:n_dist].index
        for n, dist_name in enumerate(dists):
            fitted_pdf = fitter.fitted_pdf[dist_name]
            ax.plot(fitter.x, fitted_pdf,
                    label=f'{dist_name}',
                    color=self.color_list[n],
                    lw=1., )

        ax.set_title(self.context.name(separator=' - '))
        ax.set_xlabel('Decoding Time (s)')
        ax.legend(loc='upper right')

    @staticmethod
    def _make_bar(
            plot_type: Union[str, str],
            ax: axes.Axes,
            x: List[float],
            y: List[float],
            yerr: List[float] = None,
            title: str = None,
            label: str = None,
            xlabel: str = None,
            ylabel: str = None,
            xticks: list = None,
            legend: dict = None,
            width: int = 0.8,
            scilimits: Union[Tuple[int], None] = (6, 6)):
        """

        :param plot_type: 'bar' or 'plot'
        :param ax: Matplotlib.axes.Axes
        :param x: list
        :param y: list
        :param yerr: list
        :param label: Label for line
        :param xlabel: x axis name
        :param ylabel: y axis name
        :param xticks: Labels under x ticks
        :param scilimits: scientific notation on y axis
        :return:
        """
        if plot_type == 'bar':
            ax.bar(x, y, width=width, yerr=yerr, label=label)
        elif plot_type == 'plot':
            ax.plot(x, y, color='r', linewidth=1, legend=label)

        if scilimits:
            ax.ticklabel_format(axis='y', style='scientific',
                                scilimits=scilimits)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if xticks:
            ax.set_xticks(x)
            ax.set_xticklabels(xticks)
        if title:
            ax.title(title)
        if legend:
            ax.legend(**legend)

    def run(self, real=True):
        if not real: return

        self.get_data()
        self.make_fit()
        self.make_dataframe()
        self.make_plot()

    @abstractmethod
    def get_data(self):
        pass

    @abstractmethod
    def make_fit(self):
        pass

    @abstractmethod
    def make_dataframe(self):
        """
        Use self._make_dataframes() by iteration and
        self._save_dataframes() to save DataFrame

        :return:
        """
        pass

    @abstractmethod
    def make_plot(self):
        pass

    def get_dict(self, dictionary: dict):
        ctx = self.context
        dictionary = dishevel_dictionary(dictionary, key1=ctx.video_name,
                                         key2=ctx.pattern, key3=ctx.quality,
                                         key4=ctx.tile, key5=ctx.chunk)
        return dictionary

    def update_dict(self, value: Any, dictionary: AutoDict):
        ctx = self.context
        update_dictionary(value, dictionary, key1=ctx.video_name,
                          key2=ctx.pattern,
                          key3=ctx.quality, key4=ctx.tile, key5=ctx.chunk)

    @contextmanager
    def workfolder_ctx(self, path):
        my_workfolder = self.workfolder
        self.workfolder = path
        try:
            yield
        finally:
            self.workfolder = my_workfolder


class HistByPattern(BasePlot):
    def __init__(self, config):
        folder = 'HistByPattern'
        figsize = (16.0, 4.8)
        super().__init__(config, folder=folder, figsize=figsize)

    def get_data(self):
        for self.context.pattern in self.pattern_list:
            data = self._get_data()
            update_dictionary(data, self.data)

    def make_fit(self, overwrite=False):
        for self.context.pattern in self.pattern_list:
            self._make_fit(overwrite=overwrite)

    def make_dataframe(self, overwrite=False):
        for self.context.pattern in self.pattern_list:
            self._make_dataframes(overwrite)
        self._save_dataframes(overwrite)

    def make_plot(self, overwrite=False):
        """Usado no SVR e Electronic Imaging"""
        name = self.context.name('hist_by_pattern', 'png', f'{self.bins}bins')
        path = f'{self.workfolder}/{name}'
        if not overwrite and exists(path): return

        fig = figure.Figure()
        for idx, self.context.pattern in enumerate(self.pattern_list, 1):
            ax: axes.Axes = fig.add_subplot(2, 4, idx)
            self._make_hist(ax)

        # fig.show()
        print(f'Salvando a figura')
        fig.savefig(path)


class HistByPatternByQuality(BasePlot):
    def __init__(self, config):
        folder = 'HistByPatternByQuality'
        figsize = (16.0, 4.8)
        super().__init__(config, folder=folder, figsize=figsize)

    def get_data(self):
        for self.context.pattern in self.pattern_list:
            for self.context.quality in self.quality_list:
                data = self._get_data()
                update_dictionary(data, self.data)

    def make_fit(self, overwrite=False):
        for self.context.pattern in self.pattern_list:
            for self.context.quality in self.quality_list:
                self._make_fit(overwrite=overwrite)

    def make_dataframe(self, overwrite=False):
        for self.context.pattern in self.pattern_list:
            for self.context.quality in self.quality_list:
                self._make_dataframes(overwrite)
        self._save_dataframes(overwrite)

    def make_plot(self, overwrite=False):
        """Usado no SVR e Electronic Imaging"""
        name = self.context.name('hist_by_pattern', 'png', f'{self.bins}bins')
        path = f'{self.workfolder}/{name}'
        if not overwrite and exists(path): return

        for self.context.pattern in self.pattern_list:
            fig = figure.Figure()

            for idx, quality in enumerate(self.quality_list, 1):
                ax: axes.Axes = fig.add_subplot(2, 4, idx)
                self._make_hist(ax)

            print(f'Salvando a figura')
            fig.savefig(path)


class BarByPatternByQuality(HistByPatternByQuality):
    workfolder: str = ''

    def __init__(self, config):
        folder = 'BarByPatternByQuality'
        figsize = (16.0, 4.8)
        super(HistByPatternByQuality, self).__init__(config, folder=folder,
                                                     figsize=figsize)

    def make_fit(self, overwrite=False):
        with self.workfolder_ctx(f'results/{self.project}/graphs/'
                                 f'HistByPatternByQuality'):
            super().make_fit()

    def make_dataframe(self, overwrite=False):
        with self.workfolder_ctx(f'results/{self.project}/graphs/'
                                 f'HistByPatternByQuality'):
            super().make_dataframe()

    def make_plot(self, overwrite=False):
        """Usado no SVR e Electronic Imaging"""
        name = self.context.name('BarByPatternByQuality', ext='png',
                                 other=f'{self.bins}bins')
        path = f'{self.workfolder}/{name}'
        if exists(path) and not overwrite: return

        df = self.df_paper
        fig: figure = plt.figure()
        for idx, pattern in enumerate(self.pattern_list, 1):
            ax = fig.add_subplot(2, 4, idx)

            data: Union[pd.DataFrame] = df[df['tiling'] == pattern]

            # bar plot of assets
            x = xticks = data['quality']
            time_avg = data['Mean Time']
            time_std = data['Deviation Time']
            ylabel = 'Decoding time (s)' if idx in [1, 5] else None
            self._make_bar('bar', ax=ax,
                           x=x, y=time_avg, yerr=time_std,
                           title=f'{pattern}',
                           xlabel='CRF',
                           ylabel=ylabel,
                           xticks=xticks,
                           width=5,
                           scilimits=None)

            # line plot of bit rate
            ax = ax.twinx()
            rate_avg = data['Mean Rate']
            patch = mpatches.Patch(color='#1f77b4', label='Time')
            line = mlines.Line2D([], [], color='red', label='Bitrate')
            legend = {'handles': (patch, line),
                      'loc'    : 'upper right'}
            ylabel = 'Bit Rate (Mbps)' if idx in [1, 5] else None

            self._make_bar('plot', ax=ax,
                           x=x, y=rate_avg,
                           legend=legend,
                           ylabel=ylabel,
                           scilimits=None)
        # fig.show()
        print(f'Salvando a figura')
        fig.savefig(path)


class HistByPatternFullFrame(HistByPattern):
    def __init__(self, config):
        folder = 'HistByPatternFullFrame'
        figsize = (16.0, 4.8)
        super(HistByPattern, self).__init__(config, folder=folder,
                                            figsize=figsize)

    def get_data(self):
        get_data = self.data_handler.get_data

        for tiling in self.tiling_list:
            self.context.pattern = tiling.pattern
            new_data = DectimeData()
            for video in self.video_list:
                self.context.video_name = video.name
                for quality in self.config.quality_list:
                    self.context.quality = quality
                    for chunk in video.chunks:
                        self.context.chunk = chunk
                        data = get_data()
                        new_data.time.append(sum(data.time))
                        new_data.rate.append(sum(data.rate))
            self.context.clear()
            self.context.pattern = tiling.pattern
            self.update_dict(new_data, self.data)

    def make_plot(self, overwrite=False):
        """Usado no SVR e Electronic Imaging"""
        super().make_plot(overwrite)


class BarByPatternFullFrame(HistByPatternFullFrame):
    workfolder: str = ''

    def __init__(self, config):
        folder = 'BarByPatternFullFrame'
        figsize = (16.0, 4.8)
        super(HistByPattern, self).__init__(config, folder=folder,
                                            figsize=figsize)

    def make_fit(self, overwrite=False):
        with self.workfolder_ctx(f'results/{self.project}/graphs/'
                                 f'HistByPatternFullFrame'):
            super().make_fit()

    def make_dataframe(self, overwrite=False):
        with self.workfolder_ctx(f'results/{self.project}/graphs/'
                                 f'HistByPatternFullFrame'):
            super().make_dataframe()

    def make_plot(self, overwrite=False):
        """Usado no SVR e Electronic Imaging"""
        name = self.context.name('BarByPatternFullFrame', ext='png',
                                 other=f'{self.bins}bins')
        path = f'{self.workfolder}/{name}'
        if exists(path) and not overwrite: return

        fig = figure.Figure()
        axt, axr = fig.subplots(2, 1)

        data = self.df_stats

        x = list(range(len(self.pattern_list)))
        time_avg = data['time_average']
        time_std = data['time_std']

        self._make_bar(plot_type='bar',
                       ax=axt,
                       x=x, y=time_avg,
                       ylabel='Decoding Time (s)',
                       yerr=time_std)

        time_avg = data['rate_average']
        time_std = data['rate_std']
        xticks = self.pattern_list
        self._make_bar(plot_type='bar',
                       ax=axr, x=x, y=time_avg,
                       ylabel='Bitrate (Mbps)',
                       xticks=xticks,
                       yerr=time_std)

        # fig.show()
        print(f'Salvando a figura')
        fig.savefig(path)
