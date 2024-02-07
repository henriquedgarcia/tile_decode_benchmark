from abc import ABC
from collections import defaultdict
from itertools import combinations
from operator import mul
from pathlib import Path

import matplotlib.axes as axes
import matplotlib.figure as figure
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib as mpl

from cycler import cycler
import numpy as np
import pandas as pd
import scipy.stats
from fitter import Fitter

from ._tilequality import SegmentsQualityPaths
from ._userquality import GetTilesPath
from .util import load_json, save_json, save_pickle, load_pickle
from .assets import AutoDict


class DectimeGraphsPaths(SegmentsQualityPaths, GetTilesPath, ABC):
    n_dist = 6
    bins = 30
    stats = defaultdict(list)
    corretations_bucket = defaultdict(list)
    dists_colors = {'burr12': 'tab:blue',
                    'fatiguelife': 'tab:orange',
                    'gamma': 'tab:green',
                    'invgauss': 'tab:red',
                    'rayleigh': 'tab:purple',
                    'lognorm': 'tab:brown',
                    'genpareto': 'tab:pink',
                    'pareto': 'tab:gray',
                    'halfnorm': 'tab:olive',
                    'expon': 'tab:cyan'}

    @property
    def workfolder(self) -> Path:
        folder = self.project_path / self.graphs_folder / self.__class__.__name__
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def workfolder_data(self) -> Path:
        folder = self.workfolder / 'data'
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def data_bucket_file(self) -> Path:
        """
        Need None
        :return:
        """
        path = self.workfolder_data / f'data_bucket.json'
        return path

    @property
    def seen_tiles_data_file(self) -> Path:
        """
        Need None
        :return:
        """
        path = self.project_path / self.graphs_folder / f'seen_tiles.json'
        return path

    @property
    def stats_file(self) -> Path:
        """
        Need bins
        :return:
        """
        stats_file = self.workfolder / f'stats_{self.bins}bins.csv'
        return stats_file

    @property
    def correlations_file(self) -> Path:
        """
        Need None
        :return:
        """
        correlations_file = self.workfolder / f'correlations.csv'
        return correlations_file

    @staticmethod
    def find_dist(dist_name, params):
        if dist_name == 'burr12':
            return dict(name='Burr Type XII',
                        parameters=f'c={params[0]}, d={params[1]}',
                        loc=params[2],
                        scale=params[3])
        elif dist_name == 'fatiguelife':
            return dict(name='Birnbaum-Saunders',
                        parameters=f'c={params[0]}',
                        loc=params[1],
                        scale=params[2])
        elif dist_name == 'gamma':
            return dict(name='Gamma',
                        parameters=f'a={params[0]}',
                        loc=params[1],
                        scale=params[2])
        elif dist_name == 'invgauss':
            return dict(name='Inverse Gaussian',
                        parameters=f'mu={params[0]}',
                        loc=params[1],
                        scale=params[2])
        elif dist_name == 'rayleigh':
            return dict(name='Rayleigh',
                        parameters=f' ',
                        loc=params[0],
                        scale=params[1])
        elif dist_name == 'lognorm':
            return dict(name='Log Normal',
                        parameters=f's={params[0]}',
                        loc=params[1],
                        scale=params[2])
        elif dist_name == 'genpareto':
            return dict(name='Generalized Pareto',
                        parameters=f'c={params[0]}',
                        loc=params[1],
                        scale=params[2])
        elif dist_name == 'pareto':
            return dict(name='Pareto Distribution',
                        parameters=f'b={params[0]}',
                        loc=params[1],
                        scale=params[2])
        elif dist_name == 'halfnorm':
            return dict(name='Half-Normal',
                        parameters=f' ',
                        loc=params[0],
                        scale=params[1])
        elif dist_name == 'expon':
            return dict(name='Exponential',
                        parameters=f' ',
                        loc=params[0],
                        scale=params[1])
        else:
            raise ValueError(f'Distribution unknown: {dist_name}')

    @staticmethod
    def rc_config():
        rc_param = {"figure": {'figsize': (7.0, 1.2), 'dpi': 600, 'autolayout': True},
                    "axes": {'linewidth': 0.5, 'titlesize': 8, 'labelsize': 6,
                             'prop_cycle': cycler(color=[plt.get_cmap('tab20')(i) for i in range(20)])},
                    "xtick": {'labelsize': 6},
                    "ytick": {'labelsize': 6},
                    "legend": {'fontsize': 6},
                    "font": {'size': 6},
                    "patch": {'linewidth': 0.5, 'edgecolor': 'black', 'facecolor': '#3297c9'},
                    "lines": {'linewidth': 0.5, 'markersize': 2},
                    "errorbar": {'capsize': 4},
                    "boxplot": {'flierprops.marker': '+', 'flierprops.markersize': 1, 'flierprops.linewidth': 0.5,
                                'boxprops.linewidth': 0.0,
                                'capprops.linewidth': 1,
                                'medianprops.linewidth': 0.5,
                                'whiskerprops.linewidth': 0.5,
                                }
                    }

        for group in rc_param:
            mpl.rc(group, **rc_param[group])


class ByPatternProps(DectimeGraphsPaths):
    workfolder_data: Path
    workfolder: Path
    metric_label = {'time': {'scilimits': (-3, -3),
                             'xlabel': f'Average Decoding Time (ms)'},
                    'time_std': {'scilimits': (-3, -3),
                                 'xlabel': f'Std Dev Decoding Time (ms)'},
                    'rate': {'scilimits': (6, 6),
                             'xlabel': f'Average Bit Rate (ms)'},
                    'PSNR': {'scilimits': (0, 0),
                             'xlabel': f'Average PSNR'},
                    'WS-PSNR': {'scilimits': (0, 0),
                                'xlabel': f'Average WS-PSNR'},
                    'S-PSNR': {'scilimits': (0, 0),
                               'xlabel': f'Average S-PSNR'}}

    @property
    def metric(self):
        return self._metric

    @metric.setter
    def metric(self, value):
        self._metric = value

    @property
    def proj(self):
        return self._proj

    @proj.setter
    def proj(self, value):
        self._proj = value

    @property
    def tiling(self):
        return self._tiling

    @tiling.setter
    def tiling(self, value):
        self._tiling = value


    @property
    def fitter_pickle_file(self) -> Path:
        """
        Need: metric, proj, tiling, and bins

        :return:  Path(fitter_pickle_file)
        """
        fitter_file = self.workfolder_data / f'fitter_{self.metric}_{self.proj}_{self.tiling}_{self.bins}bins.pickle'
        return fitter_file

    @property
    def workfolder(self) -> Path:
        """
        Need None
        set _workfolder=None if locked
        :return:
        """
        if self._workfolder is None:
            folder = self.project_path / self.graphs_folder / self.__class__.__name__ / 'aggregate'
            folder.mkdir(parents=True, exist_ok=True)
            return folder
        else:
            return self._workfolder

    @property
    def pdf_file(self) -> Path:
        """
        Need: proj, and metric
        :return:
        """
        folder = self.workfolder / 'pdf_cdf'
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f'pdf_{self.proj}_{self.metric}.png'

    @property
    def boxplot_file(self) -> Path:
        """
        Need: proj, and metric
        :return:
        """
        folder = self.workfolder / 'boxplot'
        folder.mkdir(parents=True, exist_ok=True)
        mid = self.metric_list.index(self.metric)
        img_file = folder / f'boxplot_pattern_{mid}_{self.metric}_{self.proj}.png'
        return img_file

    @property
    def cdf_file(self) -> Path:
        """
        Need: proj, and metric
        :return:
        """
        folder = self.workfolder / 'pdf_cdf'
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f'cdf_{self.proj}_{self.metric}.png'

    _fig_pdf = {}
    @property
    def fig_pdf(self) -> figure.Figure:
        key = (self.metric, self.proj)
        try:
            _fig_pdf = self._fig_pdf[key]
        except KeyError:
            self._fig_pdf = {}
            _fig_pdf = self._fig_pdf[key] = plt.figure(figsize=(12.0, 2),
                                                       dpi=600,
                                                       linewidth=0.5,
                                                       tight_layout=True)
        return _fig_pdf

    _fig_cdf = {}
    @property
    def fig_cdf(self) -> figure.Figure:
        key = (self.metric, self.proj)
        try:
            _fig_cdf =  self._fig_cdf[key]
        except KeyError:
            self._fig_cdf = {}
            _fig_cdf = self._fig_cdf[key] = plt.figure(figsize=(12.0, 2),
                                                       dpi=600,
                                                       linewidth=0.5,
                                                       tight_layout=True)
        return _fig_cdf

    _fig_boxplot = {}
    @property
    def fig_boxplot(self) -> figure.Figure:
        # make an image for each metric and projection
        key = (self.metric, self.proj)
        try:
            _fig_boxplot =  self._fig_boxplot[key]
        except KeyError:
            self._fig_boxplot = {}
            _fig_boxplot = self._fig_boxplot[key] = plt.figure(figsize=(6.0, 2),
                                                               dpi=600,
                                                               linewidth=0.5,
                                                               tight_layout=True)
        return _fig_boxplot

class ByPattern(ByPatternProps):
    overwrite = False

    def loop(self):
        self.rc_config()
        self.error_type = 'sse'
        self.data_bucket = {}
        print(f'\n====== Make hist - error_type={self.error_type}, n_dist={self.n_dist} ======')

        for self.metric in self.metric_list:
            for self.proj in ['erp']:
                for self.tiling in self.tiling_list:
                    yield

    def worker(self):
        if not self.stats_file.exists() or self.overwrite:
            self.calc_stats()

        if not self.correlations_file.exists() or self.overwrite:
            self.calc_corr()

        if not self.pdf_file.exists() or self.overwrite:
            # make an image for each metric and projection
            self.make_hist()

        if not self.boxplot_file.exists() or self.overwrite:
            self.make_boxplot()

        # if not self.violinplot_file.exists() or self.overwrite:
        #     self.make_violinplot()

    def make_boxplot(self, overwrite=False):
        print(f'    Make BoxPlot - {self.metric} {self.proj} {self.tiling} - {self.bins} bins')

        try:
            samples = self.data_bucket[self.metric][self.proj][self.tiling]
        except KeyError:
            try:
                self.data_bucket = load_json(self.data_bucket_file, object_hook=dict)
            except FileNotFoundError:
                self.get_data_bucket()
            samples = self.data_bucket[self.metric][self.proj][self.tiling]

        if self.metric in [ 'PSNR', 'WS-PSNR', 'S-PSNR']:
            samples = [data for data in samples if data < 1000]

        index = self.tiling_list.index(self.tiling) + 1
        scilimits = self.metric_label[self.metric]['scilimits']

        ax: axes.Axes = self.fig_boxplot.add_subplot(1, 5, index)
        boxplot_sep = ax.boxplot((samples,), widths=0.8,
                                 whis=(0, 100),
                                 showfliers=False,
                                 boxprops=dict(facecolor='tab:blue'),
                                 flierprops=dict(color='r'),
                                 medianprops=dict(color='k'),
                                 patch_artist=True)
        for cap in boxplot_sep['caps']: cap.set_xdata(cap.get_xdata() + (0.3, -0.3))

        ax.set_xticks([0])
        ax.set_xticklabels([self.tiling])
        ax.ticklabel_format(axis='y', style='scientific', scilimits=scilimits)

        if self.tiling == self.tiling_list[-1]:
            print(f'  Saving the figure')
            suptitle = self.metric_label[self.metric]['xlabel']
            self.fig_boxplot.suptitle(f'{suptitle}')
            self.fig_boxplot.savefig(self.boxplot_file)

    def calc_corr(self):
        print(f'  Processing Correlation')
        corretations_bucket = defaultdict(list)
        for metric1, metric2 in combinations(self.metric_list, r=2):
            for proj in self.proj_list:
                for tiling in self.tiling_list:
                    try:
                        samples1 = self.data_bucket[metric1][proj][tiling]
                    except KeyError:
                        try:
                            self.data_bucket = load_json(self.data_bucket_file, object_hook=dict)
                        except FileNotFoundError:
                            self.get_data_bucket()
                        samples1 = self.data_bucket[metric1][proj][tiling]

                    samples2 = self.data_bucket[metric2][proj][tiling]
                    corrcoef = np.corrcoef((samples1, samples2))[1][0]

                    corretations_bucket[f'metric'].append(f'{metric1}_{metric2}')
                    corretations_bucket[f'proj'].append(proj)
                    corretations_bucket[f'tiling'].append(tiling)
                    corretations_bucket[f'corr'].append(corrcoef)
        print(f'  Saving Correlation')
        pd.DataFrame(corretations_bucket).to_csv(self.correlations_file, index=False)

    def calc_stats(self):
        print(f'\n\n====== Making Statistics - Bins = {self.bins} ======')

        # Get samples and Fitter
        try:
            self.data_bucket = load_json(self.data_bucket_file, object_hook=dict)
        except FileNotFoundError:
            self.get_data_bucket()
        samples = self.data_bucket[self.metric][self.proj][self.tiling]

        try:
            self.fitter = load_pickle(self.fitter_pickle_file)
        except FileNotFoundError:
            self.make_fit()


        # Calculate percentiles
        percentile = np.percentile(samples, [0, 25, 50, 75, 100]).T

        # Calculate errors
        df_errors: pd.DataFrame = self.fitter.df_errors
        sse: pd.Series = df_errors['sumsquare_error']
        bic: pd.Series = df_errors['bic']
        aic: pd.Series = df_errors['aic']
        n_bins = len(self.fitter.x)
        rmse = np.sqrt(sse / n_bins)
        nrmse = rmse / (sse.max() - sse.min())

        # Append info and stats on Dataframe
        self.stats[f'proj'].append(self.proj)
        self.stats[f'tiling'].append(self.tiling)
        self.stats[f'metric'].append(self.metric)
        self.stats[f'bins'].append(self.bins)

        self.stats[f'average'].append(np.average(samples))
        self.stats[f'std'].append(float(np.std(samples)))

        self.stats[f'min'].append(percentile[0])
        self.stats[f'quartile1'].append(percentile[1])
        self.stats[f'median'].append(percentile[2])
        self.stats[f'quartile3'].append(percentile[3])
        self.stats[f'max'].append(percentile[4])

        # Append distributions on Dataframe
        for dist in sse.keys():
            params = self.fitter.fitted_param[dist]
            dist_info = self.find_dist(dist, params)

            self.stats[f'rmse_{dist}'].append(rmse[dist])
            self.stats[f'nrmse_{dist}'].append(nrmse[dist])
            self.stats[f'sse_{dist}'].append(sse[dist])
            self.stats[f'bic_{dist}'].append(bic[dist])
            self.stats[f'aic_{dist}'].append(aic[dist])
            self.stats[f'param_{dist}'].append(dist_info['parameters'])
            self.stats[f'loc_{dist}'].append(dist_info['loc'])
            self.stats[f'scale_{dist}'].append(dist_info['scale'])

        if self.metric == self.metric_list[-1] and self.tiling == self.tiling_list[-1]:
            print(f'  Saving Stats')
            pd.DataFrame(self.stats).to_csv(self.stats_file, index=False)

    def make_hist(self):
        print(f'    Make Histogram - {self.metric} {self.proj} {self.tiling} - {self.bins} bins')

        try:    # Load fitter
            self.fitter = load_pickle(self.fitter_pickle_file)
        except FileNotFoundError:
            self.make_fit()

        index = self.tiling_list.index(self.tiling) + 1

        width = self.fitter.x[1] - self.fitter.x[0]
        cdf_bins_height = np.cumsum([y * width for y in self.fitter.y])
        error_key = 'sumsquare_error' if self.error_type == 'sse' else 'bic'
        error_sorted = self.fitter.df_errors[error_key].sort_values()[0:self.n_dist]
        dists = error_sorted.index
        fitted_pdf = self.fitter.fitted_pdf
        scilimits = self.metric_label[self.metric]['scilimits']
        xlabel = self.metric_label[self.metric]['xlabel']

        # <editor-fold desc="Make PDF">
        # Make bars of histogram
        ax: axes.Axes = self.fig_pdf.add_subplot(1, 5, index)
        ax.bar(self.fitter.x, self.fitter.y, label='empirical', color='#dbdbdb', width=width)

        # Make plot for n_dist distributions
        for dist_name in dists:
            label = f'{dist_name} - {self.error_type.upper()} {error_sorted[dist_name]: 0.3e}'
            ax.plot(self.fitter.x, fitted_pdf[dist_name], color=self.dists_colors[dist_name],
                    label=label)

        ax.ticklabel_format(axis='x', style='scientific', scilimits=scilimits)
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
        ax.set_title(f'{self.proj.upper()} - {self.tiling}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Density' if index in [1] else None)
        ax.legend(loc='upper right')

        # </editor-fold>

        # <editor-fold desc="Make CDF">
        # Make bars of CDF
        ax: axes.Axes = self.fig_cdf.add_subplot(1, 5, index)
        ax.bar(self.fitter.x, cdf_bins_height, label='empirical', color='#dbdbdb', width=width)

        # make plot for n_dist distributions cdf
        for dist_name in dists:
            dist: scipy.stats.rv_continuous = eval("scipy.stats." + dist_name)
            param = self.fitter.fitted_param[dist_name]
            fitted_cdf = dist.cdf(self.fitter.x, *param)
            label = f'{dist_name} - {self.error_type.upper()} {error_sorted[dist_name]: 0.3e}'
            ax.plot(self.fitter.x, fitted_cdf, color=self.dists_colors[dist_name], label=label)

        ax.ticklabel_format(axis='x', style='scientific', scilimits=scilimits)
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
        ax.set_title(f'{self.proj.upper()}-{self.tiling}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Cumulative' if index in [1] else None)
        ax.legend(loc='lower right')
        # </editor-fold>

        if self.tiling == self.tiling_list[-1]:
            print(f'  Saving the CDF and PDF ')
            self.fig_pdf.savefig(self.pdf_file)
            self.fig_cdf.savefig(self.cdf_file)

    def make_fit(self):
        "deve salvar o arquivo"
        # [metric][vid_proj][tiling] = [video, quality, tile, chunk]
        # 1x1 - 10014 chunks - 1/181 tiles por tiling
        # 3x2 - 60084 chunks - 6/181 tiles por tiling
        # 6x4 - 240336 chunks - 24/181 tiles por tiling
        # 9x6 - 540756 chunks - 54/181 tiles por tiling
        # 12x8 - 961344 chunks - 96/181 tiles por tiling
        # total - 1812534 chunks - 181/181 tiles por tiling

        print(f'  Fitting - {self.metric} {self.proj} {self.tiling}... ', end='')

        try:
            samples = self.data_bucket[self.metric][self.proj][self.tiling]
        except KeyError:
            try:
                self.data_bucket = load_json(self.data_bucket_file, object_hook=dict)
            except FileNotFoundError:
                self.get_data_bucket()
            samples = self.data_bucket[self.metric][self.proj][self.tiling]

        # Make a fit
        self.fitter = Fitter(samples, bins=self.bins, distributions=self.config['distributions'], timeout=1500)
        self.fitter.fit()

        # Save
        print(f'  Saving Fitter... ', end='')
        save_pickle(self.fitter, self.fitter_pickle_file)
        print(f'  Finished.')

    def get_data_bucket(self, remove_outliers=False):
        # [metric][vid_proj][tiling] = [video, quality, tile, chunk]
        # 1x1 - 10014 chunks - 1/181 tiles por tiling
        # 3x2 - 60084 chunks - 6/181 tiles por tiling
        # 6x4 - 240336 chunks - 24/181 tiles por tiling
        # 9x6 - 540756 chunks - 54/181 tiles por tiling
        # 12x8 - 961344 chunks - 96/181 tiles por tiling
        # total - 1812534 chunks - 181/181 tiles por tiling

        self.data_bucket = AutoDict()
        json_metrics = lambda metric: {'rate': self.bitrate_result_json,
                                       'time': self.dectime_result_json,
                                       'time_std': self.dectime_result_json,
                                       'PSNR': self.quality_result_json,
                                       'WS-PSNR': self.quality_result_json,
                                       'S-PSNR': self.quality_result_json}[metric]

        def process(metric, value):
            # Process value according the metric
            if metric == 'time':
                new_value = float(np.round(np.average(value['times']), decimals=3))
            elif metric == 'time_std':
                new_value = float(np.round(np.std(value['times']), decimals=6))
            elif metric == 'rate':
                new_value = float(value['rate'])
            else:
                # if self.metric in ['PSNR', 'WS-PSNR', 'S-PSNR']:
                metric_value = value[metric]
                new_value = metric_value
                if float(new_value) == float('inf'):
                    new_value = 1000
                else:
                    new_value = new_value
            return new_value

        for metric in self.metric_list:
            for self.video in self.videos_list:
                vid_data = load_json(json_metrics(metric), object_hook=dict)[self.vid_proj]

                for tiling in self.tiling_list:
                    print(f'\r  {metric} - {self.vid_proj} {self.name} {tiling}. len = ', end='')
                    n_tiles = mul(*map(int, tiling.split('x')))
                    tile_list = list(map(str, range(n_tiles)))
                    bucket = self.data_bucket[metric][self.vid_proj][tiling]

                    for quality in self.quality_list:
                        if quality in '0': continue
                        for tile in tile_list:
                            quality_bucket=[]
                            for chunk in self.chunk_list:
                                chunk_data = vid_data[tiling][quality][tile][chunk]
                                chunk_data = process(metric, chunk_data)

                                if metric in ['PSNR', 'WS-PSNR', 'S-PSNR', 'time', 'time_std', 'rate']:
                                    quality_bucket.append(chunk_data)

                                    if chunk != self.chunk_list[-1]:
                                        continue
                                    else:
                                        chunk_data = np.average(quality_bucket)

                                try:
                                    bucket.append(chunk_data)
                                except AttributeError:
                                    bucket = self.data_bucket[metric][self.vid_proj][tiling] = [chunk_data]

                    print(len(bucket))

        if remove_outliers: self.remove_outliers(self.data_bucket)

        print(f'  Saving ... ', end='')
        save_json(self.data_bucket, self.data_bucket_file)
        print(f'  Finished.')

    def make_violinplot(self, overwrite=False):
        print(f'\n====== Make Violin - Bins = {self.bins} ======')
        folder = self.workfolder / 'violinplot'
        folder.mkdir(parents=True, exist_ok=True)

        subplot_pos = [(1, 5, x) for x in range(1, 6)]  # 1x5
        colors = {'cmp': 'tab:green', 'erp': 'tab:blue'}
        data_bucket = load_json(self.data_bucket_file)
        legend_handles = [mpatches.Patch(color=colors['erp'], label='ERP'),
                          # mpatches.Patch(color=colors['cmp'], label='CMP'),
                          ]

        # make an image for each metric and projection
        for mid, self.metric in enumerate(self.metric_list):
            for self.proj in self.proj_list:
                img_file = folder / f'violinplot_pattern_{mid}{self.metric}_{self.proj}.png'

                if img_file.exists() and not overwrite:
                    print(f'Figure exist. Skipping')
                    continue

                # <editor-fold desc="Format plot">
                if self.metric == 'time':
                    scilimits = (-3, -3)
                    title = f'Average Decoding {self.metric.capitalize()} (ms)'
                elif self.metric == 'time_std':
                    scilimits = (-3, -3)
                    title = f'Std Dev - Decoding {self.metric.capitalize()} (ms)'
                elif self.metric == 'rate':
                    scilimits = (6, 6)
                    title = f'Bit {self.metric.capitalize()} (Mbps)'
                else:
                    scilimits = (0, 0)
                    title = self.metric
                # </editor-fold>

                fig = figure.Figure(figsize=(6.8, 3.84))
                fig.suptitle(f'{title}')

                for self.tiling, (nrows, ncols, index) in zip(self.tiling_list, subplot_pos):
                    # Get data
                    tiling_data = data_bucket[self.metric][self.proj][self.tiling]

                    if self.metric in [ 'PSNR', 'WS-PSNR', 'S-PSNR']:
                        tiling_data = [data for data in tiling_data if data < 1000]

                    ax: axes.Axes = fig.add_subplot(nrows, ncols, index)
                    ax.violinplot([tiling_data], positions=[1],
                                  showmedians=True, widths=0.9)

                    ax.set_xticks([1])
                    ax.set_xticklabels([self.tiling_list[index-1]])
                    ax.ticklabel_format(axis='y', style='scientific', scilimits=scilimits)

                print(f'  Saving the figure')
                fig.savefig(img_file)

    @staticmethod
    def remove_outliers(data, resumenamecsv=None):
        ### Fliers analysis data[self.proj][self.tiling][self.metric]
        print(f' Removing outliers... ', end='')
        resume = defaultdict(list)
        for proj in data:
            for tiling in data[proj]:
                for metric in data[proj][tiling]:
                    data_bucket = data[proj][tiling][metric]

                    min, q1, med, q3, max = np.percentile(data_bucket, [0, 25, 50, 75, 100]).T
                    iqr = 1.5 * (q3 - q1)
                    clean_left = q1 - iqr
                    clean_right = q3 + iqr

                    data_bucket_clean = [d for d in data_bucket
                                         if (clean_left <= d <= clean_right)]
                    data[proj][tiling][metric] = data_bucket

                    resume['projection'] += [proj]
                    resume['tiling'] += [tiling]
                    resume['metric'] += [metric]
                    resume['min'] += [min]
                    resume['q1'] += [q1]
                    resume['median'] += [med]
                    resume['q3'] += [q3]
                    resume['max'] += [max]
                    resume['iqr'] += [iqr]
                    resume['clean_left'] += [clean_left]
                    resume['clean_right'] += [clean_right]
                    resume['original_len'] += [len(data_bucket)]
                    resume['clean_len'] += [len(data_bucket_clean)]
                    resume['clear_rate'] += [len(data_bucket_clean) / len(data_bucket)]
        print(f'  Finished.')
        if resumenamecsv is not None:
            pd.DataFrame(resume).to_csv(resumenamecsv)


DectimeGraphsOptions = {'0': ByPattern,
                        '1': ByPattern,
                        '2': ByPattern,
                        '3': ByPattern,
                        '4': ByPattern,
                        '5': ByPattern,
                        }
