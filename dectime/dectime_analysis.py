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


class BasePlot(ABC):
    fitter = AutoDict()
    df_error = AutoDict()
    data = AutoDict()
    colormap = [plt.get_cmap('tab20')(i) for i in range(20)]
    color_list = ("#000000", "#ff8888", "#687964", "#11cc22", "#0f2080",
                  "#ff9910", "#ffc428", "#990099", "#f5793a", "#c9bd9e",
                  "#85c0f9")
    config: Config

    def __init__(self, config: str, folder: str,
                 figsize: Tuple[float, float] = (6.4, 4.8),
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

        self.workfolder: str = f'results/{self.project}/graphs/{folder}'
        os.makedirs(f'{self.workfolder}/data', exist_ok=True)

        self.stats_df = defaultdict(list)
        self.dist_df: Union[dict, pd.DataFrame] = defaultdict(list)
        self.paper_df: Dict[Union[int, str], list] = defaultdict(list)

        mpl.rc('figure', figsize=figsize, dpi=300, autolayout=True)
        mpl.rc('lines', linewidth=0.5, markersize=3)
        mpl.rc('patch', linewidth=0.5, edgecolor='black', facecolor='#3297c9')
        mpl.rc('axes', linewidth=0.5, prop_cycle=cycler(color=self.colormap))

    @contextmanager
    def workfolder_ctx(self, path):
        my_workfolder = self.workfolder
        self.workfolder = path
        try:
            yield
        finally:
            self.workfolder = my_workfolder

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

    def _make_dist_df(self,
                      video_name: Union[str, None] = None,
                      pattern: Union[str, None] = None,
                      quality: Union[int, None] = None,
                      tile: Union[int, None] = None,
                      chunk: Union[int, None] = None, ):
        error: pd.Series = dishevel_dictionary(self.df_error, video_name, pattern,
                                               quality, tile, chunk)
        dist_list = error.sort_values().index

        for dist in dist_list:
            fitter: Fitter = dishevel_dictionary(self.fitter, video_name,
                                                 pattern, quality, tile, chunk)
            df_error = fitter.df_errors['sumsquare_error']
            params = fitter.fitted_param[dist]
            error_metric = df_error[dist]
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

            if video_name: self.dist_df['Video Name'].append(video_name)
            if pattern: self.dist_df['Format'].append(pattern)
            if quality: self.dist_df['Quality'].append(quality)
            if tile: self.dist_df['Tile'].append(tile)
            if chunk: self.dist_df['Chunk'].append(chunk)

            self.dist_df['Distribution'].append(name)
            self.dist_df[self.error_metric.name].append(error_metric)
            self.dist_df['Parameters'].append(parameters)
            self.dist_df['Loc'].append(loc)
            self.dist_df['Scale'].append(scale)

    def _get_paper_df(self, time_stats, rate_stats,
                      video_name: Union[str, None] = None,
                      pattern: Union[str, None] = None,
                      quality: Union[int, None] = None,
                      tile: Union[int, None] = None,
                      chunk: Union[int, None] = None, ):
        if video_name: self.paper_df['video_name'].append(video_name)
        if pattern: self.paper_df['pattern'].append(pattern)
        if quality: self.paper_df['quality'].append(quality)
        if tile: self.paper_df['tile'].append(tile)
        if chunk: self.paper_df['chunk'].append(chunk)

        self.paper_df['Mean Time'].append(time_stats['average'])
        self.paper_df['Deviation Time'].append(time_stats['std'])
        self.paper_df['Mean Rate'].append(rate_stats['average'])
        self.paper_df['Deviation Rate'].append(rate_stats['std'])
        self.paper_df['Correlation'].append(time_stats['correlation'])

    def _get_stats_df(self, time_stats, rate_stats,
                      video_name: Union[str, None] = None,
                      pattern: Union[str, None] = None,
                      quality: Union[int, None] = None,
                      tile: Union[int, None] = None,
                      chunk: Union[int, None] = None):
        if video_name:
            self.stats_df['video_name'].append(video_name)
        if pattern:
            self.stats_df['pattern'].append(pattern)
        if quality:
            self.stats_df['quality'].append(quality)
        if tile:
            self.stats_df['tile'].append(tile)
        if chunk:
            self.stats_df['chunk'].append(chunk)

        time_stats = {f'time_{key}': time_stats[key]
                      for key in time_stats}
        rate_stats = {f'rate_{key}': rate_stats[key]
                      for key in rate_stats}
        time_stats.update(rate_stats)

        # _ = [self.stats_df[key].append(time_stats[key]) for key in time_stats]

    def make_name(self, base_name, ext: Union[str, None] = None,
                  video_name: Union[str, None] = None,
                  pattern: Union[str, None] = None,
                  quality: Union[int, None] = None,
                  tile: Union[int, None] = None,
                  chunk: Union[int, None] = None,
                  other: Any = None):
        name = base_name
        if video_name: name = f'{name}_{video_name}'
        if pattern: name = f'{name}_{pattern}'
        if quality: name = f'{name}_{self.factor}{quality}'
        if tile: name = f'{name}_tile{tile}'
        if chunk: name = f'{name}_chunk{chunk}'
        if other: name = f'{name}_{other}'
        if ext: name = f'{name}.{ext}'
        return name

    def _load_fit(self, video_name: Union[str, None] = None,
                  pattern: Union[str, None] = None,
                  quality: Union[int, None] = None,
                  tile: Union[int, None] = None,
                  chunk: Union[int, None] = None,
                  out_file: str = None):
        # Create output filename for fitter
        name = self.make_name('fitter_time', 'pickle', video_name=video_name,
                              pattern=pattern, quality=quality, tile=tile,
                              chunk=chunk, other=f'{self.bins}bins')
        out_file = out_file if out_file else f'{self.workfolder}/data/{name}'

        # Load fitter from pickle
        print(f'Loading pickle from {out_file}.')
        with open(out_file, 'rb') as fd:
            fitter = pickle.load(fd)
        update_dictionary(fitter, self.fitter, key1=video_name, key2=pattern,
                          key3=quality, key4=tile, key5=chunk)

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

        update_dictionary(errors, self.df_error, key1=video_name, key2=pattern,
                          key3=quality, key4=tile, key5=chunk)

    def _make_fit(self, video_name: Union[str, None] = None,
                  pattern: Union[str, None] = None,
                  quality: Union[int, None] = None,
                  tile: Union[int, None] = None,
                  chunk: Union[int, None] = None,
                  overwrite=False):
        name = self.make_name('fitter_time', 'pickle', video_name=video_name,
                              pattern=pattern, quality=quality, tile=tile,
                              chunk=chunk, other=f'{self.bins}bins')
        out_file = f'{self.workfolder}/data/{name}'
        if os.path.exists(out_file) and not overwrite: return

        print(f'Making fit for {name}')
        data: DectimeData = dishevel_dictionary(self.data, video_name, pattern,
                                                quality, tile, chunk)
        fitter = Fitter(data.time, bins=self.bins, distributions=self.dist)
        fitter.fit()

        # Saving fit
        with open(out_file, 'wb') as fd:
            pickle.dump(fitter, fd, pickle.HIGHEST_PROTOCOL)

    def _make_dataframe(self, video_name: Union[str, None] = None,
                        pattern: Union[str, None] = None,
                        quality: Union[int, None] = None,
                        tile: Union[int, None] = None,
                        chunk: Union[int, None] = None):
        self._load_fit(video_name=video_name, pattern=pattern,
                       quality=quality, tile=tile, chunk=chunk)

        # Calculate statistics
        data: DectimeData = dishevel_dictionary(self.data, video_name, pattern,
                                                quality, tile, chunk)
        time_stats, rate_stats = self._calc_stats(data.time, data.rate)

        # Statistics for paper_df
        self._get_paper_df(time_stats, rate_stats,
                           video_name=video_name, pattern=pattern,
                           quality=quality, tile=tile, chunk=chunk)

        # Update stats dict to stats_df
        self._get_stats_df(time_stats, rate_stats,
                           video_name=video_name, pattern=pattern,
                           quality=quality, tile=tile, chunk=chunk)

        # Register distributions, parameters and fit errors
        self._make_dist_df(video_name=video_name,
                           pattern=pattern,
                           quality=quality,
                           tile=tile,
                           chunk=chunk)

    def _save_dataframes(self, stats_dataframe='stats_dataframe',
                         fitted_dataframe='fitted_dataframe',
                         paper_dataframe='paper_dataframe'):
        stats_path = f'{self.workfolder}/{stats_dataframe}.csv'
        dist_path = f'{self.workfolder}/{fitted_dataframe}.csv'
        paper_path = f'{self.workfolder}/{paper_dataframe}.csv'

        self.stats_df = pd.DataFrame(self.stats_df)
        self.stats_df.to_csv(stats_path, index_label='pattern')

        self.dist_df = pd.DataFrame(self.dist_df)
        self.dist_df.to_csv(dist_path, index=False)

        self.paper_df = pd.DataFrame(self.paper_df)
        self.paper_df.to_csv(paper_path, index=False)

    def _load_dataframes(self, stats_dataframe='stats_dataframe',
                         fitted_dataframe='fitted_dataframe',
                         paper_dataframe='paper_dataframe'):
        self.stats_df = pd.read_csv(f'{self.workfolder}/{stats_dataframe}.csv',
                                    index_col=0)
        self.dist_df = pd.read_csv(f'{self.workfolder}/{fitted_dataframe}.csv')
        self.paper_df = pd.read_csv(f'{self.workfolder}/{paper_dataframe}.csv')

    def _make_hist(self, ax: axes.Axes,
                   video_name: Union[str, None] = None,
                   pattern: Union[str, None] = None,
                   quality: Union[int, None] = None,
                   tile: Union[int, None] = None,
                   chunk: Union[int, None] = None,
                   n_dist=3,
                   label='empirical'):
        color = self.color_list
        fitter: Fitter = dishevel_dictionary(self.fitter, video_name, pattern,
                                             quality, tile, chunk)
        dist_error: pd.Series = dishevel_dictionary(self.df_error, video_name,
                                                    pattern, quality, tile, chunk)
        title = self.make_name('histogram', ext=None, video_name=video_name,
                               pattern=pattern, quality=quality, tile=tile,
                               chunk=chunk)
        data: DectimeData = dishevel_dictionary(self.data, video_name, pattern,
                                                quality, tile, chunk)
        ax.hist(data.time,
                bins=self.bins,
                histtype='bar',
                density=True,
                label=label,
                color='#3297c9')

        # ax.bar(fitter.x, fitter.y,
        #        width=fitter.x[1] - fitter.x[0],
        #        label=label)

        dists = dist_error.sort_values()[0:n_dist].index

        for n, dist_name in enumerate(dists):
            fitted_pdf = fitter.fitted_pdf[dist_name]
            ax.plot(fitter.x, fitted_pdf,
                    label=f'{dist_name}',
                    color=color[n],
                    lw=1., )

        ax.set_title(f'{title}')
        ax.set_xlabel('Decoding Time (s)')
        ax.legend(loc='upper right')

    def create(self):
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
        pass

    @abstractmethod
    def make_plot(self):
        pass


class HistByPattern(BasePlot):
    def __init__(self, config):
        folder = 'HistByPattern'
        figsize = (16.0, 4.8)
        super().__init__(config, folder=folder, figsize=figsize)

    def get_data(self):
        get_data = self.data_handler.get_data

        for tilling in self.tiling_list:
            pattern = tilling.pattern
            self.data[pattern] = get_data(tiling_list=[tilling])

    def make_fit(self, overwrite=False):
        for tiling in self.tiling_list:
            self._make_fit(pattern=tiling.pattern,
                           overwrite=overwrite)

    def make_dataframe(self, overwrite=False):
        exist = (os.path.exists(f'{self.workfolder}/fitted_dataframe.csv') and
                 os.path.exists(f'{self.workfolder}/stats_dataframe.csv') and
                 os.path.exists(f'{self.workfolder}/paper_dataframe.csv'))
        if exist and not overwrite:
            self._load_dataframes()
            return

        for tiling in self.tiling_list:
            pattern = tiling.pattern
            self._make_dataframe(pattern=pattern)

        self._save_dataframes()

    def make_plot(self, overwrite=False):
        """Usado no SVR e Electronic Imaging"""
        fig = figure.Figure()

        for n, tilling in enumerate(self.tiling_list, 1):
            pattern = tilling.pattern
            self._load_fit(pattern=pattern)

            if n > 2 * 4: break
            ax: axes.Axes = fig.add_subplot(2, 4, n)

            self._make_hist(ax, pattern=pattern)

        print(f'Salvando a figura')
        # fig.show()
        name = self.make_name('hist_by_pattern', ext='png',
                              other=f'{self.bins}bins')
        fig.savefig(f'{self.workfolder}/{name}')


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
