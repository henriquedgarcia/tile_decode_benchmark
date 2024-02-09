import datetime
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from time import time
from typing import Union, Callable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim, mean_squared_error as mse

from ._tiledecodebenchmark import TileDecodeBenchmarkPaths, Utils
from .assets import Bcolors, Config, Log, AutoDict
from .transform import hcs2erp, hcs2cmp
from .util import save_json, load_json, save_pickle, load_pickle, iter_frame


class SegmentsQualityPaths(TileDecodeBenchmarkPaths):
    quality_folder = Path('quality')

    @property
    def video_quality_folder(self) -> Path:
        folder = self.project_path / self.quality_folder / self.basename2
        # folder = self.project_path / self.quality_folder / self.basename
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def video_quality_csv(self) -> Path:
        return self.video_quality_folder / f'tile{self.tile}_{int(self.chunk):03d}.csv'

    @property
    def quality_result_json(self) -> Path:
        folder = self.project_path / self.quality_folder
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f'quality_{self.video}.json'

    @property
    def quality_result_img(self) -> Path:
        folder = self.project_path / self.quality_folder / '0-graphs'
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f'{self.video}_{self.tiling}_crf{self.quality}.png'


class SegmentsQualityProps(SegmentsQualityPaths, Utils, Log):
    sph_points_mask: np.ndarray
    weight_ndarray: Union[np.ndarray, object]
    mask: np.ndarray
    chunk_quality: dict[str, list]
    results: AutoDict
    old_tile: str
    results_dataframe: pd.DataFrame
    method = dict[str, Callable]
    original_quality = '0'
    log_text: dict
    _video = None
    _tiling: str = None
    metric_list = ['MSE', 'SSIM', 'WS-MSE', 'S-MSE']

    def init(self):
        self.sph_points_mask = np.zeros(0)
        self.weight_ndarray = np.zeros(0)
        self.old_tile = ''

    def _prepare_weight_ndarray(self):
        # for self.projection == 'erp' only
        height, width = self.video_shape[:2]
        r = height / 4

        if self.vid_proj == 'erp':
            def func(y, x):
                return np.cos((y + 0.5 - height / 2) * np.pi / height)
        elif self.vid_proj == 'cmp':
            def func(y, x):
                x = x % (height / 2)
                y = y % (height / 2)
                d = (x + 0.5 - r) ** 2 + (y + 0.5 - r) ** 2
                return (1 + d / (r ** 2)) ** (-1.5)
        else:
            raise ValueError(f'Wrong self.vid_proj. Value == {self.vid_proj}')

        self.weight_ndarray = np.fromfunction(func, (height, width), dtype='float')

    def load_sph_file(self, shape: tuple[int, int] = None):
        """
        Load 655362 sample points (elevation, azimuth). Angles in degree.

        :param shape:
        :return:
        """
        sph_points_mask_file = Path(f'datasets/sph_points_{self.vid_proj}_{"x".join(map(str, shape[::-1]))}_mask.pickle')
        # sph_points_mask_file = Path(f'config/sph_points_erp_4320x2160_mask.pickle')
        try:
            sph_points_mask = load_pickle(sph_points_mask_file)
            return sph_points_mask
        except FileNotFoundError:
            pass

        shape = self.video_shape[:2] if shape is None else shape
        sph_points_mask = np.zeros(shape)

        sph_file = Path('datasets/sphere_655362.txt')
        iter_file = sph_file.read_text().splitlines()[1:]
        # for each line (sample), convert to cartesian system and horizontal system
        for line in iter_file:
            el, az = list(map(np.deg2rad, map(float, line.strip().split())))  # to rad

            if self.vid_proj == 'erp':
                # convert to erp image coordinate
                m, n = hcs2erp(az, el, shape)
            elif self.vid_proj == 'cmp':
                m, n, face = hcs2cmp(az, el, shape)
            else:
                raise ValueError(f'wrong value to self.vid_proj == {self.vid_proj}')

            sph_points_mask[n, m] = 1

        save_pickle(sph_points_mask, sph_points_mask_file)
        return sph_points_mask

    @property
    def quality_list(self) -> list[str]:
        quality_list: list = self.config['quality_list']
        try:
            quality_list.remove('0')
        except ValueError:
            pass
        return quality_list


class SegmentsQuality(SegmentsQualityProps):
    def main(self):
        self.init()
        for self.video in self.videos_list:
            for self.tiling in self.tiling_list:
                for self.quality in self.quality_list:
                    for self.tile in self.tile_list:
                        for self.chunk in self.chunk_list:
                            self.all()

    def skip(self):
        try:
            chunk_quality_df = pd.read_csv(self.video_quality_csv, encoding='utf-8')
            if len(chunk_quality_df['frame']) == 30:
                print(f'[{self.vid_proj}][{self.video}][{self.tiling}][CRF{self.quality}][tile{self.tile}][chunk{self.chunk}]] - '
                      f'EXIST', end='\r')
                return True

        except FileNotFoundError:
            pass

        self.log('video_quality_csv SMALL. Cleaning.', self.segment_file)
        self.video_quality_csv.unlink(missing_ok=True)

        if not self.segment_file.exists():
            self.log('segment_file NOTFOUND', self.segment_file)
            return True

        if not self.reference_segment.exists():
            self.log('reference_segment NOTFOUND', self.reference_segment)
            return True

        return False

    def all(self):
        if self.skip(): return

        print(f'[{self.vid_proj}][{self.video}][{self.tiling}][CRF{self.quality}][tile{self.tile}][chunk{self.chunk}]]')
        chunk_quality = {}
        start = time()

        print("\t ssim.")
        with Pool(8) as p:
            ssim_value = p.starmap(self._ssim, zip(iter_frame(self.reference_segment), iter_frame(self.segment_file)))
        print("\t mse.")
        with Pool(8) as p:
            mse_value = p.starmap(self._mse, zip(iter_frame(self.reference_segment), iter_frame(self.segment_file)))
        print("\t wsmse.")
        with Pool(8) as p:
            wsmse_value = p.starmap(self._wsmse, zip(iter_frame(self.reference_segment), iter_frame(self.segment_file)))
        print("\t smse.")
        with Pool(8) as p:
            smse_nn_value = p.starmap(self._smse_nn, zip(iter_frame(self.reference_segment), iter_frame(self.segment_file)))

        chunk_quality['MSE'] = mse_value
        chunk_quality['SSIM'] = ssim_value
        chunk_quality['WS-MSE'] = wsmse_value
        chunk_quality['S-MSE'] = smse_nn_value

        pd.DataFrame(chunk_quality).to_csv(self.video_quality_csv, encoding='utf-8', index_label='frame')
        print(f"\t time={time() - start}.")

    @staticmethod
    def _mse(im_ref: np.ndarray, im_deg: np.ndarray) -> float:
        """
        https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

        Images must be only one channel (luminance)
        (height, width) = im_ref.shape()
        "float32" = im_ref.dtype()

        :param im_ref:
        :param im_deg:
        :return:
        """
        # im_sqr_err = (im_ref - im_deg) ** 2
        # mse = np.average(im_sqr_err)
        return mse(im_ref, im_deg)

    @staticmethod
    def _ssim(im_ref: np.ndarray, im_deg: np.ndarray) -> float:
        """
        https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

        Images must be only one channel (luminance)
        (height, width) = im_ref.shape()
        "float32" = im_ref.dtype()

        :param im_ref:
        :param im_deg:
        :return:
        """
        # im_sqr_err = (im_ref - im_deg) ** 2
        # mse = np.average(im_sqr_err)
        return ssim(im_ref, im_deg,
                    data_range=255.0,
                    gaussian_weights=True, sigma=1.5,
                    use_sample_covariance=False)

    def _wsmse(self, im_ref: np.ndarray, im_deg: np.ndarray) -> float:
        """
        Must be same size
        :param im_ref:
        :param im_deg:
        :return:
        """
        x1, y1, x2, y2 = self.tile_position[self.tile]
        weight_tile = self.weight_ndarray[y1:y2, x1:x2]
        wmse = np.sum(weight_tile * (im_ref - im_deg) ** 2) / np.sum(weight_tile)
        return wmse

    def _smse_nn(self, im_ref: np.ndarray, im_deg: np.ndarray):
        """
        Calculate of S-PSNR between two images. All arrays must be on the same
        resolution.

        :param im_ref: The original image
        :param im_deg: The image degraded
        :return:
        """
        if self.tile != self.old_tile or self.tile == '0':
            x1, y1, x2, y2 = self.tile_position[self.tile]
            self.mask = self.sph_points_mask[y1:y2, x1:x2]
            self.old_tile = self.tile

        im_ref_m = im_ref * self.mask
        im_deg_m = im_deg * self.mask

        sqr_dif = (im_ref_m - im_deg_m) ** 2

        smse_nn = sqr_dif.sum() / 655362
        return smse_nn

    def _collect_ffmpeg_psnr(self) -> dict[str, float]:
        # deprecated
        def get_psnr(line_txt):
            return float(line_txt.strip().split(',')[3].split(':')[1])

        def get_qp(line_txt):
            return float(line_txt.strip().split(',')[2].split(':')[1])

        psnr = None
        compressed_log = self.compressed_file.with_suffix('.log')
        content = compressed_log.read_text(encoding='utf-8')
        content = content.splitlines()

        for line in content:
            if 'Global PSNR' in line:
                psnr = {'psnr': get_psnr(line),
                        'qp_avg': get_qp(line)}
                break
        return psnr

    @property
    def video(self):
        return self._video

    @video.setter
    def video(self, value):
        self._video = value
        self._skip = False
        self.results = AutoDict()
        shape = self.video_shape[:2]
        if self.weight_ndarray.shape != shape:  # suppose that changed projection, the resolution is changed too.
            self._prepare_weight_ndarray()
        if self.sph_points_mask.shape != shape:  # suppose that change the projection
            self.sph_points_mask = self.load_sph_file(shape)

    @property
    def tiling(self) -> str:
        return self._tiling

    @tiling.setter
    def tiling(self, value: str):
        self._tiling = value
        self.tile_position = {}
        for self.tile in self.tile_list:
            ph, pw = self.video_shape[:2]
            tiling_m, tiling_n = tuple(map(int, self.tiling.split('x')))
            tw, th = int(pw / tiling_m), int(ph / tiling_n)
            tile_x, tile_y = int(self.tile) % tiling_m, int(self.tile) // tiling_m
            x1, x2 = tile_x * tw, tile_x * tw + tw  # not inclusive
            y1, y2 = tile_y * th, tile_y * th + th  # not inclusive
            self.tile_position[self.tile] = [x1, y1, x2, y2]


class CollectResults(SegmentsQualityProps):
    _skip: bool

    @property
    def video(self):
        return self._video

    @video.setter
    def video(self, value):
        self._video = value
        self._skip = False
        self.results = AutoDict()

    def main(self):
        self.get_chunk_value()
        # self.get_tile_image()

    def get_chunk_value(self):
        for self.video in self.videos_list:
            if self.quality_result_json.exists():
                self._skip = True
                self.results = load_json(self.quality_result_json, object_hook=AutoDict)
                print(Bcolors.FAIL + f'The file {self.quality_result_json} exist. Loading.' + Bcolors.ENDC)

            for self.tiling in self.tiling_list:
                for self.quality in self.quality_list:
                    for self.tile in self.tile_list:
                        for self.chunk in self.chunk_list:
                            if self.quality_result_json.exists():
                                try:
                                    assert (len(self.results[self.vid_proj][self.name][self.tiling][self.quality][self.tile][self.chunk]['MSE'])) == 30
                                    continue
                                except (AssertionError, KeyError):
                                    pass
                            self.work()

            print('')
            if not self._skip:
                save_json(self.results, self.quality_result_json)

    def work(self):
        if self._skip: self._skip = False
        try:
            chunk_quality_df = pd.read_csv(self.video_quality_csv, encoding='utf-8')
            if len(chunk_quality_df['frame']) != 30:
                self.video_quality_csv.unlink(missing_ok=True)
                print(f'MISSING_FRAMES')
                self.log(f'MISSING_FRAMES', self.video_quality_csv)
                return
        except FileNotFoundError:
            print(f'CSV_NOTFOUND_ERROR')
            self.log('NOTFOUND_ERROR', self.video_quality_csv)
            return
        except pd.errors.EmptyDataError:
            self.video_quality_csv.unlink(missing_ok=True)
            print(f'CSV_EMPTY_DATA_ERROR')
            self.log('NOTFOUND_ERROR', self.video_quality_csv)
            return

        # https://ffmpeg.org/ffmpeg-filters.html#psnr
        print(f'\rProcessing [{self.vid_proj}][{self.video}][{self.tiling}][crf{self.quality}][tile{self.tile}][chunk{self.chunk}]', end='')
        chunk_results = self.results[self.vid_proj][self.name][self.tiling][self.quality][self.tile][self.chunk]
        for metric in self.metric_list:
            chunk_quality_list = chunk_quality_df[metric].tolist()
            if 0 in chunk_quality_list:
                self.log(f'ERROR_0_FOUND', self.video_quality_csv)
            chunk_results[metric] = chunk_quality_list

    def get_tile_image(self):
        for self.video in self.videos_list:
            for self.tiling in self.tiling_list:
                for self.quality in self.quality_list:
                    self.work2()

    def work2(self):
        if self.quality_result_img.exists():
            print(Bcolors.FAIL + f'The file {self.quality_result_img} exist. Skipping.' + Bcolors.ENDC)
            return

        print(f'\rProcessing [{self.vid_proj}][{self.video}][{self.tiling}][crf{self.quality}]', end='')

        fig, axes = plt.subplots(2, 3, figsize=(8, 5), dpi=200)
        axes: list[plt.Axes] = list(np.ravel(axes))
        fig: plt.Figure

        for self.tile in self.tile_list:
            for i, metric in enumerate(self.metric_list):
                try:
                    result = self.get_result(metric)[:]
                except TypeError:
                    self.results = load_json(self.quality_result_json)
                    result = self.get_result(metric)

                axes[i].plot(result, label=f'{self.tile}')
                axes[i].set_title(metric)

        fig.suptitle(f'[{self.vid_proj}][{self.video}][{self.tiling}][crf{self.quality}][tile{self.tile}]')
        fig.tight_layout()
        # fig.show()
        fig.savefig(self.quality_result_img)
        plt.close()

    def get_result(self, metric):
        chunks_metric = self.results[self.vid_proj][self.name][self.tiling][self.quality][self.tile]
        result = [chunks_metric[self.chunk][metric] for self.chunk in self.chunk_list]
        return result


QualityAssessmentOptions = {'0': SegmentsQuality,
                            '1': CollectResults,
                            }
