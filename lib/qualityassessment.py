import pickle
from enum import Enum
from pathlib import Path
from typing import Union, Callable
from time import time

from .assets2 import Base
from .tiledecodebenchmark import TileDecodeBenchmarkPaths
import numpy as np
from .util import AutoDict, iter_frame, save_json, load_json, save_pickle, splitx, load_pickle
from collections import defaultdict
import pandas as pd
from matplotlib import pyplot as plt


class SegmentsQualityPaths(TileDecodeBenchmarkPaths):
    quality_folder = Path('quality')

    @property
    def video_quality_folder(self) -> Path:
        name =(f'{self.name}_'
               f'{self.resolution}_'
               f'{self.fps}_'
               f'{self.tiling}_'
               f'{self.config["rate_control"]}{self.quality}')
        folder = self.project_path / self.quality_folder / name
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def video_quality_csv(self) -> Path:
        chunk = self.chunk if isinstance(self.chunk, int) else int(self.chunk)
        return self.video_quality_folder / f'tile{self.tile}_{chunk:03d}.csv'

    @property
    def quality_result_json(self) -> Path:
        folder = self.project_path / self.quality_folder
        folder.mkdir(parents=True, exist_ok=True)
        return folder /  f'quality_{self.video}.json'


class SegmentsQualityProps(SegmentsQualityPaths):
    sph_file = Path('lib/sphere_655362.txt')
    sph_points_mask: np.ndarray
    weight_ndarray: np.ndarray
    mask: np.ndarray
    chunk_quality: dict[str, list]
    results: AutoDict
    old_tile: str
    results_dataframe: pd.DataFrame
    method = dict[str, Callable]
    original_quality='0'

    @staticmethod
    def _mse2psnr(mse: float) -> float:
        return 10 * np.log10((255. ** 2 / mse))

    def _prepare_weight_ndarray(self):
        # for self.projection == 'erp' only
        height, width = self.video_shape[:2]
        func = lambda y, x: np.cos((y + 0.5 - height / 2) * np.pi / height)
        self.weight_ndarray: Union[np.ndarray, object] = np.fromfunction(func, (height, width), dtype='float32')

    @property
    def metric_list(self) -> list[str]:
        return ['MSE', 'WS-MSE', 'S-MSE']

    @property
    def quality_list(self) -> list[str]:
        quality_list: list = self.config['quality_list']
        try:
            quality_list.remove('0')
        except ValueError:
            pass
        return quality_list


class SegmentsQuality(SegmentsQualityProps):


    def __init__(self):
        self.print_resume()
        self.init()
        for self.video in self.videos_list:
            for self.tiling in self.tiling_list:
                for self.quality in self.quality_list:
                    for self.tile in self.tile_list:
                        for self.chunk in self.chunk_list:
                            self.all()

    def init(self):
        self.sph_points_mask: np.ndarray = np.zeros(0)
        self.chunk_quality = defaultdict(list)
        self.results = AutoDict()
        self.results_dataframe = pd.DataFrame()
        self.weight_ndarray = np.zeros(0)
        self.method = {'MSE': self._mse,
                       'WS-MSE': self._wsmse,
                       'S-MSE': self._smse_nn}
        self.old_tile = ''

    def all(self):
        print(f'Processing [{self.vid_proj}][{self.video}][{self.tiling}][crf{self.quality}][tile{self.tile}][chunk{self.chunk}]')

        if self.video_quality_csv.exists():
            print(f'The chunk quality csv {self.video_quality_csv} exist. Skipping')
            return

        if not self.segment_file.exists():
            print(f'Segment {self.segment_file} not exist. Skipping')
            return

        if not self.reference_segment.exists():
            print(f'Reference {self.reference_segment} not exist. Skipping')
            return

        frames = zip(iter_frame(self.reference_segment), iter_frame(self.segment_file))
        chunk_quality = defaultdict(list)
        start = time()

        for n, (frame_video1, frame_video2) in enumerate(frames):
            for metric in self.metric_list:
                metric_value = self.method[metric](frame_video1, frame_video2)
                chunk_quality[metric].append(metric_value)
                psnr = self._mse2psnr(metric_value)
                chunk_quality[metric.replace('MSE', 'PSNR')].append(psnr)

            print(f'[{self.vid_proj}][{self.video}][{self.tiling}][{self.tile}][{self.chunk}] - '
                  f'Frame {n} - {time() - start: 0.3f} s', end='\r')
        print('')
        pd.DataFrame(chunk_quality).to_csv(self.video_quality_csv, encoding='utf-8', index_label='frame')

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
        im_sqr_err = (im_ref - im_deg) ** 2
        mse = np.average(im_sqr_err)
        return mse

    def _wsmse(self, im_ref: np.ndarray, im_deg: np.ndarray) -> float:
        """
        Must be same size
        :param im_ref:
        :param im_deg:
        :return:
        """
        shape = self.video_shape[:2]

        if shape != self.weight_ndarray.shape:
            self._prepare_weight_ndarray()

        x1, x2, y1, y2 = self.tile_position()
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
        shape = self.video_shape[:2]

        if self.sph_points_mask.shape != shape:
            self.sph_points_mask = load_sph_file(self.sph_file, shape)

        if self.tile == '0' or self.old_tile != self.tile:
            ph, pw = shape
            M, N = splitx(self.tiling)
            tw, th = int(pw / M), int(ph / N)
            tile = int(self.tile)
            x1 = tile * tw
            y1 = tile * th
            x2 = tile * tw + tw  # not inclusive
            y2 = tile * th + th  # not inclusive
            self.mask = self.sph_points_mask[y1:y2, x1:x2]
            self.old_tile = self.tile

        im_ref_m = im_ref * self.mask
        im_deg_m = im_deg * self.mask

        sqr_dif = (im_ref_m - im_deg_m) ** 2

        smse_nn = sqr_dif.sum() / self.mask.sum()
        return smse_nn

    def collect_results(self):
        print(f'[{self.vid_proj}][{self.video}][{self.tiling}][{self.tile}][{self.chunk}]')

        quality_result_json = self.quality_result_json

        if quality_result_json.exists():
            print(f'The file {quality_result_json} exist. Loading.')
            json_content = quality_result_json.read_text(encoding='utf-8')
            self.results = load_json(json_content)
        else:
            self.results = AutoDict()

        print(f'Processing {self.quality_result_json}')
        if self.quality == self.original_quality:
            # info('Skipping original quality')
            return 'continue'

        if not self.video_quality_csv.exists():
            print(f'The file {self.video_quality_csv} not exist. Skipping.')
            return 'continue'

        csv_dataframe = pd.read_csv(self.video_quality_csv, encoding='utf-8', index_col=0)

        results = self.results[self.vid_proj][self.name][self.tiling][self.quality][self.tile][self.chunk]

        for metric in self.metric_list:
            if results[metric] != {}:
                print(f'The metric {metric} exist for Result.'
                        f'Skipping this metric')
                return

            if len(csv_dataframe[metric]) == 0:
                print(f'The metric {metric} not exist for csv_dataframe'
                        f'. Skipping this metric')
                return

            results[metric] = csv_dataframe[metric].tolist()

        if self.tiling != self.tiling_list[-1]:
            quality_result_json = self.quality_result_json
            save_json(self.results, quality_result_json)

            quality_result_pickle = quality_result_json.with_suffix('.pickle')
            save_pickle(self.results, quality_result_pickle)

        if self.chunk == 1:
            results = self.results[self.vid_proj][self.name][self.tiling][self.quality][self.tile]
            results.update(self._collect_ffmpeg_psnr())

    def _collect_ffmpeg_psnr(self) -> dict[str, float]:
        get_psnr = lambda l: float(l.strip().split(',')[3].split(':')[1])
        get_qp = lambda l: float(l.strip().split(',')[2].split(':')[1])
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


class QualityAssessmentOptions(Enum):
    ALL_METRICS = 0
    GET_RESULTS = 1

    def __repr__(self):
        return str({self.value: self.name})


class QualityAssessment(Base):
    operations = {'ALL_METRICS': SegmentsQuality,
                  'GET_RESULTS': SegmentsQuality,
                  }


def show(img):
    plt.imshow(img)
    plt.show()


def load_sph_file(sph_file: Path, shape: tuple[int, int] = None):
    """
    Load 655362 sample points (elevation, azimuth). Angles in degree.

    :param sph_file:
    :param shape:
    :return:
    """
    sph_points_mask_file = Path('config/sph_points_erp_mask.pickle')

    if sph_points_mask_file.exists():
        sph_points_mask = load_pickle(sph_points_mask_file)
        return sph_points_mask

    iter_file = sph_file.read_text().splitlines()
    sph_points_mask = np.zeros(shape)
    height, width = shape

    pi = np.pi
    pi2 = pi * 2
    pi_2 = pi / 2

    # for each line (sample), convert to cartesian system and horizontal system
    for line in iter_file[1:]:
        el, az = list(map(np.deg2rad, map(float, line.strip().split())))

        # convert to erp image coordinate
        m = int(np.ceil(width * (az + pi) / pi2 - 1))
        n = int(-np.floor(height * (el - pi_2) / pi) - 1)

        if n >= height:
            n = height - 1

        sph_points_mask[n, m] = 1
    save_pickle(sph_points_mask, sph_points_mask_file)
    return sph_points_mask

