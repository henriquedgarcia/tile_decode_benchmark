import json
import os.path
from logging import debug, info
from typing import List

from assets.dectime_types import Frame, Tiling, Video


class Config:
    original_folder = 'original'
    lossless_folder = 'lossless'
    compressed_folder = 'compressed'
    segment_folder = 'segment'
    dectime_folder = 'assets'
    stats_folder = 'stats'
    graphs_folder = "graphs"
    _tiling_list: List[Tiling] = []
    _videos_list: List[Video] = []

    def __init__(self, config: str):
        debug('Starting config.')
        folder = os.path.dirname(config)

        debug('Opening files.')
        with open(f'{config}', 'r') as f:
            self.config_data = json.load(f)
            factors_file = self.config_data['factors']
            video_file = self.config_data['videos']

        with open(f'{folder}/{factors_file}', 'r') as f:
            self.factors_data = json.load(f)

        with open(f'{folder}/{video_file}', 'r') as f:
            self.videos_data = json.load(f)

        debug('Get main configurations.')
        self.project = self.config_data['project']
        self.error_metric = self.config_data['error_metric']
        self.decoding_num = self.config_data['decoding_num']
        self.scale = self.config_data['scale']
        self.frame = Frame(self.scale)
        self.projection = self.config_data['projection']
        self.codec = self.config_data['codec']
        self.fps = self.config_data['fps']
        self.gop = self.config_data['gop']

        debug('Get factors configurations.')
        self.distributions = self.factors_data['distributions']
        self.rate_control = self.factors_data['rate_control']
        self.quality_list = self.factors_data['quality_list']
        self.tiling_list = self.factors_data['pattern_list']

        debug('Get video configurations.')
        self.videos_list = self.videos_data['videos_list']
        info('Config completed.')

    @property
    def tiling_list(self):
        return self._tiling_list

    @tiling_list.setter
    def tiling_list(self, pattern_list):
        debug('Setting tiling_list')
        tiling_list = self._tiling_list
        for pattern in pattern_list:
            tiling_list.append(Tiling(pattern, self.frame))

    @property
    def videos_list(self):
        return self._videos_list

    @videos_list.setter
    def videos_list(self, video_list):
        debug('Setting videos_list')
        for name in video_list:
            video_tuple = Video(name, video_list[name])
            self._videos_list.append(video_tuple)
