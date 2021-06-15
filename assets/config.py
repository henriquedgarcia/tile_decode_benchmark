import json
from typing import Any, Dict, List
from assets.util import ConfigBase


class Params:
    original_folder = 'original'
    lossless_folder = 'lossless'
    compressed_folder = 'compressed'
    segment_folder = 'segment'
    dectime_folder = 'dectime'
    stats_folder = 'stats'
    graphs_folder = "graphs"
    siti_folder = "siti"
    project: str
    error_metric: str
    decoding_num: int
    scale: str
    projection: str
    codec: str
    fps: int
    gop: int
    distributions: List[str]
    rate_control: str
    quality_list: List[int]
    pattern_list: List[str]
    videos_list: Dict[str, Any]
    videos_file: str


class Config(ConfigBase, Params):
    def __init__(self, config):
        super().__init__(config)

        with open(f'config/{self.videos_file}', 'r') as f:
            video_list = json.load(f)
            self.videos_list: Dict[str, Any] = video_list['videos_list']
