import json
from typing import Any, Dict, List


class Config:
    original_folder = 'original'
    lossless_folder = 'lossless'
    compressed_folder = 'compressed'
    segment_folder = 'segment'
    dectime_folder = 'dectime'
    stats_folder = 'stats'
    graphs_folder = "graphs"
    siti_folder = "siti"
    _config_data = {}

    def __init__(self, config: str):
        with open(f'{config}', 'r') as f:
            self._config_data.update(json.load(f))
        with open(f'config/{self._config_data["videos"]}', 'r') as f:
            self._config_data.update(json.load(f))

        self.project: str = self._config_data['project']
        self.error_metric: str = self._config_data['error_metric']
        self.decoding_num: int = self._config_data['decoding_num']
        self.scale: str = self._config_data['scale']
        self.projection: str = self._config_data['projection']
        self.codec: str = self._config_data['codec']
        self.fps: int = self._config_data['fps']
        self.gop: int = self._config_data['gop']
        self.distributions: List[str] = self._config_data['distributions']
        self.rate_control: str = self._config_data['rate_control']
        self.quality_list: List[int] = self._config_data['quality_list']
        self.pattern_list: List[str] = self._config_data['pattern_list']
        self.videos_list: Dict[str, Any] = self._config_data['videos_list']
