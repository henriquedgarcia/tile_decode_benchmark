import json


def splitx(string: str) -> tuple:
    return tuple(map(int, string.split('x')))


class AutoDict(dict):
    def __init__(self, return_type='AutoDict'):
        super().__init__()
        self.return_type = return_type

    def __missing__(self, key):
        value = self[key] = eval(f'{self.return_type}()')
        return value


class Config:
    def __init__(self, config: str):
        from dectime.video_state import Frame

        with open(f'{config}', 'r') as f:
            self.config_data = json.load(f)

        self.original_folder = 'original'
        self.lossless_folder = 'lossless'
        self.compressed_folder = 'compressed'
        self.segment_folder = 'segment'
        self.dectime_folder = 'dectime'
        self.decode_num = self.config_data['decode_times']

        self.project = self.config_data['project']
        self.factor = self.config_data['factor']
        self.frame = Frame(self.config_data['scale'])
        self.fps = self.config_data['fps']
        self.gop = self.config_data['gop']

        self.quality_list = self.config_data['quality_list']
        self.videos_list = []
        self.pattern_list = []
        self.distributions = []

        self._videos_list()
        self._pattern_list()
        print()

    def _videos_list(self):
        from dectime.video_state import Video

        videos_list = self.config_data['videos_list']
        for name in videos_list:
            video_tuple = Video(name, videos_list[name])
            self.videos_list.append(video_tuple)

    def _pattern_list(self):
        from dectime.video_state import Pattern

        pattern_list = self.config_data['pattern_list']

        for pattern_str in pattern_list:
            pattern = Pattern(pattern_str, self.frame)
            self.pattern_list.append(pattern)
