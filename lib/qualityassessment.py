from enum import Enum
from pathlib import Path
from typing import Union

from .assets2 import Base
from .tiledecodebenchmark import TileDecodeBenchmarkPaths


class QualityAssessmentPaths(TileDecodeBenchmarkPaths):
    quality_folder = Path('quality')

    @property
    def quality_video_csv(self) -> Union[Path, None]:
        folder = self.project_path / self.quality_folder / self.basename
        folder.mkdir(parents=True, exist_ok=True)
        chunk = int(str(self.chunk))
        return folder / f'tile{self.tile}_{chunk:03d}.mp4.csv'

    @property
    def quality_result_json(self) -> Union[Path, None]:
        folder = self.project_path / self.quality_folder
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f'quality_{self.video}.json'


class QualityAssessment(QualityAssessmentPaths):
    pass


class QualityAssessmentOptions(Enum):
    USERS_METRICS = 0
    VIEWPORT_METRICS = 1
    GET_TILES = 2

    def __repr__(self):
        return str({self.value: self.name})


class DectimeGraphs(Base):
    operations = {'PROCESS_NASRABADI': QualityAssessment,
                  'TEST_DATASET': QualityAssessment,
                  'GET_TILES': QualityAssessment,
                  }
