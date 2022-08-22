from enum import Enum
from pathlib import Path
from assets2 import Config, GlobalPaths

global config
config: Config


class DectimeGraphsOptions(Enum):
    USERS_METRICS = 0
    VIEWPORT_METRICS = 1
    GET_TILES = 2


class DectimeGraphsPaths(GlobalPaths):
    graphs_folder = Path('graphs')

    @property
    def workfolder(self) -> Path:
        folder = self.project_path / self.graphs_folder / f'{self.__class__.__name__}'
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def workfolder_data(self) -> Path:
        folder = self.workfolder / 'data'
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    # Data Bucket
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

    # Stats file
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
