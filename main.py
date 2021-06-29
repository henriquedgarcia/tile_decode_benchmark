#!/usr/bin/env python3
import assets.dectime as dt
import logging
from datetime import datetime

date = str(datetime.now()).replace(':', '-')
logging.basicConfig(filename=f'{date}.log', level=logging.DEBUG,
                    format='%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s')


def main():
    """Config"""
    # config = f'config/config_nas_cmp.json'
    config = f'config/config_nas_erp.json'
    # config = f'config/config_test.json'
    # config = f'config/config_ffmpeg_crf_12videos_60s.json'

    """Tile_dectime"""
    # dt.TileDecodeBenchmark(config).run('PREPARE', overwrite=False)
    dt.TileDecodeBenchmark(config).run('COMPRESS', overwrite=False)
    # dt.TileDecodeBenchmark(config).run('SEGMENT', overwrite=False)
    # dt.TileDecodeBenchmark(config).run('DECODE', overwrite=False)
    # dt.TileDecodeBenchmark(config).run('RESULTS', overwrite=False)
    # dt.TileDecodeBenchmark(config).run('SITI', overwrite=False, animate_graph=False, save=True)

    # """Check files"""
    # dt.CheckProject(config=config).run('ORIGINAL', rem_error=False)
    # dt.CheckProject(config=config).run('LOSSLESS', rem_error=False)
    dt.CheckProject(config=config).run('COMPRESS', rem_error=False)
    # dt.CheckProject(config=config).run('SEGMENT', rem_error=False)
    # dt.CheckProject(config=config).run('DECODE', rem_error=False)

    # """Process graphs and sheets"""
    # dta.HistByPattern(config).run(False)
    # dta.HistByPatternByQuality(config).run(False)
    # dta.BarByPatternByQuality(config).run(False)
    # dta.HistByPatternFullFrame(config).run(False)
    # dta.BarByPatternFullFrame(config).run(False)
    print('## Finish. ##')


if __name__ == '__main__':
    main()
