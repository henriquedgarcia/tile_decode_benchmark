#!/usr/bin/env python3
import assets.dectime as dt
import logging

# logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.WARNING)


def main():
    """Config"""
    config = f'config/config_nas_erp.json'
    config = f'config/config_nas_cmp.json'
    # config = f'config/config_test.json'
    # config = f'config/config_ffmpeg_crf_12videos_60s.json'

    """Tile_dectime"""
    dt.TileDecodeBenchmark(config).run('PREPARE', overwrite=False)
    dt.TileDecodeBenchmark(config).run('COMPRESS', overwrite=False)
    dt.TileDecodeBenchmark(config).run('SEGMENT', overwrite=False)
    # dt.TileDecodeBenchmark(config).run('DECODE', overwrite=False)
    # dt.TileDecodeBenchmark(config).run('RESULTS', overwrite=False)
    # dt.TileDecodeBenchmark(config).run('SITI', overwrite=False)

    # """Check files"""
    dt.CheckProject(config=config).run('ORIGINAL', rem_error=False)
    dt.CheckProject(config=config).run('LOSSLESS', rem_error=False)
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
