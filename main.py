#!/usr/bin/env python3
import dectime.dectime as dt
import dectime.dectime_analysis as dta


def main():
    """Config"""
    config = f'config/confgit  ig_user_dectime_28videos_nas.json'
    # config = f'config/config_user_dectime_9videos_co_lo.json'
    # config = f'config/config_test.json'

    """Tile_dectime"""
    tile_dectime = dt.TileDecodeBenchmark(config)

    """Processing Calling"""
    tile_dectime.run(dt.Role.PREPARE)
    tile_dectime.run(dt.Role.SITI)
    tile_dectime.run(dt.Role.COMPRESS)
    tile_dectime.run(dt.Role.SEGMENT)
    for _ in range(tile_dectime.config.decode_num):
        tile_dectime.run(dt.Role.DECODE)
    tile_dectime.run(dt.Role.RESULTS)

    """Check files"""
    dt.CheckProject(config_file=config)

    """Process graphs and sheets"""
    tile_dectime.calcule_siti(overwrite=False)
    dta.HistByPattern(config=config)  # .create()
    dta.HistByPatternByQuality(config=config)
    dta.BarByPatternByQuality(config=config)
    dta.HistByPatternFullFrame(config=config).create()
    print('Finish.')


if __name__ == '__main__':
    main()
