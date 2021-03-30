#!/usr/bin/env python3
import assets.dectime as dt


def main():
    """Config"""
    config = f'config/config_test.json'
    # config = f'config/config_user_dectime_28videos_nas.json'
    # config = f'config/config_user_dectime_9videos_co_lo.json'

    """Tile_dectime"""
    # dt.TileDecodeBenchmark(config).run('PREPARE', overwrite=True)
    dt.TileDecodeBenchmark(config).run('COMPRESS')
    # dt.TileDecodeBenchmark(config).run('SEGMENT', overwrite=True)
    # dt.TileDecodeBenchmark(config).run('DECODE')
    # dt.TileDecodeBenchmark(config).run('RESULTS')
    # dt.TileDecodeBenchmark(config).run('SITI', overwrite=True)
    #
    # """Check files"""
    # dt.CheckProject(config_file=config)
    #
    # """Process graphs and sheets"""
    # dta.HistByPattern(config).run(False)
    # dta.HistByPatternByQuality(config).run(False)
    # dta.BarByPatternByQuality(config).run(False)
    # dta.HistByPatternFullFrame(config).run(False)
    # dta.BarByPatternFullFrame(config).run(False)
    print('Finish.')


if __name__ == '__main__':
    main()
