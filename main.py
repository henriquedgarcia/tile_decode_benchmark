#!/usr/bin/env python3
from dectime.dectime import Dectime, Role, CheckDectime
from dectime.dectime_analysis import HistByPattern, HistByPatternByQuality


def main():
    config = f'config/config_user_dectime_28videos_nas.json'
    # config = f'config/config_user_dectime_9videos_co_lo.json'
    # config = f'config/config_test.json'
    tile_dectime = Dectime(config)

    """Processing Calling"""
    # tile_dectime.run(Role.PREPARE)
    # tile_dectime.run(Role.SITI)
    # tile_dectime.run(Role.COMPRESS)
    # tile_dectime.run(Role.SEGMENT)
    #     tile_dectime.run(Role.DECODE)
    # for _ in range(tile_dectime.config.decoding_num):
    # tile_dectime.run(Role.RESULTS)

    """Check files"""
    CheckDectime(config_file=config, automate=True)

    """Process graphs and sheets"""
    # plots.plot_siti(one_plot=True)
    # HistByPattern(folder='HistByPattern', config=config,
    #               figsize=(16.0, 4.8)).create()
    # HistByPatternByQuality(folder='HistByPatternByQuality', config=config,
    #                        figsize=(16.0, 4.8)).create()
    print('Finish.')


if __name__ == '__main__':
    main()
