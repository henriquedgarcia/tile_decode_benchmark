from dectime.dectime import Dectime, Role, CheckDectime
from dectime.dectime_analysis import HistByPattern, HistByPatternByQuality


def main():
    # config = f'config/config_user_dectime_28videos_nas.json'
    # tile_dectime = Dectime(config)

    """Processing Calling"""
    # tile_dectime.run(Role.PREPARE)
    # tile_dectime.run(Role.SITI)
    # tile_dectime.run(Role.COMPRESS)
    # tile_dectime.run(Role.SEGMENT)
    # for _ in range(tile_dectime.config.decode_num):
    #     tile_dectime.run(Role.DECODE)
    # tile_dectime.run(Role.RESULTS)

    """Check files"""
    # CheckDectime(config_file=config, automate=True)

    """Process graphs and sheets"""
    # plots = PaperPlots(config)
    # plots.hist_by_pattern()
    # plots.plot_siti(one_plot=True)

    print('Finish.')


if __name__ == '__main__':
    main()
