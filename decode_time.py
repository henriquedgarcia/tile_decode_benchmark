from lib.dectime import TileDecodeBenchmark, CheckTileDecodeBenchmark

config_list = []
# config_list += [f'config/config_nas_cmp.json']
# config_list += [f'config/config_nas_erp.json']
config_list += [f'config/config_test.json']
# config_list += [f'config/config_ffmpeg_crf_12videos_60s.json']


for config in config_list:
    # TileDecodeBenchmark(config, 'PREPARE', overwrite=False)
    TileDecodeBenchmark(config, 'COMPRESS', overwrite=False)
    # TileDecodeBenchmark(config, 'SEGMENT', overwrite=False)
    # TileDecodeBenchmark(config, 'DECODE', overwrite=False)
    # TileDecodeBenchmark(config, 'COLLECT_RESULTS', overwrite=False)
    # TileDecodeBenchmark(config, 'MAKE_ALL_GRAPHS', overwrite=False)

    # CheckTileDecodeBenchmark(config, 'CHECK_ORIGINAL')
    # CheckTileDecodeBenchmark(config, 'CHECK_LOSSLESS')

    CheckTileDecodeBenchmark(config, 'CHECK_COMPRESS', only_error=True,
                             check_log=True, check_video=True, check_gop=False)
    # CheckTileDecodeBenchmark(config, 'CHECK_SEGMENT', clean=False,
    #                          check_gop=False)
    # CheckTileDecodeBenchmark(config, 'CLEAN',
    #                          table_of_check=(
    #                              'CHECK_COMPRESS-table-2021-09-30 15:54:29.901483.csv'),
    #                          clean_log=True, clean_video=True,
    #                          video_of_log=False)
    # CheckTileDecodeBenchmark(config, 'CHECK_DECODE')
    # CheckTileDecodeBenchmark(config, 'CHECK_RESULTS')
