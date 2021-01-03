from dectime.dectime import Dectime, Role, CheckDectime


def main():
    config = f'config/config_user_dectime_28videos_nas.json'

    tile_dectime = Dectime(config)

    tile_dectime.run(Role.PREPARE)
    tile_dectime.run(Role.COMPRESS)
    tile_dectime.run(Role.SEGMENT)
    for _ in range(tile_dectime.config.decode_num):
        tile_dectime.run(Role.DECODE)
    tile_dectime.run(Role.RESULTS)

    # CheckDectime(
    #     config_file=f'config_file/config_user_dectime_28videos_nas.json',
    #     automate=True)


if __name__ == '__main__':
    main()
