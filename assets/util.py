import json
import subprocess
from logging import warning, info, debug, critical
from typing import Any, Dict, Hashable, Iterable, NamedTuple, Tuple, Union
from pathlib import Path
import numpy as np
from pathlib import Path


class AutoDict(dict):
    def __missing__(self, key):
        self[key] = type(self)()
        return self[key]


class ConfigBase:
    _config_data: dict = {}

    def load_config(self, config_file: Union[Path, str]):
        with open(config_file, 'r') as f:
            self._config_data.update(json.load(f))

        for key in self._config_data:
            value = self._config_data[key]
            setattr(self, key, value)


def run_command(command: str, log_to_save: Union[str, Path], mode: str = 'w'):
    """
    Run a shell command with subprocess module with realtime output.
    :param command: A command string to run.
    :param log_to_save: A path-like to save the process output.
    :param mode: The write mode: 'w' or 'a'.
    :return: stdout.
    """
    info(command)

    with open(log_to_save, mode, encoding='utf-8') as f:
        f.write(f'{command}\n')
        subprocess.run(command, shell=True, stdout=f,
                       stderr=subprocess.STDOUT, encoding='utf-8')


def save_json(data: dict, filename, compact=False):
    if compact:
        separators = (',', ':')
        indent = None
    else:
        separators = None
        indent = 2

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, separators=separators, indent=indent)


def splitx(string: str) -> tuple:
    return tuple(map(int, string.split('x')))


def update_dictionary(value, dictionary: AutoDict, key1: Hashable = None,
                      key2: Hashable = None, key3: Hashable = None,
                      key4: Hashable = None, key5: Hashable = None):
    dict_ = dictionary
    if key1:
        if key2: dict_ = dict_[key1]
        else: dict_[key1] = value
    if key2:
        if key3: dict_ = dict_[key2]
        else: dict_[key2] = value
    if key3:
        if key4: dict_ = dict_[key3]
        else: dict_[key3] = value
    if key4:
        if key5: dict_ = dict_[key4]
        else: dict_[key4] = value
    if key5:
        dict_[key5] = value

    return dictionary


def dishevel_dictionary(dictionary: dict, key1: Hashable = None,
                        key2: Hashable = None, key3: Hashable = None,
                        key4: Hashable = None, key5: Hashable = None) -> Any:
    disheveled_dictionary = dictionary
    if key1: disheveled_dictionary = disheveled_dictionary[key1]
    if key2: disheveled_dictionary = disheveled_dictionary[key2]
    if key3: disheveled_dictionary = disheveled_dictionary[key3]
    if key4: disheveled_dictionary = disheveled_dictionary[key4]
    if key5: disheveled_dictionary = disheveled_dictionary[key5]
    return disheveled_dictionary


def menu(options_dict: Dict[int, Any]):
    options = []
    text = f'Options:\n'
    for idx in options_dict:
        text += f'{idx} - {options_dict[idx]}\n'
        options.append(idx)
    text += f': '

    c = None
    while c not in options:
        c = input(text)
    return c


class StatsData(NamedTuple):
    average: float = None
    std: float = None
    correlation: float = None
    min: float = None
    quartile1: float = None
    median: float = None
    quartile3: float = None
    max: float = None


def calc_stats(data1: Iterable, data2: Iterable) \
        -> Tuple[StatsData, StatsData]:
    # Percentiles & Correlation
    per = [0, 25, 50, 75, 100]
    corr = np.corrcoef((data1, data2))[1][0]

    # Struct statistics results
    percentile_data1 = np.percentile(data1, per).T
    stats_data1 = StatsData(average=np.average(data1),
                            std=float(np.std(data1)),
                            correlation=corr,
                            min=percentile_data1[0],
                            quartile1=percentile_data1[1],
                            median=percentile_data1[2],
                            quartile3=percentile_data1[3],
                            max=percentile_data1[4],
                            )

    percentile_data2 = np.percentile(data2, per).T
    stats_data2 = StatsData(average=np.average(data2),
                            std=float(np.std(data2)),
                            correlation=corr,
                            min=percentile_data2[0],
                            quartile1=percentile_data2[1],
                            median=percentile_data2[2],
                            quartile3=percentile_data2[3],
                            max=percentile_data2[4],
                            )

    return stats_data1, stats_data2


def grouper(iterable, n, fillvalue=None):
    """
    Collect data into fixed-length chunks or blocks.
    Usage: grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx

    :param iterable: A iterable. ;-)
    :param n: The length of blocks.
    :param fillvalue: If "len(iterable)" is not multiple of "n", fill with this.
    :return: A tuple with "n" elements of "iterable".
    """
    from itertools import zip_longest
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)
