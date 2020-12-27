

def splitx(string: str) -> tuple:
    return tuple(map(int, string.split('x')))


class AutoDict(dict):
    def __init__(self, return_type='dict'):
        super().__init__()
        self.return_type = return_type

    def __missing__(self, key):
        value = self[key] = eval(f'{self.return_type}()')
        return value


