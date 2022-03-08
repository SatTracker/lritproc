import json
from os import PathLike
from typing import Union


class JSONManager:
    def __init__(self, path: Union[str, bytes, PathLike[str], PathLike[bytes], int], encoder: json.JSONEncoder = None):
        if str(path)[-5:].lower != '.json':
            self._path = f'{path}.json'
        else:
            self._path = path
        self._enc = json.JSONEncoder(sort_keys=True, indent=4) if encoder is None else encoder
        self._data = None

    def load(self) -> None:
        self._data = json.loads("\n".join(open(self._path, 'r').readlines()))
        return

    def save(self) -> None:
        open(self._path, 'w').write(self._enc.encode(self._data))
        return

    def get(self, key: str) -> Union[str, int, bool, list, tuple, dict, None]:
        if key in self._data:
            return self._data[key]
        else:
            return None

    def set(self, key: str, data: Union[str, int, bool, list, tuple, dict, None]) -> None:
        self._data[key] = data
        return

    def overwrite(self, data: Union[str, int, bool, list, tuple, dict, None]) -> None:
        self._data = data

    def delete(self, key: str) -> None:
        if key in self._data[key]:
            del self._data[key]
        return

    def has(self, key: str) -> bool:
        if key in self._data and self._data[key] is not None:
            return True
        return False

    def getKeys(self) -> list[str]:
        return self._data.keys()

    def hasChanged(self) -> bool:
        data = "\r".join(open(self._path).readlines())
        return data == self._enc.encode(self._data)
