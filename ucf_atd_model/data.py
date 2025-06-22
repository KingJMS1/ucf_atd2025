import os
from typing import *

def data_loc(filename: str) -> str:
    path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), "datasets", filename))
    if os.path.isfile(path):
        return path
    else:
        return filename

class ResultCache:
    def __init__(self, caller: str):
        self.caller = caller
        self.cachePath = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), "cache", caller))

    def test_cache(self, filename: str) -> Tuple[bool, str]:
        filePath = os.path.join(self.cachePath, filename)
        return os.path.isfile(filePath), filePath
