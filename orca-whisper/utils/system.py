import os
from typing import List


def make_folder(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def make_folders(paths: List[str]):
    for path in paths:
        make_folder(path)