import os
import pathlib


def get_relative_path(this, path):
    return os.path.normpath(os.path.abspath(os.path.join(
        pathlib.Path(this).parent.absolute(), path)
    ))


def path_join(*args):
    args = list(args)
    if len(args) == 0:
        return None
    elif len(args) == 1:
        return args[0]
    else:
        path = args.pop(0)
        for i in args:
            path = os.path.normpath(os.path.abspath(os.path.join(path, i)))
        return path
