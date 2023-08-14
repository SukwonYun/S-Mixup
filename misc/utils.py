import os
import numpy as np
import pickle
import gzip
from glob import glob
from enum import Enum, auto
import logging

_os_cwd = os.path.realpath('.')

class DIR_TYPE(Enum):
    data_dir = auto()
    rst_dir = auto()
    tensor_logdir = auto()

def get_dir(dir_type:DIR_TYPE, dataname=None):
    if DIR_TYPE.data_dir == dir_type:
        assert isinstance(dataname, str)
        path = os.path.join(_work_dir(), _work_dir(), 'data', dataname)
    elif DIR_TYPE.rst_dir == dir_type:
        path = os.path.join(_work_dir(), _work_dir(), 'rst')
    elif DIR_TYPE.tensor_logdir == dir_type:
        path = os.path.join(_work_dir(), _work_dir(), 'rst', 'logdir')
    
    return path

def _work_dir():
    work_dir = _os_cwd
    return work_dir

def delete_files(dir_type:DIR_TYPE, file_reg=''):
    path = get_dir(dir_type)
    for fd in glob(os.path.join(path, ''.join(['*', file_reg, '*']))):
        if os.path.isfile(fd):
            os.remove(fd)
        elif os.path.isdir(fd):
            for file in glob(os.path.join(fd, '*')):
                os.remove(file)
            os.rmdir(fd)

def save_pickle(obj, name='test.pickle', path=_work_dir()):
    with gzip.open(os.path.join(path, name), 'wb') as f:
        print(f'-----SAVE {f.name}')
        pickle.dump(obj, f)

def load_pickle(name='test.pickle', path=_work_dir()):
    with gzip.open(os.path.join(path, name), 'rb') as f:
        data = pickle.load(f)
    return data

def get_logger(args):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s\t%(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logs_path = f'./logs/{args.dataset}'
    os.makedirs(logs_path, exist_ok=True)

    n_runs = args.n_runs
    file_handler = logging.FileHandler(os.path.join(logs_path, f'n_runs_{n_runs}.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f'{args.dataset}\t-------------------------------------------------')
    logger.info(args)

    return logger