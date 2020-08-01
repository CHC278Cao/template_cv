# encoding: utf-8
"""
@author: ccj
@contact:
"""

import time


def init_logger(cfg):
    from logging import getLogger, DEBUG, FileHandler, Formatter, StreamHandler

    log_format = '%(asctime)s %(levelname)s %(message)s'

    stream_handler = StreamHandler()
    stream_handler.setLevel(DEBUG)
    stream_handler.setFormatter(Formatter(log_format))

    file_handler = FileHandler(cfg.log_file)
    file_handler.setFormatter(Formatter(log_format))

    logger = getLogger(cfg.project_name)
    logger.setLevel(DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


