# encoding: utf-8
"""
@author: ccj
@contact:
"""


def reduce_fn(vals):
    """
        Reduce function to average values when using TPU
    :param vals: value
    :return:
        the average value
    """
    return sum(vals) / len(vals)