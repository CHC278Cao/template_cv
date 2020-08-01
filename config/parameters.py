# encoding: utf-8
"""
@author: ccj
@contact:
"""

import argparse


def get_args(**kwargs):
    """

    :param kwargs:
    :return:
    """
    args = kwargs
    parser = argparse.ArgumentParser(description="Train models on images for classification",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # dictionary and file
    # parser.add_argument('--img_dir', '--image-dir', metavar='')

    parser.add_argument('--lr', '--learning-rate', metavar='LR', type=float, nargs='?',
                        default=0.005, help='Learning rate', dest='learning_rate')
    parser.add_argument()
    config = vars(parser.parse_args())
    args.update(config)

    return dict(args)
