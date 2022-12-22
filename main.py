#!/usr/bin/env python3
import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

from dataset import Dataset
from log import Log
from model import Model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rebuild', action='store_true', help='rebuild the model from scratch')
    args = parser.parse_args()

    tf.config.threading.set_intra_op_parallelism_threads(12)
    tf.config.threading.set_inter_op_parallelism_threads(12)

    dataset = Dataset(rebuild=args.rebuild)
    log = Log(rebuild=args.rebuild)
    model = Model(dataset=dataset, log=log, rebuild=args.rebuild)

    model.train(epochs=2)  # TODO: epochs from args
    model.generate_text(seed_text='защо ', length=256)  # TODO: seed_text from args

    log.plot()
