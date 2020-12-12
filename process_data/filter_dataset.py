# -*- coding: utf-8 -*-

from __future__ import print_function, division, unicode_literals
import sys
import json
import pickle
import argparse
import logging
from tqdm import trange
import numpy as np


def load_data(path):
    with open(path, "r") as f:
        return f.read().splitlines()

if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO, \
                    format = '%(asctime)s  %(levelname)-5s %(message)s', \
                    datefmt =  "%Y-%m-%d-%H-%M-%S")

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--smaller', action="store_true")
    parser.add_argument('--min_len', type=int, required=True)

    args = parser.parse_args()
    dataset = args.dataset
    smaller = args.smaller
    min_len = args.min_len

    # load data
    for split in ["train", "valid", "test"]:
        print("*"*30+split+"*"*30)
        src_path = "./data/{0}/{1}-src.txt".format(dataset, split)
        tgt_path = "./data/{0}/{1}-tgt.txt".format(dataset, split)
        src_path_save = "./data/{0}/{1}-src-filtered-min{2}.txt".format(dataset, split, min_len)
        tgt_path_save = "./data/{0}/{1}-tgt-filtered-min{2}.txt".format(dataset, split, min_len)
        if smaller:
            src_path = "./data/{0}/{1}-src-smaller.txt".format(dataset, split)
            tgt_path = "./data/{0}/{1}-tgt-smaller.txt".format(dataset, split)
            src_path_save = "./data/{0}/{1}-src-smaller-filtered-min{2}.txt".format(dataset, split, min_len)
            tgt_path_save = "./data/{0}/{1}-tgt-smaller-filtered-min{2}.txt".format(dataset, split, min_len)
        
        print("Loading data from {0}".format(src_path))
        src = load_data(src_path)
        tgt = load_data(tgt_path)
        
        # filter out empty src or tgt
        f_src = open(src_path, 'w')
        f_tgt = open(tgt_path, 'w')
        
        for src_sent, tgt_sent in zip(src, tgt):
            if len(src_sent.split()) >= min_len and len(tgt_sent.split()) >= min_len:
                f_src.write(src_sent + "\n")
                f_tgt.write(tgt_sent + "\n")