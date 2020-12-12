#!/usr/bin/env python2.7
#
# thesaurus-lookup
#
# Author: Anupam Sengupta (anupam@Anupams-MacBook-Pro.local.)
#
# Copyright (C) 2014
#
# Released under the BSD license.
#
# This script provides a list of thesaurus entries from the webservice
# at <http://thesaurus.altervista.org>
#
# You will need to get a personal API key from:
# <http://thesaurus.altervista.org/mykey>
#

import sys
import string
import pickle
import json
import argparse
from collections import Counter, defaultdict
from tqdm import tqdm
import numpy as np
import torch
from pymagnitude import Magnitude

from util import load_pickle, save_pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_path', type=str, required=True)
    parser.add_argument('--embedding_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()

    vocab_path = args.vocab_path
    embedding_path = args.embedding_path
    output_path = args.output_path

    print("Loading vocab...")
    vocab = torch.load(vocab_path)
    word2id = vocab['src'].base_field.vocab.stoi
    id2word = vocab['src'].base_field.vocab.itos
    print("vocab size: {0}".format(len(word2id)))

    print("Loading magnitude...")
    word_vectors = Magnitude(embedding_path)
    dim = len(word_vectors.query(id2word[0]))

    print("Building vocab embedding...")
    vocab_embedding = torch.zeros((len(word2id), dim))
    for w, _id in tqdm(word2id.items()):
        vocab_embedding[_id] = torch.from_numpy(word_vectors.query(w))

    # save vocab embedding
    print("Saving vocab embedding...")
    torch.save(vocab_embedding, output_path)