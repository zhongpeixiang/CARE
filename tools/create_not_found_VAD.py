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
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet
import numpy as np
import torch
from util import load_file, load_pickle, save_pickle, load_vocab, get_punctuation_words, get_digit_words


def get_wordnet_synonyms(wordnet, word):
    synonyms = []
    for ss in wordnet.synsets(word):
        synonyms.extend([lemma.lower() for lemma in ss.lemma_names()])
    return set(synonyms) - {word}

def get_conceptnet_synonyms(conceptnet_synonyms, word):
    return set(conceptnet_synonyms[word])

def get_senticnet_synonyms(senticnet, word):
    if word in senticnet:
        return set(senticnet[word][-5:])
    else:
        return set()

def get_synonyms(word, expanded=True):
    synonyms = set()
    if expanded:
        synonyms = synonyms.union(get_wordnet_synonyms(wordnet, word))
        synonyms = synonyms.union(get_conceptnet_synonyms(conceptnet_synonyms, word))
        synonyms = synonyms.union(get_senticnet_synonyms(senticnet, word))
    return synonyms

def has_VAD(k, VAD, expanded=True):
    if k in VAD or porter.stem(k) in VAD or any([syn in VAD for syn in get_synonyms(k, expanded)]) or any([syn in VAD for syn in get_synonyms(porter.stem(k), expanded)]):
        return True
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--smaller', action="store_true")
    args = parser.parse_args()

    dataset = args.dataset
    smaller = args.smaller

    # load voacb
    vocab, vocab_filtered = load_vocab(dataset, smaller)
    
    print("Loading dictionaries...")
    # load VAD
    VAD = load_pickle("./data/KB/VAD/VAD.pkl")
    porter = PorterStemmer()
    
    # load ConceptNet
    conceptnet = load_pickle("./data/KB/ConceptNet/conceptnet_en.pkl")
    # get all synonyms
    conceptnet_synonyms = defaultdict(list)
    for h, r, t, w in conceptnet:
        if r in ["Synonym", "SimilarTo", "FormOf", "DerivedFrom"]:
            conceptnet_synonyms[h].append(t)
            conceptnet_synonyms[t].append(h)
    
    # load senticnet
    senticnet = load_pickle("./data/KB/SenticNet/senticnet.pkl")

    not_found = []
    for k,cnt in vocab_filtered:
        if not has_VAD(k, VAD, expanded=True):
            not_found.append((k, cnt))
    print("Number of not founds: {0}".format(len(not_found)))

    # save expanded VAD
    print("Saving not found vocab...")
    save_pickle(not_found, "./data/KB/VAD/{0}{1}_not_found_v2.pkl".format(dataset.lower(), "-smaller" if smaller else ""))