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
from util import load_pickle, save_pickle, load_file, load_vocab, get_punctuation_words, get_digit_words

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

def get_datamuse_synonyms(datamuse, word):
    if word in datamuse:
        return set(datamuse[word])
    else:
        return set()

def get_altervista_synonyms(altervista, word):
    if word in altervista:
        return set(altervista[word])
    else:
        return set()

def get_synonyms(word, expanded=True):
    synonyms = set()
    if expanded:
        synonyms = synonyms.union(get_wordnet_synonyms(wordnet, word))
        synonyms = synonyms.union(get_conceptnet_synonyms(conceptnet_synonyms, word))
        synonyms = synonyms.union(get_senticnet_synonyms(senticnet, word))
        synonyms = synonyms.union(get_datamuse_synonyms(datamuse, word))
        synonyms = synonyms.union(get_altervista_synonyms(altervista, word))
    return synonyms

def has_VAD(k, VAD, expanded=True):
    if k in VAD or porter.stem(k) in VAD or any([syn in VAD for syn in get_synonyms(k, expanded)]) or any([syn in VAD for syn in get_synonyms(porter.stem(k), expanded)]):
        return True
    return False


def VAD_coverage(VAD, vocab, filtered=True, expanded=True):
    # vocab: a list of (word, cnt)
    unique_coverage = 0
    token_coverage = 0
    
    if filtered:
        # remove punctuations
        keys_to_remove = get_punctuation_words(vocab)
        # remove number
        keys_to_remove += get_digit_words(vocab)
        filtered_vocab = [(w,cnt) for w,cnt in vocab if w not in keys_to_remove]
    else:
        filtered_vocab = vocab

    for k,cnt in filtered_vocab:
        if has_VAD(k, VAD, expanded):
            unique_coverage += 1
            token_coverage += cnt
    print_str = "filtered: {0}, expanded: {1}, unique coverage: {2:.4f}, total coverage: {3:.4f}"
    print(print_str.format(filtered, expanded, unique_coverage/len(filtered_vocab), token_coverage/sum(cnt for k,cnt in filtered_vocab)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    # parser.add_argument('--vocab_size', type=int, default=30000)
    parser.add_argument('--smaller', action="store_true")
    args = parser.parse_args()

    dataset = args.dataset
    # vocab_size = args.vocab_size
    smaller = args.smaller

    vocab, vocab_filtered = load_vocab(dataset, smaller)
    # # load dataset
    # if dataset.lower() == "reddit":
    #     dataset_path = "./data/Reddit/train-{0}.txt"
    # elif dataset.lower() == "twitter":
    #     dataset_path = "./data/Twitter/train-{0}.txt"
    # print("Loading dataset...")
    # src = load_file(dataset_path.format("src"))
    # tgt = load_file(dataset_path.format("tgt"))
    # data = src + tgt
    # print("number of sentence pairs: ", len(src))

    # # vocab
    # print("Building vocab...")
    # vocab = Counter([w for l in data for w in l])
    # print("vocab size: {0}".format(len(vocab)))
    # vocab_filtered = vocab.most_common(vocab_size)
    # coverage = sum(cnt for k,cnt in vocab_filtered)/sum(vocab.values())
    # print("filtered vocab size: {0}, coverage: {1:.4f}".format(vocab_size, coverage))

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

    # load datamuse
    datamuse_raw = load_pickle("./data/KB/VAD/{0}{1}_not_found_v2_{2}_raw.pkl".format(dataset.lower(), "-smaller" if smaller else "", "datamuse"))
    datamuse = defaultdict(list)
    for k,v in datamuse_raw.items():
        if len(v) != 0:
            for syn in v:
                datamuse[k].append(syn["word"])
    print("datamuse size: {0}".format(len([1 for k,v in datamuse.items() if len(v) > 0])))

    # load altervista
    altervista_raw = load_pickle("./data/KB/VAD/{0}{1}_not_found_v2_{2}_raw.pkl".format(dataset.lower(), "-smaller" if smaller else "", "altervista"))
    altervista = {}
    for k in altervista_raw:
        syns = []
        for e in altervista_raw[k]["response"]:
            terms = e["list"]['synonyms'].split("|")
            for term in terms:
                term = term.lower()
                if "term" not in term and "antonym" not in term and "related" not in term:
                    syns.append(term)
                elif "(similar term)" in term:
                    syns.append(term.replace(" (similar term)", ""))
                elif "(generic term)" in term:
                    syns.append(term.replace(" (generic term)", ""))
        altervista[k] = set(syns)
    print("altervista size: {0}".format(len(altervista)))
    
    # coverage stats
    VAD_coverage(VAD, vocab_filtered, filtered=False, expanded=False)
    VAD_coverage(VAD, vocab_filtered, filtered=True, expanded=False)
    VAD_coverage(VAD, vocab_filtered, filtered=False, expanded=True)
    VAD_coverage(VAD, vocab_filtered, filtered=True, expanded=True)

    # expand VAD
    print("Expanding VAD...")
    for w,cnt in vocab_filtered:
        if w not in VAD and porter.stem(w) not in VAD:
            syns = get_synonyms(w, expanded=True).union(get_synonyms(porter.stem(w), expanded=True))
            syns_VAD = [VAD[w] for w in syns if w in VAD]
            syns_VAD += [VAD[porter.stem(w)] for w in syns if porter.stem(w) in VAD]
            if len(syns_VAD) > 0:
                mean_VAD = np.array(syns_VAD).mean(axis=0)
                VAD[w] = tuple(mean_VAD)
    
    # expanded coverage stats
    VAD_coverage(VAD, vocab_filtered, filtered=False, expanded=False)
    VAD_coverage(VAD, vocab_filtered, filtered=True, expanded=False)
    VAD_coverage(VAD, vocab_filtered, filtered=False, expanded=True)
    VAD_coverage(VAD, vocab_filtered, filtered=True, expanded=True)

    # save expanded VAD
    print("Saving expanded VAD...")
    save_pickle(VAD, "./data/KB/VAD/expanded_VAD_{0}{1}_v2.pkl".format(dataset.lower(), "-smaller" if smaller else ""))