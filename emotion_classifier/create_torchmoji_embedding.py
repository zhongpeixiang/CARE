# -*- coding: utf-8 -*-

""" Use torchMoji to score texts for emoji distribution.

The resulting emoji ids (0-63) correspond to the mapping
in emoji_overview.png file at the root of the torchMoji repo.

Writes the result to a csv file.
"""
from __future__ import print_function, division, unicode_literals
import sys
import json
import pickle
import argparse
import logging
from tqdm import trange
import numpy as np

# sys.path.append("..") # Adds higher directory to python modules path.

from torchmoji.sentence_tokenizer import SentenceTokenizer
from torchmoji.model_def import torchmoji_feature_encoding
from torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH


def torchmoji_embedding(model, st, sentences, chunksize=5000):
    print("computing emoji embedding...")
    print("sample sentences: ")
    print(sentences[:3])
    total_embedding = []
    for i in trange(0, len(sentences), chunksize):
        tokenized, _, _ = st.tokenize_sentences(sentences[i:i+chunksize])
        # print(tokenized[:3])
        total_embedding.extend(model(tokenized))
    print("embedidng length: ", len(total_embedding))
    return total_embedding

def load_data(path):
    with open(path, "r") as f:
        return f.read().splitlines()

if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO, \
                    format = '%(asctime)s  %(levelname)-5s %(message)s', \
                    datefmt =  "%Y-%m-%d-%H-%M-%S")

    maxlen = 30

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="Reddit")
    parser.add_argument('--smaller', action="store_true")
    parser.add_argument('--path', type=str, default="")

    args = parser.parse_args()
    dataset = args.dataset
    smaller = args.smaller
    path = args.path

    print("Loading torchmoji vocab and model...")
    with open(VOCAB_PATH, 'r') as f:
        vocabulary = json.load(f)
    st = SentenceTokenizer(vocabulary, maxlen)
    emoji_model = torchmoji_feature_encoding(PRETRAINED_PATH)

    # load data
    if path=="":
        for split in ["train", "valid", "test"]:
            print("*"*30+split+"*"*30)
            load_path = "./data/{0}/{1}-tgt.txt".format(dataset, split)
            save_path="./data/{0}/{1}-tgt-embedding.npy".format(dataset, split)
            if smaller:
                load_path = "./data/{0}/{1}-tgt-smaller.txt".format(dataset, split)
                save_path="./data/{0}/{1}-tgt-smaller-embedding.npy".format(dataset, split)
            
            print("Loading data from {0}".format(load_path))
            data = load_data(load_path)
            
            print("Computing or loading torchmoji embedding...")        
            embedding = torchmoji_embedding(emoji_model, st, data) # (num_examples, feature_dim)
            
            # pickle.dump(embedding, open(save_path, "wb"))
            print("Saving embedding to {0}".format(save_path))
            np.save(save_path, embedding)
    else:
        # create embeddings for the file
        print("Creating embedding for {0}...".format(path))
        load_path = path
        save_path = path.replace(".txt", "-torchmoji.npy")
        
        print("Loading data from {0}".format(load_path))
        data = load_data(load_path)
        
        print("Computing or loading torchmoji embedding...")        
        embedding = torchmoji_embedding(emoji_model, st, data) # (num_examples, feature_dim)
        
        # pickle.dump(embedding, open(save_path, "wb"))
        print("Saving embedding to {0}".format(save_path))
        np.save(save_path, embedding)