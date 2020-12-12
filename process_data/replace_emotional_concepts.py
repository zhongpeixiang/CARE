# -*- coding: utf-8 -*-
import os
import time
import torch
import random
import argparse
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from util import load_file, load_pickle, save_pickle, get_all_entities, get_concepts

id2emotion = ["neutral", "joy", "sadness", "surprise", "fear", "anger"]
def read_file(path):
    with open(path, "r") as f:
        data = [int(line.strip("\n")) for line in f.readlines()]
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--emotion_file', type=str, required=True)
    parser.add_argument('--emotion_lexicon_file', type=str, required=True)
    # parser.add_argument('--conceptnet_path', type=str, required=True)
    parser.add_argument('--dataset_vocab_path', type=str, required=True)
    parser.add_argument('--dataset_vocab_embedding_path', type=str, required=True)
    parser.add_argument('--top_k', type=int, default=100)

    args = parser.parse_args()
    input_file = args.input_file
    emotion_file = args.emotion_file
    emotion_lexicon_file = args.emotion_lexicon_file
    # conceptnet_path = args.conceptnet_path
    dataset_vocab_path = args.dataset_vocab_path
    dataset_vocab_embedding_path = args.dataset_vocab_embedding_path
    top_k = args.top_k

    num_emotional_words = top_k//4

    concept_words = load_pickle(input_file) # tuples of np.array
    emotions = read_file(emotion_file)
    # CN = load_pickle(conceptnet_path)
    emotion_lexicon = load_pickle(emotion_lexicon_file)
    # vocab_embedding = torch.load(dataset_vocab_embedding_path) # (vocab, emb_dim)

    # load vocab
    print("Loading dataset vocab from ", dataset_vocab_path)
    vocab_ckpt = torch.load(dataset_vocab_path)
    word2id = vocab_ckpt["src"].base_field.vocab.stoi
    id2word = vocab_ckpt["src"].base_field.vocab.itos
    print("dataset vocab size: ", len(word2id))

    new_word_ids = []
    new_word_scores = []
    new_word_VAD_scores = []
    for sent_word_ids, sent_word_scores, sent_word_VAD_scores, emotion in tqdm(zip(concept_words[0], concept_words[1], concept_words[2], emotions), total=len(concept_words[0])):
        emotional_words = []
        while len(emotional_words) < num_emotional_words:
            w = random.choice(emotion_lexicon[id2emotion[emotion]])
            if w in word2id and w not in emotional_words:
                emotional_words.append(word2id[w])
        sent_word_ids[:num_emotional_words] = np.array(emotional_words, dtype=np.int)
        sent_word_scores[:num_emotional_words] = sent_word_scores[num_emotional_words:].mean()
        sent_word_VAD_scores[:num_emotional_words] = sent_word_VAD_scores[num_emotional_words:].mean()
        new_word_ids.append(sent_word_ids)
        new_word_scores.append(sent_word_scores)
        new_word_VAD_scores.append(sent_word_VAD_scores)
    

    save_path = input_file.replace(".pkl", "-replaced.pkl")
    print("saving file to ", save_path)
    save_pickle((np.stack(new_word_ids, axis=0), np.stack(new_word_scores, axis=0), np.stack(new_word_VAD_scores, axis=0)), save_path)


