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
    # parser.add_argument('--emotion_file', type=str, required=True)
    parser.add_argument('--conceptnet_path', type=str, required=True)
    parser.add_argument('--dataset_vocab_path', type=str, required=True)
    parser.add_argument('--top_k', type=int, default=100)

    args = parser.parse_args()
    input_file = args.input_file
    # emotion_file = args.emotion_file
    conceptnet_path = args.conceptnet_path
    dataset_vocab_path = args.dataset_vocab_path
    top_k = args.top_k

    concept_words = load_pickle(input_file)
    # emotions = read_file(emotion_file)
    CN = load_pickle(conceptnet_path)

    # load vocab
    print("Loading dataset vocab from ", dataset_vocab_path)
    vocab_ckpt = torch.load(dataset_vocab_path)
    word2id = vocab_ckpt["src"].base_field.vocab.stoi
    id2word = vocab_ckpt["src"].base_field.vocab.itos
    print("dataset vocab size: ", len(word2id))

    associated_concepts = defaultdict(list)
    for h,r,t,w in tqdm(CN):
        associated_concepts[h].append((r,t,w))
    
    # to clean concept words and save as np.array to save space
    cleaned_concept_words = []
    sent_words_array = []
    sent_scores_array = []
    sent_VAD_scores_array = []
    # for (sent_words, sent_scores, sent_VAD_scores, sent_relations), emotion in tqdm(zip(concept_words, emotions)):
    for sent_words, sent_scores, sent_VAD_scores, sent_relations in tqdm(concept_words):
        if len(sent_words) == top_k:
            sent_words_array.append(sent_words)
            sent_scores_array.append(sent_scores)
            sent_VAD_scores_array.append(sent_VAD_scores)
            # cleaned_concept_words.append((np.array(sent_words, dtype=np.int), np.array(sent_scores, dtype=np.float32), np.array(sent_VAD_scores, dtype=np.float32)))
            continue
        for w,score,VAD_score,relation in zip(sent_words, sent_scores, sent_VAD_scores, sent_relations):
            for r,t,w in associated_concepts[id2word[w]]:
                if t in word2id and t not in sent_words:
                    sent_words.append(w)
                    sent_scores.append(score)
                    sent_VAD_scores.append(VAD_score)
                    sent_relations.append(relation)
                    if len(sent_words) == top_k:
                        break
            if len(sent_words) == top_k:
                break
        sent_words_array.append(sent_words)
        sent_scores_array.append(sent_scores)
        sent_VAD_scores_array.append(sent_VAD_scores)
        # cleaned_concept_words.append((np.array(sent_words, dtype=np.int), np.array(sent_scores, dtype=np.float32), np.array(sent_VAD_scores, dtype=np.float32)))

    save_path = input_file.replace(".pkl", "-cleaned.pkl")
    print("saving file to ", save_path)
    save_pickle((np.array(sent_words_array, dtype=np.int), np.array(sent_scores_array, dtype=np.float32), np.array(sent_VAD_scores_array, dtype=np.float32)), save_path)
    # save_pickle(cleaned_concept_words, save_path)

