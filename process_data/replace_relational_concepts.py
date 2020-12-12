# -*- coding: utf-8 -*-
import os
import time
import torch
import random
import argparse
from collections import defaultdict, Counter
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
    parser.add_argument('--src_concept_file', type=str, required=True)
    parser.add_argument('--conceptnet_path', type=str, required=True)
    parser.add_argument('--dataset_vocab_path', type=str, required=True)
    # parser.add_argument('--dataset_vocab_embedding_path', type=str, required=True)
    parser.add_argument('--top_k', type=int, default=100)

    args = parser.parse_args()
    input_file = args.input_file
    emotion_file = args.emotion_file
    emotion_lexicon_file = args.emotion_lexicon_file
    src_concept_file = args.src_concept_file
    conceptnet_path = args.conceptnet_path
    dataset_vocab_path = args.dataset_vocab_path
    # dataset_vocab_embedding_path = args.dataset_vocab_embedding_path
    top_k = args.top_k

    num_unemotional_words = top_k

    concept_words = load_pickle(input_file) # tuples of np.array
    emotions = read_file(emotion_file)
    # emotion_lexicon = load_pickle(emotion_lexicon_file)
    CN = load_pickle(conceptnet_path) # the augmented conceptnet
    associated_concepts = defaultdict(list)
    for h,r,t,w in tqdm(CN):
        associated_concepts[h].append((r,t,w))
    # vocab_embedding = torch.load(dataset_vocab_embedding_path) # (vocab, emb_dim)

    # load vocab
    print("Loading dataset vocab from ", dataset_vocab_path)
    vocab_ckpt = torch.load(dataset_vocab_path)
    word2id = vocab_ckpt["src"].base_field.vocab.stoi
    id2word = vocab_ckpt["src"].base_field.vocab.itos
    print("dataset vocab size: ", len(word2id))
    all_words = list(word2id.keys())

    src_concepts = load_file(src_concept_file)

    new_word_ids = []
    new_word_scores = []
    new_word_VAD_scores = []
    unemotional_words_count = []
    for sent_word_ids, sent_word_scores, sent_word_VAD_scores, concepts, emotion in \
        tqdm(zip(concept_words[0], concept_words[1], concept_words[2], src_concepts, emotions), total=len(concept_words[0])):
        
        qualified = []
        emotions_to_exclude = [e for e in id2emotion if e != id2emotion[emotion]]
        neighbor_hood_size = 1
        while len(set([t for r,t,w in qualified])) < num_unemotional_words:
            if neighbor_hood_size == 4:
                # print("only {0} concepts".format(len(set([t for r,t,w in qualified]))))
                break
            unemotional_words = []
            for concept in concepts:
                if concept in associated_concepts:
                    unemotional_words.extend(associated_concepts[concept])
            unemotional_words = sorted(unemotional_words, key=lambda x: x[-1], reverse=True)
            for r,t,w in unemotional_words:
                if t in word2id and r not in emotions_to_exclude:
                    qualified.append((r,t,w))
            concepts = [t for r,t,w in unemotional_words]
            neighbor_hood_size += 1
            
        unemotional_words = []
        for r,t,w in qualified:
            if word2id[t] not in unemotional_words:
                unemotional_words.append(word2id[t])
            if len(unemotional_words) == num_unemotional_words:
                break
        unemotional_words_count.append(len(unemotional_words))
        
        random_concepts = []
        num_random_words = top_k - len(unemotional_words)
        while len(random_concepts) < num_random_words:
            w = random.choice(all_words)
            if w in word2id and w not in random_concepts and w not in unemotional_words:
                random_concepts.append(word2id[w])
        
        if (len(random_concepts) + len(unemotional_words)) != top_k:
            print(len(random_concepts), len(unemotional_words))
        
        new_word_ids.append(np.array(random_concepts+list(unemotional_words), dtype=np.int))
        new_word_scores.append(np.ones(top_k, dtype=np.float32))
        new_word_VAD_scores.append(np.ones(top_k, dtype=np.float32))
    
    print(Counter(unemotional_words_count).most_common(10))
    save_path = input_file.replace(".pkl", "-graphsearch-full.pkl")
    print("saving file to ", save_path)
    save_pickle((np.stack(new_word_ids, axis=0), np.stack(new_word_scores, axis=0), np.stack(new_word_VAD_scores, axis=0)), save_path)


