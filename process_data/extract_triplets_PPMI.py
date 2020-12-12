# -*- coding: utf-8 -*-
import math
import pickle
import argparse
from collections import Counter, defaultdict
from functools import partial
import itertools
from tqdm import tqdm
import numpy as np
from util import load_file, load_pickle, save_pickle, get_all_entities, get_ngrams, is_a_ngram_stopword, get_concepts

emotion2id = {
    "neutral": 0,
    "joy": 1,
    "sadness": 2,
    "surprise": 3,
    "fear": 4,
    "anger": 5
}
id2emotion = ["neutral", "joy", "sadness", "surprise", "fear", "anger"]

def get_concept_pairs(src, tgt, tgt_emotions, CN_concepts, stopwords):
    concept_pairs_by_emotion = defaultdict(list)
    concept_pairs_counter_by_emotion = {}
    dataset_concept_pairs = []
    
    for msg, res, emo in tqdm(zip(src, tgt, tgt_emotions)):
        msg_concepts = get_concepts(msg, CN_concepts, stopwords)
        res_concepts = get_concepts(res, CN_concepts, stopwords)
        if len(msg_concepts) > 0 and len(res_concepts) > 0:
            concept_pairs = list(itertools.product(msg_concepts, res_concepts))
            concept_pairs_by_emotion[emo].extend(concept_pairs)
            dataset_concept_pairs.append(concept_pairs)
    for emo in concept_pairs_by_emotion:
        concept_pairs_counter_by_emotion[emo] = Counter(concept_pairs_by_emotion[emo])
    return dataset_concept_pairs, concept_pairs_counter_by_emotion


def combine_emotion_counters(counter_by_emotion):
    total = Counter()
    for v in counter_by_emotion.values():
        total = total + v
    return total

def filter_emotion_counters(counter_by_emotion, concept_pairs_counter, threshold):
    print("filtering concept pairs from dataset...")
    filtered_counter_by_emotion = {}
    for emo in counter_by_emotion:
        filtered_counter_by_emotion[emo] = Counter({k:v for k,v in counter_by_emotion[emo].items() if concept_pairs_counter[k] >= threshold})
    return filtered_counter_by_emotion

def filter_dataset_concept_pairs(dataset_concept_pairs, concept_pairs_vocab):
    print("filtering concept pairs from dataset concept pairs...")
    filtered_dataset_concept_pairs = []
    for pairs in tqdm(dataset_concept_pairs):
        filterd_pairs = []
        for pair in pairs:
            if pair in concept_pairs_vocab:
                filterd_pairs.append(pair)
        if len(filterd_pairs) > 0:
            filtered_dataset_concept_pairs.append(filterd_pairs)
    return filtered_dataset_concept_pairs

def PPMI_smoothed(concept_pair, emotion, counter_by_emotion, combined_counter, emotion_counter, total_cnt, total_cnt_smoothed):
    p_concept_pair = combined_counter[concept_pair]**0.75/total_cnt_smoothed
    p_emotion = emotion_counter[emotion]/total_cnt
    p_cooccurrence = (1e-6 + counter_by_emotion[emotion][concept_pair])/total_cnt
    return max(math.log(p_cooccurrence/(p_concept_pair*p_emotion), 2), 0)

# def PPMI_smoothed(concept_pair, emotion, counter_by_emotion, combined_counter, emotion_counter, total_cnt, total_cnt_smoothed):
#     p_concept_pair = combined_counter[concept_pair]/total_cnt
#     p_emotion = emotion_counter[emotion]**0.75/total_cnt_smoothed
#     p_cooccurrence = (1e-6 + counter_by_emotion[emotion][concept_pair])/total_cnt
#     return max(math.log(p_cooccurrence/(p_concept_pair*p_emotion), 2), 0)

def exceed_margin(v, margin):
    sorted_v = sorted(v)
    return sorted_v[-1] - sorted_v[-2] >= margin


def PPMI_from_matrix(M):
    total = M.sum()
    msg_total = M.sum(axis=1, keepdims=True)
    res_total = M.sum(axis=0, keepdims=True)
    
    # msg_prob = msg_total/total
    msg_prob = np.power(msg_total, 0.75)/np.power(msg_total, 0.75).sum()
    res_prob = np.power(res_total, 0.75)/np.power(res_total, 0.75).sum()
    joint_prob = (M+1e-6)/total
    
    PPMI = np.maximum(np.log2(joint_prob/(msg_prob*res_prob)), 0)
    return PPMI

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--smaller', action="store_true")
    parser.add_argument('--conceptnet_path', type=str, required=True)
    parser.add_argument('--pair_frequency_threshold', type=int, default=5)
    parser.add_argument('--PPMI_threshold', type=float, default=1)
    parser.add_argument('--PPMI_margin', type=float, default=1)
    parser.add_argument('--concept_pairs_path', type=str, default="")

    args = parser.parse_args()
    dataset = args.dataset.lower()
    smaller = args.smaller
    conceptnet_path = args.conceptnet_path
    pair_frequency_threshold = args.pair_frequency_threshold
    PPMI_threshold = args.PPMI_threshold
    PPMI_margin = args.PPMI_margin
    concept_pairs_path = args.concept_pairs_path

    two_step_PPMI = True

    # load data
    if dataset == "reddit":
        src_path = "./data/Reddit/train-src{0}.txt".format("-smaller" if smaller else "")
        tgt_path = "./data/Reddit/train-tgt{0}.txt".format("-smaller" if smaller else "")
        tgt_emotion_path = "./data/Reddit/train-tgt{0}-emotions.txt".format("-smaller" if smaller else "")
    elif dataset == "twitter":
        src_path = "./data/Twitter/train-src.txt"
        tgt_path = "./data/Twitter/train-tgt.txt"
        tgt_emotion_path = "./data/Twitter/train-tgt-emotions.txt"
    
    print("Loading data from {0}".format([src_path, tgt_path, tgt_emotion_path]))
    src = load_file(src_path)
    tgt = load_file(tgt_path)
    tgt_emotions = load_file(tgt_emotion_path)
    tgt_emotions = [id2emotion[int(emo[0])] for emo in tgt_emotions]
    assert len(src) == len(tgt) and len(src) == len(tgt_emotions)
    print("number of conversation pairs: {0}".format(len(src)))
    
    # load conceptnet
    print("loading conceptnet from {0}...".format(conceptnet_path))
    CN = load_pickle(conceptnet_path)
    CN_concepts = set(get_all_entities(CN))
    print("number of conceptnet triplets: {0}".format(len(CN)))
    print("number of conceptnet entities: {0}".format(len(CN_concepts)))

    # load stopwords
    stopwords = load_pickle("./data/KB/stopwords.pkl")

    print()
    # extract triplets from dataset for each emotion
    print("extracting concept pairs from dataset...")
    if concept_pairs_path:
        dataset_concept_pairs, concept_pairs_counter_by_emotion = load_pickle(concept_pairs_path)
    else:
        dataset_concept_pairs, concept_pairs_counter_by_emotion = get_concept_pairs(src, tgt, tgt_emotions, CN_concepts, stopwords)
        concept_pairs_path = conceptnet_path.replace(".pkl", "_{0}_pairs.pkl".format(dataset))
        print("saving reddit_concept_pairs to {0}...".format(concept_pairs_path))
        save_pickle((dataset_concept_pairs, concept_pairs_counter_by_emotion), concept_pairs_path)
    
    # filter out infrequent pairs
    print()
    print("before filtering, percentage of conversations that have concept pairs: ", len(dataset_concept_pairs)/len(src))
    print("Emotion", "Unique Counts", "Total Counts")
    for emo, counter in concept_pairs_counter_by_emotion.items():
        print(emo, len(counter), sum(counter.values()))
    concept_pairs_counter = combine_emotion_counters(concept_pairs_counter_by_emotion)
    print("before filtering, number of unique concept pairs: ", len(concept_pairs_counter))
    
    filtered_counter_by_emotion = filter_emotion_counters(concept_pairs_counter_by_emotion, concept_pairs_counter, pair_frequency_threshold)
    print("Emotion", "Unique Counts", "Total Counts")
    for emo, counter in filtered_counter_by_emotion.items():
        print(emo, len(counter), sum(counter.values()))

    filtered_concept_pairs_counter = combine_emotion_counters(filtered_counter_by_emotion)
    print("after filtering, number of unique concept pairs: ", len(filtered_concept_pairs_counter))
    filtered_dataset_concept_pairs = filter_dataset_concept_pairs(dataset_concept_pairs, filtered_concept_pairs_counter)
    print("after filtering, percentage of conversations that have concept pairs: ", len(filtered_dataset_concept_pairs)/len(src))

    if two_step_PPMI:
        print()
        print("computing 1st PPMI...")
        # 1st PPMI to extract related concept pairs
        msg_concepts = [k[0] for k,v in filtered_concept_pairs_counter.items() if v >= 5]
        res_concepts = [k[1] for k,v in filtered_concept_pairs_counter.items() if v >= 5]

        msg_concept2id = {c:i for i,c in enumerate(set(msg_concepts))}
        res_concept2id = {c:i for i,c in enumerate(set(res_concepts))}

        msg_id2concept = {i:c for c,i in msg_concept2id.items()}
        res_id2concept = {i:c for c,i in res_concept2id.items()}
        print("msg and res vocabs: ", len(msg_concept2id), len(res_concept2id))

        M = np.zeros((len(msg_concept2id), len(res_concept2id)))
        for k,v in filtered_concept_pairs_counter.items():
            M[msg_concept2id[k[0]], res_concept2id[k[1]]] = v
        print("cooccurrence matrix sparsity level: ", np.sum(M > 0)/(M.shape[0] * M.shape[1]))

        M_PPMI = PPMI_from_matrix(M)
        
        concept_pairs_satisfying_PPMI = set()
        for k,v in filtered_concept_pairs_counter.items():
            ppmi = M_PPMI[msg_concept2id[k[0]], res_concept2id[k[1]]]
            if ppmi >= PPMI_threshold:
                concept_pairs_satisfying_PPMI.add(k)
        print("PPMI>=1, total: ", len(concept_pairs_satisfying_PPMI), len(filtered_concept_pairs_counter))
    
    # 2nd PPMI
    print()
    print("computing 2nd PPMI...")
    emotion_counter = Counter({k:sum(v.values()) for k,v in filtered_counter_by_emotion.items()})
    total_cnt = sum(filtered_concept_pairs_counter.values())
    total_cnt_smoothed = sum([v**0.75 for v in filtered_concept_pairs_counter.values()])
    # total_cnt_smoothed = sum([v**0.75 for v in emotion_counter.values()])
    concept_pair_PPMI_smoothed = {}
    PPMI_smoothed = partial(PPMI_smoothed, 
                counter_by_emotion=filtered_counter_by_emotion,
                combined_counter=filtered_concept_pairs_counter,
                emotion_counter=emotion_counter,
                total_cnt=total_cnt,
                total_cnt_smoothed=total_cnt_smoothed)
    for k in tqdm(filtered_concept_pairs_counter):
        concept_pair_PPMI_smoothed[k] = [PPMI_smoothed(concept_pair=k, emotion=emo) for emo in id2emotion]

    print("finding the most associated emotion for each concept pair...")
    concept_pair_emotion_smoothed = {}
    for k,v in tqdm(concept_pair_PPMI_smoothed.items()):
        concept_pair_emotion_smoothed[k] = id2emotion[np.argmax(v)]
    print("before filtering, emotion distribution: ")
    print(Counter(concept_pair_emotion_smoothed.values()))

    print("filtering concepts based on its PMI values...")
    filtered_concepts = [k for k,v in concept_pair_PPMI_smoothed.items() if max(v) >= 1 and exceed_margin(v, PPMI_margin)]
    print("number of selected concept pairs: ", len(filtered_concepts))
    
    # create new relations
    print("adding new assertions to conceptnet...")
    new_triplets = []
    for k in filtered_concepts:
        if two_step_PPMI:
            if k in concept_pairs_satisfying_PPMI:
                new_triplets.append((k[0], concept_pair_emotion_smoothed[k], k[1], 1))
        else:
            new_triplets.append((k[0], concept_pair_emotion_smoothed[k], k[1], 1))
    CN.extend(new_triplets)
    print("after filtering, we got {0} new triplets, the emotion distribution is: ".format(len(new_triplets)))
    print(Counter([k[1] for k in new_triplets]))
    
    print("number of conceptnet triplets: {0}".format(len(CN)))
    print("number of conceptnet entities: {0}".format(len(set(get_all_entities(CN)))))
    
    # save augmented conceptnet
    new_conceptnet_path = conceptnet_path.replace(".pkl", "_{0}_PPMI_threshold_{1:.1f}_margin_{2:.1f}.pkl".format(dataset, PPMI_threshold, PPMI_margin))
    print("saving augmented conceptnet to {0}...".format(new_conceptnet_path))
    save_pickle(CN, new_conceptnet_path)