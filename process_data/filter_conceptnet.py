import pickle
import argparse
import logging
from collections import Counter
from tqdm import tqdm

def get_all_entities(conceptnet):
    heads = [triplet[0] for triplet in conceptnet]
    tails = [triplet[2] for triplet in conceptnet]
    return heads + tails

def is_valid_English_ngram(ngram, en_vocab):
    if "_" in ngram:
        return True
    if ngram in en_vocab:
        return True
    return False

if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO, \
                    format = '%(asctime)s  %(levelname)-5s %(message)s', \
                    datefmt =  "%Y-%m-%d-%H-%M-%S")

    parser = argparse.ArgumentParser()
    parser.add_argument('--min_weight', type=float, required=True)
    parser.add_argument('--min_freq', type=int, required=True)


    args = parser.parse_args()
    min_weight = args.min_weight
    min_freq = args.min_freq
    
    # load conceptnet
    logging.info("loading conceptnet...")
    with open("./data/KB/ConceptNet/conceptnet_en.pkl", "rb") as f:
        conceptnet = pickle.load(f) # a list of (h, r, t, w)
    logging.info("number of conceptnet triplets: {0}".format(len(conceptnet)))
    logging.info("number of conceptnet entities: {0}".format(len(set(get_all_entities(conceptnet)))))

    # load English words
    logging.info("loading English vocab...")
    with open("./data/KB/ConceptNet/words_alpha.txt", "r") as f:
        en_vocab = set(f.read().splitlines()) # a list of words
    logging.info("English vocab size: {0}".format(len(en_vocab)))

    # filter conceptnet by english vocab
    logging.info("filtering by English vocab...")
    conceptnet = [triplet for triplet in tqdm(conceptnet) if is_valid_English_ngram(triplet[0], en_vocab) and is_valid_English_ngram(triplet[2], en_vocab)]
    logging.info("number of conceptnet triplets: {0}".format(len(conceptnet)))
    logging.info("number of conceptnet entities: {0}".format(len(set(get_all_entities(conceptnet)))))

    # filter conceptnet by remove triplets that have the same head and tail concepts
    logging.info("filtering by removing triplets that have the same head and tail concepts...")
    conceptnet = [triplet for triplet in conceptnet if triplet[0] != triplet[2]]
    logging.info("number of conceptnet triplets: {0}".format(len(conceptnet)))
    logging.info("number of conceptnet entities: {0}".format(len(set(get_all_entities(conceptnet)))))

    # filter conceptnet by relation weight
    if min_weight > 0:
        logging.info("filtering by relation weight...")
        conceptnet = [triplet for triplet in conceptnet if triplet[-1] >= min_weight]
        logging.info("number of conceptnet triplets: {0}".format(len(conceptnet)))
        logging.info("number of conceptnet entities: {0}".format(len(set(get_all_entities(conceptnet)))))

    # filter conceptnet by concept frequency
    if min_freq > 1:
        logging.info("filtering by concept frequency...")
        all_concepts = get_all_entities(conceptnet)
        concept_counter = Counter(all_concepts)
        filtered_concepts = set()
        for k,v in concept_counter.items():
            if v>=min_freq:
                filtered_concepts.add(k)
        conceptnet = [triplet for triplet in conceptnet if triplet[0] in filtered_concepts and triplet[2] in filtered_concepts]
        logging.info("number of conceptnet triplets: {0}".format(len(conceptnet)))
        logging.info("number of conceptnet entities: {0}".format(len(set(get_all_entities(conceptnet)))))

    logging.info("saving filtered conceptnet...")
    with open("./data/KB/ConceptNet/conceptnet_en_weight_{0:.1f}_freq_{1}.pkl".format(min_weight, min_freq), "wb") as f:
        pickle.dump(conceptnet, f)