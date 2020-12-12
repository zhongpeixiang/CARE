import argparse
import itertools
from tqdm import tqdm
import numpy as np
from util import load_file, load_pickle, get_all_entities, get_concepts


def get_concept_pairs(src, tgt, CN_concept_pairs, CN_concepts, stopwords):
    dataset_concept_pairs = []
    
    for msg, res in tqdm(zip(src, tgt)):
        msg_concepts = get_concepts(msg, CN_concepts, stopwords)
        res_concepts = get_concepts(res, CN_concepts, stopwords)
        if len(msg_concepts) > 0 and len(res_concepts) > 0:
            for pair in itertools.product(msg_concepts, res_concepts):
                if pair in CN_concept_pairs:
                    dataset_concept_pairs.append(pair)
    return dataset_concept_pairs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--src_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--conceptnet_path', type=str, required=True)

    args = parser.parse_args()
    src_path = args.src_path
    output_path = args.output_path
    conceptnet_path = args.conceptnet_path
    
    print("Loading src and output files from ", src_path, output_path)
    src = load_file(src_path)
    output = load_file(output_path)
    assert len(src) == len(output)

    # load conceptnet
    print("loading conceptnet from {0}...".format(conceptnet_path))
    CN = load_pickle(conceptnet_path)
    CN_concepts = set(get_all_entities(CN))
    CN_concept_pairs = set([(h,t) for h,r,t,w in CN])
    print("number of conceptnet triplets: {0}".format(len(CN)))
    print("number of conceptnet entities: {0}".format(len(CN_concepts)))

    # load stopwords
    stopwords = load_pickle("./data/KB/stopwords.pkl")

    print()
    # extract triplets from dataset for each emotion
    print("extracting concept pairs from src and output")
    dataset_concept_pairs = get_concept_pairs(src, output, CN_concept_pairs, CN_concepts, stopwords)

    print("avg number of concept pairs per sentence: ", len(dataset_concept_pairs)/len(src))
    print("avg number of unique concept pairs per sentence: ", len(set(dataset_concept_pairs))/len(src))