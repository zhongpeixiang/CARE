# -*- coding: utf-8 -*-
import argparse
from tqdm import tqdm
from util import load_file, load_pickle, get_all_entities, get_concepts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--smaller', action="store_true")
    parser.add_argument('--conceptnet_path', type=str, required=True)

    args = parser.parse_args()
    dataset = args.dataset.lower()
    smaller = args.smaller
    conceptnet_path = args.conceptnet_path

    # load conceptnet
    print("loading conceptnet from {0}...".format(conceptnet_path))
    CN = load_pickle(conceptnet_path)
    CN_concepts = set(get_all_entities(CN))
    print("number of conceptnet triplets: {0}".format(len(CN)))
    print("number of conceptnet entities: {0}".format(len(CN_concepts)))

    # load stopwords
    stopwords = load_pickle("./data/KB/stopwords.pkl")

    # load data
    for split in ["train", "valid", "test"]:
        if dataset == "reddit":
            src_path = "./data/Reddit/{0}-src{1}.txt".format(split, "-smaller" if smaller else "")
        elif dataset == "twitter":
            src_path = "./data/Twitter/{0}-src.txt".format(split)
        
        print("Loading data from {0}".format(src_path))
        src = load_file(src_path)
        
        print("extracting concepts from dataset...")
        src_concepts = [get_concepts(line, CN_concepts, stopwords) for line in tqdm(src)]
        
        # save augmented conceptnet
        src_concepts_path = src_path.replace(".txt", "-concepts.txt")
        print("saving src concepts to {0}...".format(src_concepts_path))
        with open(src_concepts_path, "w") as f:
            for concepts in src_concepts:
                f.write(", ".join(concepts) + "\n")