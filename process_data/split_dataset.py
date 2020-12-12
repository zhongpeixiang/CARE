import os
import random
import argparse
from tqdm import tqdm


def load_data(path):
    with open(path, "r") as f:
        return f.readlines()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--num_valid', type=int, required=True)
    parser.add_argument('--num_test', type=int, required=True)

    args = parser.parse_args()
    dataset = args.dataset
    num_valid = args.num_valid
    num_test = args.num_test

    # load original data
    src = []
    tgt = []
    pairs = []
    if dataset=="Reddit":
        for split in ["train", "valid", "test"]:
            print("*"*30+split+"*"*30)
            print("loading data...")
            split_src = load_data("./data/{0}/{1}-src.txt".format(dataset, split))
            split_tgt = load_data("./data/{0}/{1}-tgt.txt".format(dataset, split))
            split_pairs = list(zip(split_src, split_tgt))
            random.shuffle(split_pairs) # shuffle within each split
            pairs.extend(split_pairs)
    elif dataset=="Twitter":
        for split in ["train", "valid", "test"]:
            print("*"*30+split+"*"*30)
            print("loading data...")
            split_src = load_data("./data/{0}/{1}-src.txt".format(dataset, split))
            split_tgt = load_data("./data/{0}/{1}-tgt.txt".format(dataset, split))
            src.extend(split_src) # merge three splits then shuffle
            tgt.extend(split_tgt)
        pairs = list(zip(src, tgt))
        random.shuffle(pairs)

    f_train_src = open("./data/{0}/train-src.txt".format(dataset), 'w')
    f_train_tgt = open("./data/{0}/train-tgt.txt".format(dataset), 'w')
    f_valid_src = open("./data/{0}/valid-src.txt".format(dataset), 'w')
    f_valid_tgt = open("./data/{0}/valid-tgt.txt".format(dataset), 'w')
    f_test_src = open("./data/{0}/test-src.txt".format(dataset), 'w')
    f_test_tgt = open("./data/{0}/test-tgt.txt".format(dataset), 'w')
    

    for i, (src_sent, tgt_sent) in tqdm(enumerate(pairs)):
        if i<num_valid:
            f_valid_src.write(src_sent)
            f_valid_tgt.write(tgt_sent)
        elif i<num_valid+num_test:
            f_test_src.write(src_sent)
            f_test_tgt.write(tgt_sent)
        else:
            f_train_src.write(src_sent)
            f_train_tgt.write(tgt_sent)
    
    f_train_src.close()
    f_train_tgt.close()
    f_valid_src.close()
    f_valid_tgt.close()
    f_test_src.close()
    f_test_tgt.close()