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
    parser.add_argument('--percentage', type=float, required=True)

    args = parser.parse_args()
    dataset = args.dataset
    percentage = args.percentage

    # load original data
    src = []
    tgt = []
    for split in ["train", "valid", "test"]:
        for party in ["src", "tgt"]:
            print("*"*30+split+"*"*30)
            print("loading data...")
            with open("./data/{0}/{1}-{2}.txt".format(dataset, split, party)) as f:
                lines = f.readlines()
                num_lines = int(len(lines)*percentage)
            with open("./data/{0}/{1}-{2}-{3}.txt".format(dataset, split, party, "smaller"), 'w') as f:
                for line in lines[:num_lines]:
                    f.write(line)