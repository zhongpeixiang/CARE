import json
import pickle
import argparse
import numpy as np


def load_file(path):
    with open(path, "r") as f:
        return [line.strip("\n").split(" ") for line in f.readlines()]

def get_distinct(data, k):
    # get distinct-k stats
    if k==1:
        tokens = [w for l in data for w in l]
        return len(set(tokens))/len(tokens)
    elif k==2:
        bigrams = []
        for line in data:
            bigrams.extend(zip(line[:-1], line[1:]))
        return len(set(bigrams))/len(bigrams)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_id', type=int, required=True)
    parser.add_argument('--translate_step', type=int, required=True)
    parser.add_argument('--target_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    
    args = parser.parse_args()
    config_id = args.config_id
    translate_step = args.translate_step
    target = load_file(args.target_file)
    output = load_file(args.output_file)
    dataset = args.dataset
    assert(len(target) == len(output))
    # init
    distinct_1 = 0
    distinct_2 = 0

    # diversity
    print("calculating diversity...")
    distinct_1 = get_distinct(output, 1)
    distinct_2 = get_distinct(output, 2)

    print("calculating averaged sentence length...")
    avg_sent_length = len([w for l in output for w in l])/len(output)

    result = {
        "config_id": config_id,
        "translate_step": translate_step,
        "distinct-1": distinct_1,
        "distinct-2": distinct_2,
        "avg_sent_length": avg_sent_length
    }
    with open("./log/evaluation.json", "a") as f:
        f.write('\n')
        json.dump(result, f)
        