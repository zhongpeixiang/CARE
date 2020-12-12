import os
import pickle
import random
import argparse
from collections import Counter, defaultdict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conceptnet_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)

    args = parser.parse_args()
    conceptnet_path = args.conceptnet_path
    data_dir = args.data_dir
    num_valid = 50000
    num_test = 50000

    # load conceptnet
    conceptnet = pickle.load(open(conceptnet_path, "rb")) # a list of (h, r, t, w)
    print("number of conceptnet triplets: {0}".format(len(conceptnet)))
    random.shuffle(conceptnet)
    # conceptnet_train = conceptnet[:-(num_valid+num_test)]
    # conceptnet_valid = conceptnet[-(num_valid+num_test):-num_test]
    # conceptnet_test = conceptnet[-num_test:]

    # split datasets without unseen entities
    entities_train_heads = [h for h,r,t,w in conceptnet]
    entities_train_tails = [t for h,r,t,w in conceptnet]
    entities_train = entities_train_heads + entities_train_tails
    relations_train = [r for h,r,t,w in conceptnet]

    entity_counter = Counter(entities_train)
    relation_counter = Counter(relations_train)
    used_entities = defaultdict(int)
    used_relations = defaultdict(int)

    conceptnet_train = []
    conceptnet_valid = []
    conceptnet_test = []
    for h,r,t,w in conceptnet:
        if used_entities[h] < entity_counter[h]-1 and used_entities[t] < entity_counter[t]-1 and used_relations[r] < relation_counter[r]-1:
            if len(conceptnet_valid) < num_valid:
                conceptnet_valid.append((h,r,t,w))
                used_entities[h] += 1
                used_entities[t] += 1
                used_relations[r] += 1
            elif len(conceptnet_test) < num_test:
                conceptnet_test.append((h,r,t,w))
                used_entities[h] += 1
                used_entities[t] += 1
                used_relations[r] += 1
            else:
                conceptnet_train.append((h,r,t,w))
        else:
            conceptnet_train.append((h,r,t,w))
    
    # entity ids
    concept2id = {}
    relation2id = {}

    for h, r, t, w in conceptnet:
        if h not in concept2id:
            concept2id[h] = len(concept2id)
        if t not in concept2id:
            concept2id[t] = len(concept2id)
        if r not in relation2id:
            relation2id[r] = len(relation2id)
    
    with open(os.path.join(data_dir, "entity2id.txt"), "w") as f:
        f.write("{0}\n".format(len(concept2id)))
        for c, c_id in concept2id.items():
            f.write("{0}\t{1}\n".format(c, c_id))
    with open(os.path.join(data_dir, "relation2id.txt"), "w") as f:
        f.write("{0}\n".format(len(relation2id)))
        for r, r_id in relation2id.items():
            f.write("{0}\t{1}\n".format(r, r_id))
    with open(os.path.join(data_dir, "train2id.txt"), "w") as f:
        f.write("{0}\n".format(len(conceptnet_train)))
        for h, r, t, w in conceptnet_train:
            f.write("{0} {1} {2}\n".format(concept2id[h], concept2id[t], relation2id[r]))
    with open(os.path.join(data_dir, "valid2id.txt"), "w") as f:
        f.write("{0}\n".format(len(conceptnet_valid)))
        for h, r, t, w in conceptnet_valid:
            f.write("{0} {1} {2}\n".format(concept2id[h], concept2id[t], relation2id[r]))
    with open(os.path.join(data_dir, "test2id.txt"), "w") as f:
        f.write("{0}\n".format(len(conceptnet_test)))
        for h, r, t, w in conceptnet_test:
            f.write("{0} {1} {2}\n".format(concept2id[h], concept2id[t], relation2id[r]))