import os
import re
import json
from tqdm import tqdm
from spacy.lang.en import English
import preprocessor as p

nlp = English()
tokenizer = nlp.Defaults.create_tokenizer(nlp)
p.set_options(p.OPT.MENTION, p.OPT.HASHTAG, p.OPT.URL, p.OPT.RESERVED, p.OPT.NUMBER)

def tokenize(line):
    return [token.text for token in tokenizer(line)]

def process_twitter_line(line):
    line = p.clean(line)
    line = " ".join(tokenize(line))
    return line


def load_process_twitter(split):
    with open("./data/Twitter/{0}.txt".format(split), "r") as f:
        src = []
        tgt = []
        for line in tqdm(f):
            context = line.split("\t")[0][5:]
            response = line.split("\t")[1][7:]
            src.append(process_twitter_line(context) + "\n")
            tgt.append(process_twitter_line(response) + "\n")
    return src, tgt


if __name__ == "__main__":
    for split in ["train", "valid", "test"]:
        print("*"*30+split+"*"*30)
        print("loading data...")
        src, tgt = load_process_twitter(split)
        print("saving data...")
        print("number of distinct pairs: ", len(src))
        with open("./data/Twitter/{0}-src.txt".format(split), "w") as f:
            f.writelines(src)
        with open("./data/Twitter/{0}-tgt.txt".format(split), "w") as f:
            f.writelines(tgt)