import os
import re
import json
from tqdm import tqdm
from spacy.lang.en import English

nlp = English()
tokenizer = nlp.Defaults.create_tokenizer(nlp)

def tokenize(line):
    return [token.text for token in tokenizer(line)]

def load_reddit(path, split):
    fname = os.path.join(path, "reddit_"+ split + ".json")
    with open(fname, "r") as f:
        reddit = json.loads(f.read())
    return reddit

def process_reddit_line(line):
    line = line.lower().replace("&amp;", "").replace("#x200b;", "")
    line = line.strip().replace("\n\n\n\n", ". ").replace("\n\n\n", ". ").replace("\n\n", ". ").replace("\n", ". ")
    line = re.sub(r"http[a-zA-Z0-9\.\/:\?\=\+\-_]+", '', line) # remove url
    line = re.sub(r'\[[a-zA-Z0-9_]*\]', "", line)
    line = re.sub(r'\(\)\.*\s*', "", line)
    # line = line.replace(" ---", ",").replace(" --", ",").replace(" -", ",")
    # line = line.replace("...", ".").replace("..", ".")
    line = line.replace("  ", " ")
    return line

def process_reddit(dataset):
    print("processing reddit...")
    pairs = []
    for thread in dataset:
        for ex in thread:
            pairs.append((ex["context"], ex["response"]))
            for i in range(len(ex.keys()) - 3):
                if i==0:
                    pairs.append((ex["context/{0}".format(i)], ex["context"]))
                else:
                    pairs.append((ex["context/{0}".format(i)], ex["context/{0}".format(i-1)]))
    
    src = []
    tgt = []
    print("number of raw pairs: ", len(pairs))
    for src_sent, tgt_sent in tqdm(set(pairs)):
        processed_src = process_reddit_line(src_sent)
        processed_tgt = process_reddit_line(tgt_sent)
        src.append(" ".join(tokenize(processed_src)) + "\n")
        tgt.append(" ".join(tokenize(processed_tgt)) + "\n")
    return src, tgt


if __name__ == "__main__":
    for split in ["train", "valid", "test"]:
        print("*"*30+split+"*"*30)
        print("loading data...")
        data = load_reddit("/boot/data/reddit/processed/", split)
        num_threads = int(len(data)/10)
        src, tgt = process_reddit(data[:num_threads])
        print("saving data...")
        print("number of distinct pairs: ", len(src))
        with open("./data/Reddit/{0}-src-small.txt".format(split), "w") as f:
            f.writelines(src)
        with open("./data/Reddit/{0}-tgt-small.txt".format(split), "w") as f:
            f.writelines(tgt)