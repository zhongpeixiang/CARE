import pickle

def load_file(path):
    with open(path, "r") as f:
        return [line.strip("\n").split(" ") for line in f.readlines()]

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def get_all_entities(conceptnet):
    heads = [triplet[0] for triplet in conceptnet]
    tails = [triplet[2] for triplet in conceptnet]
    return heads + tails

def get_ngrams(utter, n):
    # utter: a list of tokens
    # n: up to n-grams
    total = []
    for i in range(len(utter)):
        for j in range(i, max(i-n, -1), -1):
            total.append("_".join(utter[j:i+1]))
    return total

def is_a_ngram_stopword(ngram, stopwords):
    if ngram in stopwords:
        return True
    if "_" in ngram:
        for w in ngram.split("_"):
            if w not in stopwords:
                return False
        return True
    return False

def get_concepts(tokens, CN_concepts, stopwords):
    def not_in_ngram(ngram, ngrams):
        for x in ngrams:
            if ngram in x.split("_"):
                return False
        return True

    concepts = []
    ngrams = sorted(get_ngrams(tokens, 5), key=lambda x: x.count("_"), reverse=True)
    for ngram in ngrams:
        if (ngram in CN_concepts) and not_in_ngram(ngram, concepts) and (not is_a_ngram_stopword(ngram, stopwords)):
            concepts.append(ngram)
    return concepts