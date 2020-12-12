import pickle
import string
import torch

emotion2id = {
    "neutral": 0,
    "joy": 1,
    "sadness": 2,
    "surprise": 3,
    "fear": 4,
    "anger": 5,
    "disgust": 6
}

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_file(path):
    with open(path, "r") as f:
        return [line.strip("\n").split(" ") for line in f.readlines()]

# load voacb
def load_vocab(dataset, smaller):
    print("Loading vocab...")
    vocab_path = "./data/{0}/{1}{2}.vocab.pt".format(dataset.lower().capitalize(), dataset.lower(), "-smaller" if smaller else "")
    vocab = torch.load(vocab_path)
    print("vocab size: {0}".format(len(vocab['src'].base_field.vocab.stoi)))
    vocab_filtered = []
    for w in vocab['src'].base_field.vocab.stoi:
        vocab_filtered.append((w, vocab['src'].base_field.vocab.freqs[w]))
    return vocab, vocab_filtered

def get_punctuation_words(vocab):
    # vocab: [(w,cnt), ...]
    keys_to_remove = []
    for w,cnt in vocab:
        has_all_puncs = True
        for char in w:
            if char not in string.punctuation:
                has_all_puncs = False
                break
        if has_all_puncs:
            keys_to_remove.append(w)
    return keys_to_remove

def get_digit_words(vocab):
    # vocab: [(w,cnt), ...]
    keys_to_remove = []
    for w,cnt in vocab:
        has_all_digits = True
        for char in w:
            if char not in string.digits:
                has_all_digits = False
                break
        if has_all_digits:
            keys_to_remove.append(w)
    return keys_to_remove

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