# -*- coding: utf-8 -*-

import torch
import random
import inspect
from itertools import islice
from collections import defaultdict
from onmt.utils.constants import emotion2id

def split_corpus(path, shard_size):
    with open(path, "rb") as f:
        if shard_size <= 0:
            yield f.readlines()
        else:
            while True:
                shard = list(islice(f, shard_size))
                if not shard:
                    break
                yield shard

def split_emotions(data, shard_size):
    if shard_size <= 0:
        yield data
    else:
        while True:
            shard = list(islice(data, shard_size))
            if not shard:
                break
            yield shard

def lexicon_to_id(lexicon, vocab):
    generic_vocab_indices = []
    emotion_vocab_indices = [[] for i in range(len(lexicon))]
    for e in lexicon:
        emotion_vocab_indices[emotion2id[e]] = [vocab['src'].base_field.vocab.stoi[w] for w in lexicon[e]]
    
    all_emotion_vocab_indices = []
    for indices in emotion_vocab_indices:
        all_emotion_vocab_indices.extend(indices)
    
    for idx in range(len(vocab['src'].base_field.vocab.stoi)):
        if idx not in all_emotion_vocab_indices:
            generic_vocab_indices.append(idx)
    # print(len(set(all_emotion_vocab_indices)), len(generic_vocab_indices), len(vocab['src'].base_field.vocab.stoi))
    assert len(set(all_emotion_vocab_indices)) + len(generic_vocab_indices) == len(vocab['src'].base_field.vocab.stoi)
    return generic_vocab_indices, emotion_vocab_indices


def get_parameters(param_dict, mode="numpy"):
    for param in param_dict:
        param_dict[param] = param_dict[param].cpu()
    res = {}
    for param in param_dict:
        if mode == "numpy":
            res[param] = param_dict[param].numpy()
        elif mode == "list":
            res[param] = param_dict[param].numpy().tolist()
        else:
            res[param] = param_dict[param]
    return res

def load_KG_embedding(path):
    print("Loading KG embedding from ", path)
    checkpoint = torch.load(path)
    KG_embedding = get_parameters(checkpoint["model"], mode="tensor")
    return KG_embedding["ent_embeddings.weight"], KG_embedding["rel_embeddings.weight"]

def load_KG_vocab(path):
    print("Loading KG vocab from ", path)
    word2id = {}
    id2word = {}
    with open(path, "r") as f:
        f.readline()
        for l in f:
            w, _id = l.strip().split()
            word2id[w] = int(_id)
            id2word[int(_id)] = w
    return word2id, id2word


def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def use_gpu(opt):
    """
    Creates a boolean if gpu used
    """
    return (hasattr(opt, 'gpu_ranks') and len(opt.gpu_ranks) > 0) or \
        (hasattr(opt, 'gpu') and opt.gpu > -1)


def set_random_seed(seed, is_cuda):
    """Sets the random seed."""
    if seed > 0:
        torch.manual_seed(seed)
        # this one is needed for torchtext random call (shuffled iterator)
        # in multi gpu it ensures datasets are read in the same order
        random.seed(seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True

    if is_cuda and seed > 0:
        # These ensure same initialization in multi gpu mode
        torch.cuda.manual_seed(seed)


def generate_relative_positions_matrix(length, max_relative_positions,
                                       cache=False):
    """Generate the clipped relative positions matrix
       for a given length and maximum relative positions"""
    if cache:
        distance_mat = torch.arange(-length+1, 1, 1).unsqueeze(0)
    else:
        range_vec = torch.arange(length)
        range_mat = range_vec.unsqueeze(-1).expand(-1, length).transpose(0, 1)
        distance_mat = range_mat - range_mat.transpose(0, 1)
    distance_mat_clipped = torch.clamp(distance_mat,
                                       min=-max_relative_positions,
                                       max=max_relative_positions)
    # Shift values to be >= 0
    final_mat = distance_mat_clipped + max_relative_positions
    return final_mat


def relative_matmul(x, z, transpose):
    """Helper function for relative positions attention."""
    batch_size = x.shape[0]
    heads = x.shape[1]
    length = x.shape[2]
    x_t = x.permute(2, 0, 1, 3)
    x_t_r = x_t.reshape(length, heads * batch_size, -1)
    if transpose:
        z_t = z.transpose(1, 2)
        x_tz_matmul = torch.matmul(x_t_r, z_t)
    else:
        x_tz_matmul = torch.matmul(x_t_r, z)
    x_tz_matmul_r = x_tz_matmul.reshape(length, batch_size, heads, -1)
    x_tz_matmul_r_t = x_tz_matmul_r.permute(1, 2, 0, 3)
    return x_tz_matmul_r_t


def fn_args(fun):
    """Returns the list of function arguments name."""
    return inspect.getfullargspec(fun).args
