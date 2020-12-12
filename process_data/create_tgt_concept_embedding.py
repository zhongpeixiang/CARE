# -*- coding: utf-8 -*-
import os
import time
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from util import load_file, load_pickle, save_pickle, get_all_entities, get_concepts
from onmt.modules.sparse_activations import sparsemax

id2emotion = ["neutral", "joy", "sadness", "surprise", "fear", "anger"]

def load_concepts(path):
    with open(path, "r") as f:
        return [line.strip("\n").split(", ") for line in f.readlines()]

def load_emotions(path):
    with open(path, "r") as f:
        data = [int(line.strip("\n")) for line in f.readlines()]
    return data

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

# def load_KG_embedding(path):
#     print("Loading KG embedding from ", path)
#     checkpoint = torch.load(path)
#     KG_embedding = get_parameters(checkpoint["model"], mode="tensor")
#     return KG_embedding["ent_embeddings.weight"], KG_embedding["rel_embeddings.weight"]

def load_KG_embedding(path):
    print("Loading KG embedding from ", path)
    checkpoint = torch.load(path)
    # KG_embedding = get_parameters(checkpoint["model"], mode="tensor")
    return checkpoint["ent_embeddings.weight"].cpu(), checkpoint["rel_embeddings.weight"].cpu()

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

def get_softmax_attention(src_ent, rel):
    src_ent_emb_ = ent_emb[ent2id[src_ent]]
    rel_emb_ = rel_emb[rel2id[rel]]
    return torch.softmax(-torch.norm(src_ent_emb_ + rel_emb_ - ent_emb, dim=1), dim=-1)

def get_sparse_attention(src_ent, rel):
    src_ent_emb_ = ent_emb[ent2id[src_ent]]
    rel_emb_ = rel_emb[rel2id[rel]]
    return sparsemax(-torch.norm(src_ent_emb_ + rel_emb_ - ent_emb, dim=1), -1)

def convert_concepts_to_ids(concepts, concept2id, max_num):
    if len(concepts) == 1 and concepts[0] == "":
        return [concept2id["<pad>"]]*max_num
    elif len(concepts) >= max_num:
        return [concept2id[c] for c in concepts[:max_num]]
    else:
        return [concept2id[c] for c in concepts] + (max_num-len(concepts)) * [concept2id["<pad>"]]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--smaller', action="store_true")
    parser.add_argument('--kg_embedding_path', type=str, required=True)
    parser.add_argument('--kg_vocab_dir', type=str, required=True)
    parser.add_argument('--concept_embedding_path', type=str, required=True)
    parser.add_argument('--concept_VAD_strength_dict_path', type=str, default="")
    parser.add_argument('--concept_VAD_strength_lambda', type=float, default=0.5)
    parser.add_argument('--concept_VAD_strength_temp', type=float, default=1.0)
    parser.add_argument('--sparsemax', action="store_true")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_num_concepts', type=int, default=10)
    parser.add_argument('--softmax_temp', type=float, default=1.0)
    parser.add_argument('--sparsemax_temp', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--translation_input', action="store_true")

    args = parser.parse_args()
    dataset = args.dataset.lower()
    smaller = args.smaller
    kg_embedding_path = args.kg_embedding_path
    kg_vocab_dir = args.kg_vocab_dir
    concept_embedding_path = args.concept_embedding_path
    concept_VAD_strength_dict_path = args.concept_VAD_strength_dict_path
    concept_VAD_strength_lambda = args.concept_VAD_strength_lambda
    concept_VAD_strength_temp = args.concept_VAD_strength_temp
    use_sparsemax = args.sparsemax
    batch_size = args.batch_size
    max_num_concepts = args.max_num_concepts
    softmax_temp = args.softmax_temp
    sparsemax_temp = args.sparsemax_temp
    top_k = args.top_k
    translation_input = args.translation_input

    # load KG embedding and vocab
    print("Loading KG embedding and vocab...")
    ent_emb, rel_emb = load_KG_embedding(kg_embedding_path)
    ent_emb = F.normalize(ent_emb, p=2, dim=-1)
    rel_emb = F.normalize(rel_emb, p=2, dim=-1)

    ent2id, id2ent = load_KG_vocab(os.path.join(kg_vocab_dir, "entity2id.txt"))
    rel2id, id2rel = load_KG_vocab(os.path.join(kg_vocab_dir, "relation2id.txt"))
    ent2id["<pad>"] = len(ent2id)
    # avg_emb = torch.stack(list(concept_embedding.values()), dim=0).mean(dim=0)
    # ent_emb = torch.cat([ent_emb, avg_emb.unsqueeze(0)], dim=0)
    # ent_emb = torch.cat([ent_emb, torch.zeros((1, ent_emb.shape[-1]))], dim=0)
    ent_emb = torch.cat([ent_emb, ent_emb.mean(dim=0, keepdim=True)], dim=0)
    ent_emb_dim = ent_emb.shape[-1]
    print("ent embedding shape: ", ent_emb.shape)
    
    # load concept embedding
    print("Loading concept embedding...")
    concept_embedding_dict = load_pickle(concept_embedding_path)
    # print("concept_embedding_dict size", len(concept_embedding_dict))
    emb_dim = len(concept_embedding_dict["hello"])
    concept_embedding = torch.zeros((len(concept_embedding_dict)+1, emb_dim))
    for c,emb in concept_embedding_dict.items():
        concept_embedding[ent2id[c]] = emb
    concept_embedding[ent2id["<pad>"]] = concept_embedding.mean(dim=0)
    print("concept embedding shape: ", concept_embedding.shape)

    # concept_VAD_strength_softmax = torch.ones(len(concept_embedding_dict)+1)/(len(concept_embedding_dict)+1)
    concept_VAD_strength_embedding = None
    if concept_VAD_strength_dict_path != "":
        print("Loading concept VAD strength dict from ", concept_VAD_strength_dict_path)
        concept_VAD_strength_dict = load_pickle(concept_VAD_strength_dict_path)
        concept_VAD_strength_embedding = torch.zeros(len(concept_VAD_strength_dict)+1)
        for k,v in concept_VAD_strength_dict.items():
            concept_VAD_strength_embedding[ent2id[k]] = v
        concept_VAD_strength_embedding[ent2id["<pad>"]] = 0
        # concept_VAD_strength_softmax = torch.softmax(concept_VAD_strength_embedding, dim=-1)

    smaller_suffix = "-smaller" if smaller else ""
    method_suffix = "-softmax"
    value_suffix = "{0:.1f}".format(softmax_temp)
    if use_sparsemax:
        method_suffix = "-sparsemax"
        value_suffix = "{0:.1f}".format(sparsemax_temp)
    elif top_k != 0:
        method_suffix = "-topk"
        value_suffix = "{0}".format(top_k)

    if concept_VAD_strength_dict_path != "":
        method_suffix += "-VAD-{0:.1f}-temp-{1:.1f}".format(concept_VAD_strength_lambda, concept_VAD_strength_temp)
    
    # load data
    splits = ["train", "valid", "test"]
    if translation_input:
        splits = ["valid"]
    
    for split in splits:
        print("*"*60)
        print(split)
        print("*"*60)
        if dataset == "reddit":
            concept_path = "./data/Reddit/{0}-src{1}-concepts.txt".format(split, "-smaller" if smaller else "")
            emotion_path = "./data/Reddit/{0}-tgt{1}-emotions.txt".format(split, "-smaller" if smaller else "")
            save_path = "./data/Reddit/{0}-tgt{1}-concept{2}-embedding-temp-{3}.pkl"\
                .format(split, smaller_suffix, method_suffix, value_suffix)
            if translation_input:
                emotion_path = "./data/Reddit/{0}-src{1}-input-emotions.txt".format(split, "-smaller" if smaller else "")
                save_path = "./data/Reddit/{0}-tgt{1}-input-concept{2}-embedding-temp-{3}.pkl"\
                    .format(split, smaller_suffix, method_suffix, value_suffix)
        elif dataset == "twitter":
            concept_path = "./data/Twitter/{0}-src-concepts.txt".format(split)
            emotion_path = "./data/Twitter/{0}-tgt-emotions.txt".format(split)
            save_path = "./data/Twitter/{0}-tgt-concept{1}-embedding-temp-{2}.pkl"\
                .format(split, method_suffix, value_suffix)
            if translation_input:
                emotion_path = "./data/Twitter/{0}-src-input-emotions.txt".format(split)
                save_path = "./data/Twitter/{0}-tgt-input-concept{1}-embedding-temp-{2}.pkl"\
                    .format(split, method_suffix, value_suffix)
        
        print("Loading concepts from {0}".format(concept_path))
        src_concepts = load_concepts(concept_path)

        print("Loading emotions from {0}".format(emotion_path))
        tgt_emotions = load_emotions(emotion_path)
        assert len(src_concepts) == len(tgt_emotions)

        tgt_concept_embedding = []

        batch_indices = list(range(0, len(src_concepts), batch_size)) + [len(src_concepts)]
        for s_id,e_id in tqdm(zip(batch_indices[:-1], batch_indices[1:]), total=len(batch_indices)-1):
            # create batch ent ids and rel ids
            # print(src_concepts[s_id:e_id])
            start_time = time.time()
            batch_concept_ids = torch.LongTensor([convert_concepts_to_ids(concepts, ent2id, max_num_concepts) for concepts in src_concepts[s_id:e_id]]) # (batch, max_num)
            # print(batch_concept_ids)
            batch_mask = (batch_concept_ids != ent2id["<pad>"]).float() # (batch, max_num)
            batch_mask[:,0] = 1 # to prevent NaN
            batch_rel_ids = torch.LongTensor([rel2id[id2emotion[e]] for e in tgt_emotions[s_id:e_id]]) # (batch, )
            # print(batch_rel_ids)
            # batch_score = -torch.norm(ent_emb[batch_concept_ids].unsqueeze(2) + 
            #     rel_emb[batch_rel_ids].unsqueeze(1).unsqueeze(2) - ent_emb.unsqueeze(0).unsqueeze(0), dim=-1) # (batch, max_num, vocab), costly
            
            batch_score = torch.mm((ent_emb[batch_concept_ids] + rel_emb[batch_rel_ids].unsqueeze(1)).view(-1, ent_emb_dim), ent_emb.transpose(1,0))\
                .view(len(batch_concept_ids), max_num_concepts, -1) # (batch, max_num, vocab)
            
            # batch_score = -torch.norm(ent_emb[batch_concept_ids].unsqueeze(2) * 
            #     rel_emb[batch_rel_ids].unsqueeze(1).unsqueeze(2) - ent_emb.unsqueeze(0).unsqueeze(0), p=1, dim=-1) # (batch, max_num, vocab), costly
            if s_id == 0:
                print(src_concepts[:3])
                print(tgt_emotions[:3])
            if use_sparsemax:
                batch_attn = sparsemax(sparsemax_temp * batch_score, -1) # (batch, max_num, vocab)
                combined_emb = torch.mm(batch_attn.view(-1, batch_attn.shape[-1]), concept_embedding).view(-1, max_num_concepts, emb_dim)
            elif top_k != 0:
                top_k_scores, top_k_indices = batch_score.topk(top_k, dim=2) # (batch, max_num, top_k), (batch, max_num, top_k)
                top_k_attn = torch.softmax(top_k_scores, dim=-1) # (batch, max_num, top_k)
                if s_id == 0:
                    print("Top k probs: ", top_k_attn[:3,0])
                # augment VAD
                if concept_VAD_strength_embedding is not None:
                    VAD_attn = torch.softmax(concept_VAD_strength_temp*concept_VAD_strength_embedding[top_k_indices], dim=-1)
                    if s_id == 0:
                        print("Top k VAD probs: ", VAD_attn[:3,0])
                    top_k_attn = top_k_attn * concept_VAD_strength_lambda + VAD_attn * (1-concept_VAD_strength_lambda)
                top_k_emb = concept_embedding[top_k_indices] # (batch, max_num, top_k, emb_dim)
                combined_emb = (top_k_emb * top_k_attn.unsqueeze(-1)).sum(dim=2) # (batch, max_num, emb_dim)
                if s_id == 0:
                    print("Top k total probs: ", top_k_attn[:3,0])
                    print("Top concepts: ", [id2ent[c] for sent_concepts in top_k_indices[:3,0].tolist() for c in sent_concepts])
            else:
                batch_attn = torch.softmax(softmax_temp * batch_score, dim=-1) # (batch, max_num, vocab)
                combined_emb = torch.mm(batch_attn.view(-1, batch_attn.shape[-1]), concept_embedding).view(-1, max_num_concepts, emb_dim)
            if s_id == 0 and top_k == 0:
                print("Top probs: ", batch_attn[:3,:3].topk(10, dim=2)[0])
                print("Top concepts: ", [id2ent[c] for sent_concepts in batch_attn[:3,0].topk(5, dim=-1)[1].tolist() for c in sent_concepts])
            final_emb = (combined_emb * batch_mask.unsqueeze(-1)).sum(dim=1)/batch_mask.sum(dim=1, keepdim=True)
            tgt_concept_embedding.extend(final_emb.tolist())
        
        # save embedding
        print("Saving tgt concept embeddings to {0}".format(save_path))
        save_pickle(tgt_concept_embedding, save_path)

