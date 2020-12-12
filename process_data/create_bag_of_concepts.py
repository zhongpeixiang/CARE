# -*- coding: utf-8 -*-
import os
import time
import random
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from util import load_file, load_pickle, save_pickle, get_all_entities, get_concepts
from onmt.modules.sparse_activations import sparsemax

id2emotion = ["neutral", "joy", "sadness", "surprise", "fear", "anger"]
CN_relations = ['AtLocation', 'CapableOf', 'Causes', 'CausesDesire', 
'CreatedBy', 'DefinedAs', 'DerivedFrom', 'Desires', 'DistinctFrom', 
'Entails', 'EtymologicallyDerivedFrom', 'EtymologicallyRelatedTo', 
'FormOf', 'HasA', 'HasContext', 'HasFirstSubevent', 'HasLastSubevent', 
'HasPrerequisite', 'HasProperty', 'HasSubevent', 'InstanceOf', 'IsA', 
'LocatedNear', 'MadeOf', 'MannerOf', 'MotivatedByGoal', 'NotCapableOf', 
'NotDesires', 'NotHasProperty', 'PartOf', 'ReceivesAction', 'RelatedTo', 
'SimilarTo', 'SymbolOf', 'Synonym', 'UsedFor']

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

def load_KG_embedding(path):
    print("Loading KG embedding from ", path)
    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    # KG_embedding = get_parameters(checkpoint["model"], mode="tensor")
    if "ent_embeddings.weight" in checkpoint:
        return checkpoint["ent_embeddings.weight"], checkpoint["rel_embeddings.weight"]
    if "model.ent_embeddings.weight" in checkpoint:
        return checkpoint["model.ent_embeddings.weight"], checkpoint["model.rel_embeddings.weight"]

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
    parser.add_argument('--dataset_vocab_path', type=str, required=True)
    parser.add_argument('--concept_VAD_strength_dict_path', type=str, default="")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_num_concepts', type=int, default=10)
    parser.add_argument('--top_k', type=int, default=100)
    parser.add_argument('--translation_input', action="store_true")
    parser.add_argument('--translation_input_multiple', action="store_true")
    parser.add_argument('--translation_input_selected', action="store_true")

    args = parser.parse_args()
    dataset = args.dataset.lower()
    smaller = args.smaller
    kg_embedding_path = args.kg_embedding_path
    kg_vocab_dir = args.kg_vocab_dir
    dataset_vocab_path = args.dataset_vocab_path
    concept_VAD_strength_dict_path = args.concept_VAD_strength_dict_path
    batch_size = args.batch_size
    max_num_concepts = args.max_num_concepts
    top_k = args.top_k
    translation_input = args.translation_input
    translation_input_multiple = args.translation_input_multiple
    translation_input_selected = args.translation_input_selected
    use_gpu = True
    if use_gpu:
        device = torch.device(0)

    # load KG embedding and vocab
    print("Loading KG embedding and vocab...")
    ent_emb, rel_emb = load_KG_embedding(kg_embedding_path)
    ent_emb = F.normalize(ent_emb, p=2, dim=-1)
    rel_emb = F.normalize(rel_emb, p=2, dim=-1)

    ent2id, id2ent = load_KG_vocab(os.path.join(kg_vocab_dir, "entity2id.txt"))
    rel2id, id2rel = load_KG_vocab(os.path.join(kg_vocab_dir, "relation2id.txt"))
    ent2id["<pad>"] = len(ent2id)
    id2ent[len(id2ent)] = "<pad>"
    ent_emb_dim = ent_emb.shape[-1]
    # ent_emb = torch.cat([ent_emb, F.normalize(ent_emb.mean(dim=0, keepdim=True), p=2, dim=-1)], dim=0)
    ent_emb = torch.cat([ent_emb, F.normalize(torch.randn((1, ent_emb_dim)), p=2, dim=-1)], dim=0)
    print("ent embedding shape: ", ent_emb.shape)
    
    # load vocab
    print("Loading dataset vocab from ", dataset_vocab_path)
    vocab_ckpt = torch.load(dataset_vocab_path)
    word2id = vocab_ckpt["src"].base_field.vocab.stoi
    id2word = vocab_ckpt["src"].base_field.vocab.itos
    print("dataset vocab size: ", len(word2id))

    # load stopwords
    stopwords = load_pickle("./data/KB/stopwords.pkl")

    # concept_VAD_strength_softmax = torch.ones(len(concept_embedding_dict)+1)/(len(concept_embedding_dict)+1)
    print("Loading concept VAD strength dict from ", concept_VAD_strength_dict_path)
    concept_VAD_strength_dict = load_pickle(concept_VAD_strength_dict_path)
    concept_VAD_strength_embedding = torch.zeros(len(concept_VAD_strength_dict)+1)
    for k,v in concept_VAD_strength_dict.items():
        concept_VAD_strength_embedding[ent2id[k]] = v
    concept_VAD_strength_embedding[ent2id["<pad>"]] = 0
    # concept_VAD_strength_softmax = torch.softmax(concept_VAD_strength_embedding, dim=-1)

    smaller_suffix = "-smaller" if smaller else ""
    method_suffix = "-topk"
    value_suffix = "{0}".format(top_k)

    if use_gpu:
        ent_emb = ent_emb.to(device)
        rel_emb = rel_emb.to(device)
        concept_VAD_strength_embedding = concept_VAD_strength_embedding.to(device)
    
    # load data
    splits = ["train", "valid", "test"]
    if translation_input or translation_input_multiple or translation_input_selected:
        splits = ["valid"]
    
    for split in splits:
        print("*"*60)
        print(split)
        print("*"*60)
        if dataset == "reddit":
            concept_path = "./data/Reddit/{0}-src{1}-concepts.txt".format(split, "-smaller" if smaller else "")
            emotion_path = "./data/Reddit/{0}-tgt{1}-emotions.txt".format(split, "-smaller" if smaller else "")
            save_path = "./data/Reddit/{0}-tgt{1}-concept{2}-words-temp-{3}.pkl"\
                .format(split, smaller_suffix, method_suffix, value_suffix)
            if translation_input:
                emotion_path = "./data/Reddit/{0}-src{1}-input-emotions.txt".format(split, "-smaller" if smaller else "")
                save_path = "./data/Reddit/{0}-tgt{1}-input-concept{2}-words-temp-{3}.pkl"\
                    .format(split, smaller_suffix, method_suffix, value_suffix)
            if translation_input_multiple:
                emotion_path = "./data/Reddit/{0}-src{1}-input-emotions-multiple.txt".format(split, "-smaller" if smaller else "")
                save_path = "./data/Reddit/{0}-tgt{1}-input-concept{2}-multiple-words-temp-{3}.pkl"\
                    .format(split, smaller_suffix, method_suffix, value_suffix)
            if translation_input_selected:
                concept_path = "./data/Reddit/{0}-src{1}-concepts-selected.txt".format(split, "-smaller" if smaller else "")
                emotion_path = "./data/Reddit/{0}-src{1}-selected-input-emotions.txt".format(split, "-smaller" if smaller else "")
                save_path = "./data/Reddit/{0}-tgt{1}-input-concept{2}-selected-words-temp-{3}.pkl"\
                    .format(split, smaller_suffix, method_suffix, value_suffix)
        elif dataset == "twitter":
            concept_path = "./data/Twitter/{0}-src-concepts.txt".format(split)
            emotion_path = "./data/Twitter/{0}-tgt-emotions.txt".format(split)
            save_path = "./data/Twitter/{0}-tgt-concept{1}-words-temp-{2}.pkl"\
                .format(split, method_suffix, value_suffix)
            if translation_input:
                emotion_path = "./data/Twitter/{0}-src-input-emotions.txt".format(split)
                save_path = "./data/Twitter/{0}-tgt-input-concept{1}-words-temp-{2}.pkl"\
                    .format(split, method_suffix, value_suffix)
            if translation_input_multiple:
                emotion_path = "./data/Twitter/{0}-src-input-emotions-multiple.txt".format(split)
                save_path = "./data/Twitter/{0}-tgt-input-concept{1}-multiple-words-temp-{2}.pkl"\
                    .format(split, method_suffix, value_suffix)
            if translation_input_selected:
                concept_path = "./data/Twitter/{0}-src-concepts-selected.txt".format(split)
                emotion_path = "./data/Twitter/{0}-src-selected-input-emotions.txt".format(split)
                save_path = "./data/Twitter/{0}-tgt-input-concept{1}-selected-words-temp-{2}.pkl"\
                    .format(split, method_suffix, value_suffix)
        
        print("Loading concepts from {0}".format(concept_path))
        src_concepts = load_concepts(concept_path)
        if translation_input_multiple:
            src_concepts_multiple = []
            # repeat src_concepts 6 times
            for concepts in src_concepts:
                for i in range(6):
                    src_concepts_multiple.append(concepts)
            src_concepts = src_concepts_multiple

        print("Loading emotions from {0}".format(emotion_path))
        tgt_emotions = load_emotions(emotion_path)
        assert len(src_concepts) == len(tgt_emotions)

        tgt_concepts = []
        num_relations = len(CN_relations) + 1

        batch_indices = list(range(0, len(src_concepts), batch_size)) + [len(src_concepts)]
        for s_id,e_id in tqdm(zip(batch_indices[:-1], batch_indices[1:]), total=len(batch_indices)-1):
            # create batch ent ids and rel ids
            batch_size = e_id - s_id
            batch_concept_ids = torch.LongTensor([convert_concepts_to_ids(concepts, ent2id, max_num_concepts) for concepts in src_concepts[s_id:e_id]]) # (batch, max_num)
            # batch_rel_ids = torch.LongTensor([rel2id[id2emotion[e]] for e in tgt_emotions[s_id:e_id]]) # (batch, )

            concept_lengths = []
            for concepts in src_concepts[s_id:e_id]:
                if len(concepts) == 1 and concepts[0] == "":
                    concept_lengths.append(1)
                else:
                    concept_lengths.append(min(len(concepts), max_num_concepts))
            if s_id == 0:
                print(concept_lengths)

            # add common relations
            # batch_concept_ids = batch_concept_ids.repeat((num_relations, 1))
            batch_concept_ids = torch.repeat_interleave(batch_concept_ids, repeats=num_relations, dim=0) # (batch*num_relations, max_num)
            rel_ids = []
            for rel in tgt_emotions[s_id:e_id]:
                rel_ids.extend([rel2id[id2emotion[rel]]] + [rel2id[r] for r in CN_relations])
            batch_rel_ids = torch.LongTensor(rel_ids) # (batch*num_relations)
            # print(batch_concept_ids.shape, batch_rel_ids.shape)
            # print(batch_concept_ids[:50,0], batch_rel_ids[:50])
            if use_gpu:
                batch_concept_ids = batch_concept_ids.to(device)
                batch_rel_ids = batch_rel_ids.to(device)

            batch_score = torch.mm((ent_emb[batch_concept_ids] + rel_emb[batch_rel_ids].unsqueeze(1)).view(-1, ent_emb_dim), ent_emb.transpose(1,0))\
                .view(len(batch_concept_ids), max_num_concepts, -1) # (batch*num_relations, max_num, vocab)
            
            top_k_scores, top_k_indices = batch_score.topk(top_k//2, dim=2) # (batch*num_relations, max_num, top_k), (batch*num_relations, max_num, top_k)
            VAD_scores = concept_VAD_strength_embedding[top_k_indices] # (batch*num_relations, max_num, top_k)
            for sent_indices, sent_scores, sent_VAD_scores, num_concepts, emotion in zip(torch.chunk(top_k_indices, chunks=batch_size, dim=0),\
                torch.chunk(top_k_scores, chunks=batch_size, dim=0), torch.chunk(VAD_scores, chunks=batch_size, dim=0), concept_lengths, tgt_emotions[s_id:e_id]):
                
                # sent_relations = ([id2emotion[emotion]] + CN_relations)*num_concepts*(top_k//2)
                sent_relations = []
                for rel in [id2emotion[emotion]] + CN_relations:
                    sent_relations.extend([rel]*num_concepts*(top_k//2))
                word_scores = list(zip(sent_indices[:,:num_concepts].reshape(-1).tolist(), \
                    sent_scores[:,:num_concepts].reshape(-1).tolist(), sent_VAD_scores[:,:num_concepts].reshape(-1).tolist(), sent_relations))
                selected_indices, selected_scores, selected_VAD_scores, selected_relations = [], [], [], []
                # emotional_word_scores = list(zip(sent_indices[0].view(-1).tolist(), sent_scores[0].view(-1).tolist(), sent_VAD_scores[0].view(-1).tolist()))
                # common_word_scores = list(zip(sent_indices[1:].view(-1).tolist(), sent_scores[1:].view(-1).tolist(), sent_VAD_scores[1:].view(-1).tolist()))
                # selected_emotional_indices, selected_emotional_scores, selected_emotional_VAD_scores = [], [], []
                # selected_common_indices, selected_common_scores, selected_common_VAD_scores = [], [], []
                # for idx, score, VAD_score, relation in sorted(word_scores[:num_concepts*(top_k//2)], key=lambda x: x[1], reverse=True):
                emotional_words = word_scores[:num_concepts*(top_k//2)]
                common_words = word_scores[num_concepts*(top_k//2):]
                random.shuffle(emotional_words)
                random.shuffle(common_words)
                for idx, score, VAD_score, relation in emotional_words:
                    if id2ent[idx] in word2id and word2id[id2ent[idx]] not in selected_indices:
                        selected_indices.append(word2id[id2ent[idx]])
                        selected_scores.append(score)
                        selected_VAD_scores.append(VAD_score)
                        selected_relations.append(relation)
                        if len(selected_indices) == top_k//4:
                            break
                
                # for idx, score, VAD_score, relation in sorted(word_scores[num_concepts*(top_k//2):], key=lambda x: x[1], reverse=True):
                for idx, score, VAD_score, relation in common_words:
                    if id2ent[idx] in word2id and word2id[id2ent[idx]] not in selected_indices:
                        selected_indices.append(word2id[id2ent[idx]])
                        selected_scores.append(score)
                        selected_VAD_scores.append(VAD_score)
                        selected_relations.append(relation)
                        if len(selected_indices) == top_k:
                            break
                
                if len(selected_indices) < top_k:
                    for idx, score, VAD_score, relation in common_words:
                        for w in id2ent[idx].split("_"):
                            if w in word2id and word2id[w] not in selected_indices and w not in stopwords:
                                selected_indices.append(word2id[w])
                                selected_scores.append(score)
                                selected_VAD_scores.append(VAD_score)
                                selected_relations.append(relation)
                                if len(selected_indices) == top_k:
                                    break
                        if len(selected_indices) == top_k:
                            break
                
                if len(selected_indices) < top_k:
                    for idx, score, VAD_score, relation in emotional_words:
                        for w in id2ent[idx].split("_"):
                            if w in word2id and word2id[w] not in selected_indices and w not in stopwords:
                                selected_indices.append(word2id[w])
                                selected_scores.append(score)
                                selected_VAD_scores.append(VAD_score)
                                selected_relations.append(relation)
                                if len(selected_indices) == top_k:
                                    break
                        if len(selected_indices) == top_k:
                            break
                
                if len(selected_indices) < top_k:
                    print("*"*60)
                    print("Incomplete bag of concepts!")
                    print("*"*60)
                # if num_concepts == 1:
                #     print("*"*60)
                #     print(len(selected_indices))
                #     print(len(word_scores))
                #     print(word_scores[:100])
                
                tgt_concepts.append((selected_indices, selected_scores, selected_VAD_scores, selected_relations))
            if s_id == 0:
                print([(id2word[idx], relation) for idx, relation in zip(tgt_concepts[s_id][0], tgt_concepts[s_id][3])])
            # if s_id == 0:
            #     print(src_concepts[:3])
            #     print(tgt_emotions[:3])
            #     print("Top k scores: ", top_k_scores[:3,0])
            #     print("Top VAD scores: ", VAD_scores[:3,0])
            #     print("Top concepts: ", [id2ent[c] for sent_concepts in top_k_indices[:3,0].tolist() for c in sent_concepts])
            

            # for sent_indices, sent_scores, sent_VAD_scores in \
            #     zip(top_k_indices.view(batch_size, -1).tolist(), top_k_scores.view(batch_size, -1).tolist(), VAD_scores.view(batch_size, -1).tolist()):

            #     word_scores = []
            #     for idx, rel_score, VAD_score in zip(sent_indices, sent_scores, sent_VAD_scores):
            #         if id2ent[idx] in word2id:
            #             word_scores.append((word2id[id2ent[idx]], rel_score, VAD_score))
            #     assert len(word_scores) >= top_k

            #     top_word_indices, top_word_scores, top_word_VAD_scores = [], [], []
            #     for idx, rel_score, VAD_score in sorted(word_scores, key=lambda x: x[1], reverse=True):
            #         if idx not in top_word_indices:
            #             top_word_indices.append(idx)
            #             top_word_scores.append(rel_score)
            #             top_word_VAD_scores.append(VAD_score)
            #             if len(top_word_indices) == top_k:
            #                 break
            #     tgt_concepts.append((top_word_indices, top_word_scores, top_word_VAD_scores))
        
        # save embedding
        print("Saving tgt concept embeddings to {0}".format(save_path))
        save_pickle(tgt_concepts, save_path)

