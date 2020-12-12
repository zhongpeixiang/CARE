# -*- coding: utf-8 -*-

""" Use torchMoji to score texts for emoji distribution.

The resulting emoji ids (0-63) correspond to the mapping
in emoji_overview.png file at the root of the torchMoji repo.

Writes the result to a csv file.
"""
from __future__ import print_function, division, unicode_literals
import json
import csv
import sys
import time
import random
import pickle
import argparse
import logging
import itertools
from collections import Counter, OrderedDict
import multiprocessing as mp
from tqdm import trange
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import confusion_matrix, classification_report

from torchMoji.torchmoji.sentence_tokenizer import SentenceTokenizer
from torchMoji.torchmoji.model_def import torchmoji_emojis, torchmoji_feature_encoding
from torchMoji.torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH

from util import filter_examples, remove_class, label_weights

logging.basicConfig(level = logging.INFO, \
                    format = '%(asctime)s  %(levelname)-5s %(message)s', \
                    datefmt =  "%Y-%m-%d-%H-%M-%S")

maxlen = 30

class LinearModel(nn.Module):
    def __init__(self, feature_dim, output_dim, dropout):
        super(LinearModel, self).__init__()
        self.fc = nn.Linear(feature_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        # x: (batch_size, feature_dim)
        x = self.dropout(x)
        return self.fc(x)


def torchmoji_embedding(model, st, sentences, chunksize=3000):
    print("computing emoji embedding...")
    print("sample sentences: ")
    print(sentences[:3])
    total_embedding = []
    for i in trange(0, len(sentences), chunksize):
        tokenized, _, _ = st.tokenize_sentences(sentences[i:i+chunksize])
        # print(tokenized[:3])
        total_embedding.extend(model(tokenized))
    print("embedidng length: ", len(total_embedding))
    return total_embedding

def run_epoch(data_iter, model, optimizer, criterion, training, device):
    epoch_target = []
    epoch_output = []
    epoch_loss = []
    for batch_x, batch_y in data_iter:
        batch_x = torch.FloatTensor(batch_x).to(device)
        batch_y = torch.LongTensor(batch_y).to(device)
        # print(batch_x.shape)
        # print(batch_y.shape)
        if training:
            optimizer.zero_grad()
        batch_output = model(batch_x)
        # print(batch_output.shape)
        loss = criterion(batch_output, batch_y)
        if training:
            loss.backward()
            optimizer.step()
        
        epoch_target.extend(batch_y.tolist())
        epoch_output.extend(batch_output.argmax(dim=1).tolist())
        epoch_loss.append(loss.item())
    acc = np.mean(np.array(epoch_target)==np.array(epoch_output))
    cm = confusion_matrix(epoch_target, epoch_output)
    cr = classification_report(epoch_target, epoch_output)
    return acc, np.mean(epoch_loss), cm, cr


def main(config, progress):
    logging.info("*"*80)
    logging.info("Experiment progress: {0:.2f}%".format(progress*100))
    logging.info("*"*80)
    metrics = {
        "score" : [],
    }

    train = config["train"]
    valid = config["valid"]
    test = bool(config["test"])
    feature = config["feature"]
    num_classes = config["num_classes"]
    max_ex_per_class = config["max_ex_per_class"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    lr = config["lr"]
    lr_decay = config["lr_decay"]
    dropout = config["dropout"]
    seed = config["seed"]
    device = torch.device(0)

    # set seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # print("Loading torchmoji vocab and model...")
    # with open(VOCAB_PATH, 'r') as f:
    #     vocabulary = json.load(f)
    # st = SentenceTokenizer(vocabulary, maxlen)

    # if feature=="emoji":
    #     emoji_model = torchmoji_emojis(PRETRAINED_PATH)        
    # elif feature=="embedding":
    #     emoji_model = torchmoji_feature_encoding(PRETRAINED_PATH)

    # load data
    print("Loading data...")
    train = pd.read_csv(train)
    valid = pd.read_csv(valid)
    train_text = train["tweet"].tolist()
    train_label = train["label"].tolist()
    valid_text = valid["tweet"].tolist()
    valid_label = valid["label"].tolist()
    
    print("Computing or loading torchmoji embedding...")
    saved_path_train="train"
    saved_path_valid="valid"
    if test:
        saved_path_valid="test"
    if "clean" in config["train"]:
        saved_path_train += "_clean"
    if feature=="emoji":
        saved_path_train += "_emoji.pkl"
        saved_path_valid += "_emoji.pkl"
    elif feature=="embedding":
        saved_path_train += "_embedding.pkl"
        saved_path_valid += "_embedding.pkl"
    # train_embedding = torchmoji_embedding(emoji_model, st, train_text) # (num_examples, feature_dim)
    # valid_embedding = torchmoji_embedding(emoji_model, st, valid_text)
    # pickle.dump(train_embedding, open("./data/SemEval18/{}".format(saved_path_train), "wb"))
    # pickle.dump(valid_embedding, open("./data/SemEval18/{}".format(saved_path_valid), "wb"))

    train_embedding = pickle.load(open("./data/SemEval18/{}".format(saved_path_train), "rb"))
    valid_embedding = pickle.load(open("./data/SemEval18/{}".format(saved_path_valid), "rb"))
    # sys.exit()
    if num_classes==6:
        # remove disgust
        train_embedding, train_label = remove_class(train_embedding, train_label, 6)
        valid_embedding, valid_label = remove_class(valid_embedding, valid_label, 6)

    if max_ex_per_class != -1:
        # clip examples of majority classes
        train_embedding, train_label = filter_examples(train_embedding, train_label, num_classes, max_ex_per_class)
        # valid_embedding, valid_label = filter_examples(valid_embedding, valid_label, num_classes, max_ex_per_class)

    label_weight = label_weights(train_label)
    label_weight = torch.FloatTensor(label_weight).to(device)
    label_weight = num_classes * label_weight/label_weight.sum()
    print("label weight: ", label_weight)
    print(len(train_embedding), len(train_label), len(valid_embedding), len(valid_label))
    # print("sample label: ", train_label[:3])

    feature_dim = len(train_embedding[0])
    model = LinearModel(feature_dim, num_classes, dropout)
    nn.init.xavier_uniform_(model.fc.weight)
    print(model)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=label_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: lr_decay ** epoch)

    epoch_train_losses = []
    epoch_valid_losses = []
    epoch_valid_accs = []
    epoch_valid_cms = []
    epoch_valid_crs = []
    print("Start training...")
    for epoch in range(epochs):
        train_iter = DataLoader(list(zip(train_embedding, train_label)), batch_size=batch_size, shuffle=True)
        valid_iter = DataLoader(list(zip(valid_embedding, valid_label)), batch_size=batch_size, shuffle=False)
        model.train()
        train_acc, train_loss, _, _ = run_epoch(train_iter, model, optimizer, criterion, training=True, device=device)
        model.eval()
        valid_acc, valid_loss, valid_cm, valid_cr = run_epoch(valid_iter, model, optimizer, criterion, training=False, device=device)
        scheduler.step()
        print("Epoch {0}: train loss: {1:.2f}, valid loss: {2:.2f}, train_acc: {3:.2f}, valid acc: {4:.2f}".format(epoch+1, train_loss, valid_loss, train_acc, valid_acc))
        epoch_train_losses.append(train_loss)
        epoch_valid_losses.append(valid_loss)
        epoch_valid_accs.append(valid_acc)
        epoch_valid_cms.append(valid_cm)
        epoch_valid_crs.append(valid_cr)

        if len(epoch_valid_accs) > 1 and valid_acc > max(epoch_valid_accs[:-1]):
            torch.save(model.state_dict(), "./saved_model/deepmoji_{0}.pt".format(epoch))

    config.pop("seed")
    config.pop("config_id")
    metrics["config"] = config
    metrics["score"].append(max(epoch_valid_accs))
    metrics["cm"] = epoch_valid_cms[np.argmax(epoch_valid_accs)]
    metrics["cr"] = epoch_valid_crs[np.argmax(epoch_valid_accs)]
    return metrics


def clean_config(configs):
    cleaned_configs = []
    for config in configs:
        if config not in cleaned_configs:
            cleaned_configs.append(config)
    return cleaned_configs


def merge_metrics(metrics):
    avg_metrics = {"score" : 0}
    num_metrics = len(metrics)
    for metric in metrics:
        for k in metric:
            if k != "config":
                avg_metrics[k] += np.array(metric[k])
    
    for k, v in avg_metrics.items():
        avg_metrics[k] = (v/num_metrics).tolist()

    return avg_metrics


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Config to read details', required=True)
    args = parser.parse_args()
    with open(args.config) as configfile:
        config = json.load(configfile) # config is now a python dict
    
    # pass experiment config to main
    parameters_to_search = OrderedDict() # keep keys in order
    other_parameters = {}
    keys_to_omit = [] # keys that allow a list of values
    for k, v in config.items():
        # if value is a list provided that key is not device, or kernel_sizes is a nested list
        if isinstance(v, list) and k not in keys_to_omit:
            parameters_to_search[k] = v
        elif k in keys_to_omit and isinstance(config[k], list) and isinstance(config[k][0], list):
            parameters_to_search[k] = v
        else:
            other_parameters[k] = v

    if len(parameters_to_search) == 0:
        config_id = time.perf_counter()
        config["config_id"] = config_id
        logging.info(config)
        main(config, progress=1)
    else:
        all_configs = []
        for i, r in enumerate(itertools.product(*parameters_to_search.values())):
            specific_config = {}
            for idx, k in enumerate(parameters_to_search.keys()):
                specific_config[k] = r[idx]
            
            # merge with other parameters
            merged_config = {**other_parameters, **specific_config}
            all_configs.append(merged_config)
        
        #   logging.info all configs
        for config in all_configs:
            config_id = time.perf_counter()
            config["config_id"] = config_id
            logging.critical("config id: {0}".format(config_id))
            logging.info(config)
            logging.info("\n")

        # multiprocessing
        num_configs = len(all_configs)
        # mp.set_start_method('spawn')
        pool = mp.Pool(processes=config["processes"])
        results = [pool.apply_async(main, args=(x,i/num_configs)) for i,x in enumerate(all_configs)]
        outputs = [p.get() for p in results]

        # if run multiple models using different seed and get the averaged result
        if "seed" in parameters_to_search:
            all_metrics = []
            all_cleaned_configs = clean_config([output["config"] for output in outputs])
            for config in all_cleaned_configs:
                metrics_per_config = []
                for output in outputs:
                    if output["config"] == config:
                        metrics_per_config.append(output)
                avg_metrics = merge_metrics(metrics_per_config)
                all_metrics.append((config, avg_metrics))
            # log metrics
            logging.info("Average evaluation result across different seeds: ")
            for config, metric in all_metrics:
                logging.info("-"*80)
                logging.info(config)
                logging.info(metric)

            # save to log
            with open("./log/{0}.txt".format(time.perf_counter()), "a+") as f:
                for config, metric in all_metrics:
                    f.write(json.dumps("-"*80) + "\n")
                    f.write(json.dumps(config) + "\n")
                    f.write(json.dumps(metric) + "\n")

        else:
            for output in outputs:
                logging.info("-"*80)
                logging.info(output["config"])
                logging.info(output["score"])
                logging.info(output["cm"])
                logging.info(output["cr"])