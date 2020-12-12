# -*- coding: utf-8 -*-

""" 
emotion2id = {
    "neutral": 0,
    "joy": 1,
    "sadness": 2,
    "surprise": 3,
    "fear": 4,
    "anger": 5,
    "disgust": 6
}
"""
from __future__ import print_function, division, unicode_literals
import time
import random
import argparse
import logging
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class LinearModel(nn.Module):
    def __init__(self, feature_dim, output_dim, dropout):
        super(LinearModel, self).__init__()
        self.fc = nn.Linear(feature_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        # x: (batch_size, feature_dim)
        x = self.dropout(x)
        return self.fc(x)

if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO, \
                    format = '%(asctime)s  %(levelname)-5s %(message)s', \
                    datefmt =  "%Y-%m-%d-%H-%M-%S")

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="Reddit")
    parser.add_argument('--smaller', action="store_true")
    parser.add_argument('--path', type=str, default="")

    args = parser.parse_args()
    dataset = args.dataset
    smaller = args.smaller
    path = args.path

    batch_size = 64
    seed = 1
    device = torch.device(0)
    dropout=0 # 0.3
    

    # set seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    model = LinearModel(2304, 6, dropout)
    model.load_state_dict(torch.load("./saved_model/deepmoji.pt"))
    print(model)
    model.to(device)

    if path=="":
        for split in ["train", "valid", "test"]:
            print("*"*30+split+"*"*30)
            print("Computing or loading torchmoji embedding...")
            load_path = "./data/{0}/{1}-tgt-embedding.npy".format(dataset, split)
            save_path="./data/{0}/{1}-tgt-emotions.txt".format(dataset, split)
            if smaller:
                load_path = "./data/{0}/{1}-tgt-smaller-embedding.npy".format(dataset, split)
                save_path="./data/{0}/{1}-tgt-smaller-emotions.txt".format(dataset, split)
            
            valid_embedding = np.load(load_path)
            
            print("Start inference...")
            valid_iter = DataLoader(valid_embedding, batch_size=batch_size, shuffle=False)
            model.eval()
            output = []
            for batch_x in tqdm(valid_iter):
                batch_x = torch.FloatTensor(batch_x).to(device)
                with torch.no_grad():
                    batch_output = model(batch_x)
                output.extend(batch_output.argmax(dim=1).tolist())

            # save predictions
            print("Saving inference...")
            with open(save_path, "w") as f:
                for emotion in output:
                    f.write(str(emotion) + "\n")
    else:
        print("Classifying emotion for {0}...".format(path))
        print("Computing or loading torchmoji embedding...")
        load_path = path
        save_path = path.replace("-torchmoji.npy", "-emotions.txt")
        
        valid_embedding = np.load(load_path)
        
        print("Start inference...")
        valid_iter = DataLoader(valid_embedding, batch_size=batch_size, shuffle=False)
        model.eval()
        output = []
        for batch_x in tqdm(valid_iter):
            batch_x = torch.FloatTensor(batch_x).to(device)
            with torch.no_grad():
                batch_output = model(batch_x)
            output.extend(batch_output.argmax(dim=1).tolist())

        # save predictions
        print("Saving inference...")
        with open(save_path, "w") as f:
            for emotion in output:
                f.write(str(emotion) + "\n")