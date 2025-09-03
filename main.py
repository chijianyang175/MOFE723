'''
Author: Sicen Liu
Date: 2025-05-11 11:13:39
LastEditTime: 2025-07-15 16:09:15
FilePath: /liusicen/shared_files/MOFE/main.py
Description: The main file of DynamicFusion.

Copyright (c) 2025 by ${liusicen_cs@outlook.com}, All Rights Reserved. 
'''

import os
import sys
import sys, getopt
import torch
import numpy as np
import re
import os
import random
import torch.nn as nn
from functools import partial

from src.feature_level.model import BiLSTM, Encoder,FL_DynamicFusion
from torch.optim import Adam, AdamW
from utils.args_config import parse_args
from utils.data_processing import load_dataset, proteinDataset, kfold_split, collate_fn_batch, CAID3_collate_fn_batch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from src.Trainer import Trainer


os.environ["WANDB_MODE"] = "offline"

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        use_cuda = True
        print("Running on CUDA")
    else:
        use_cuda = False
        print("Running on CPU")

    setup_seed(args.seed)

    device = torch.device('cuda')
    args.device = device
    
    criterion = {
        "FL_DynamicFusion" :nn.BCEWithLogitsLoss(reduction='none'),
        "ML_DynamicFusion" :nn.BCEWithLogitsLoss(reduction='none'),
        "Encoder" :nn.BCEWithLogitsLoss(reduction='none'),
        "BiLSTM" :nn.BCEWithLogitsLoss(reduction='none'),
    }
    args.criterion = criterion[args.model_name]

    
    # 每种特征分别进行训练并保存模型，对比实验结果选择性能最好的traditional biology features和PSSM-based features
    feature_dim_dict = {
        "aaIndex":20,
        "pssm":20,
        "energy":40,
        "esm2":1280,
        "t5":1024,
        "drBERT":768,
        "ontoProtein":30
    }
    feature_dims = []
    if args.pssm:
        feature_dims.append(feature_dim_dict['pssm'])
    if args.esm2:
        feature_dims.append(feature_dim_dict['esm2'])
    if args.drBERT:
        feature_dims.append(feature_dim_dict['drBERT'])
    

    if len(feature_dims) == 2:
        args.feature1 = True
        args.feature2 = True
        feature1_dim = feature_dims[0]
        feature2_dim = feature_dims[1]
    else:
        args.feature1 = True
        args.feature2 = False
        feature1_dim = feature_dims[0]
        feature2_dim = 0
    

    ####################################################################
    ## Load the dataset
    #####################################################################
    train_dataset = load_dataset(args, args.train_dataset_path, dataset_type=args.train_dataset)
    test_dataset = load_dataset(args, args.test_dataset_path, dataset_type=args.test_dataset)
    
    collate_fn_with_params = partial(collate_fn_batch, args=args)
    
    # K-Fold: return dataset lists
    train_Datas, val_Datas = kfold_split(train_dataset)
    
    for k, (train_Dataset, val_Dataset) in  enumerate(zip(train_Datas, val_Datas)):
        #############################Train#######################################
        train_Dataset = proteinDataset(train_Dataset)
        train_sampler = RandomSampler(train_Dataset)
        train_dataloader = DataLoader(train_Dataset,
                                      sampler=train_sampler,
                                      batch_size=args.batch_size,
                                      collate_fn=collate_fn_with_params)
        val_Dataset = proteinDataset(val_Dataset)
        val_dataloader = DataLoader(val_Dataset,
                                    sampler=SequentialSampler(val_Dataset),
                                    batch_size=1,
                                    collate_fn=collate_fn_with_params)
        
        

        if args.model_name == 'FL_DynamicFusion':
            model = FL_DynamicFusion(feature1_dim=feature1_dim, feature2_dim=feature2_dim, hidden_dim=args.hidden_size, freeze=args.freeze, expert1_path=args.expert1_path, expert2_path=args.expert2_path, expert3_path=args.expert3_path)
        elif args.model_name == "Encoder":
            model =Encoder(feature1_dim=feature1_dim, feature2_dim=feature2_dim, hidden_dim=args.hidden_size, heads=args.heads)
        elif args.model_name == 'BiLSTM':
            model =BiLSTM(feature1_dim=feature1_dim, feature2_dim=feature2_dim, hidden_dim=args.hidden_size)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            model = nn.DataParallel(model)
            model.to('cuda')
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) 

        args.optimizer = optimizer
        args.k = k
        
        trainer = Trainer(args, model, train_dataloader, val_dataloader)
        trainer.train()
    


if __name__ == "__main__":
    main()