'''
Author: Sicen Liu
Date: 2025-05-29 14:31:31
LastEditTime: 2025-07-06 15:10:11
FilePath: /liusicen/shared_files/MOFE/predict.py
Description: preict.py

Copyright (c) 2025 by ${liusicen_cs@outlook.com}, All Rights Reserved. 
'''
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
from utils.data_processing import load_dataset, proteinDataset, kfold_split, collate_fn_batch,CAID3_collate_fn_batch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from src.Trainer import Trainer
from src.evaluate import eval
import json
import pickle
from collections import OrderedDict

import wandb


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

    device = torch.device('cpu')
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
    if args.aaIndex: 
        feature_dims.append(feature_dim_dict['aaIndex'])
    if args.pssm:
        feature_dims.append(feature_dim_dict['pssm'])
    if args.energy:
        feature_dims.append(feature_dim_dict['energy'])
    
    if args.t5: 
        feature_dims.append(feature_dim_dict['t5'])
    if args.esm2:
        feature_dims.append(feature_dim_dict['esm2'])
    if args.drBERT:
        feature_dims.append(feature_dim_dict['drBERT'])
    if args.ontoProtein:
        feature_dims.append(feature_dim_dict['ontoProtein'])
    

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
    test_dataset = load_dataset(args, args.test_dataset_path, dataset_type=args.test_dataset)
    
    if args.test_dataset in ['CAID3_disorder_pdb', 'CAID3_disorder_nox','CASP','SL329']:
        collate_fn_with_params = partial(CAID3_collate_fn_batch, args=args)
    else:
        collate_fn_with_params = partial(collate_fn_batch, args=args)
    # collate_fn_with_params = partial(collate_fn_batch, args=args)
    
    #############################Test#######################################
    testDataset = proteinDataset(test_dataset)
    test_dataloader = DataLoader(testDataset,
                                    sampler=SequentialSampler(testDataset),
                                    batch_size=1,
                                    collate_fn=collate_fn_with_params)
       

    if args.model_name == 'FL_DynamicFusion':
        model = FL_DynamicFusion(feature1_dim=feature1_dim, feature2_dim=feature2_dim, hidden_dim=args.hidden_size, freeze=args.freeze, expert1_path=args.expert1_path, expert2_path=args.expert2_path, expert3_path=args.expert3_path,infer_mode=1)
    elif args.model_name == "Encoder":
        model =Encoder(feature1_dim=feature1_dim, feature2_dim=feature2_dim, hidden_dim=args.hidden_size, heads=args.heads)
    elif args.model_name == 'BiLSTM':
        model =BiLSTM(feature1_dim=feature1_dim, feature2_dim=feature2_dim, hidden_dim=args.hidden_size)

    try:
        model_dict = torch.load(args.saved_model_path, map_location=torch.device('cpu'))
        try:
            # 如果每个模块前有module前缀，则需要去掉
            # 去掉参数名中的 module. 前缀
            new_state_dict = OrderedDict()
            for k, v in model_dict['model_state_dict'].items():
                name = k.replace('module.', '')  # 去掉 `module.` 前缀
                new_state_dict[name] = v

            # 加载到模型中
            model.load_state_dict(new_state_dict)
            
        except Exception as e:
            model.load_state_dict(model_dict['model_state_dict'])
        print("=====EPOCH:{}====\n======K:{}======\n=======SEED:{}======".format(model_dict['epoch'], model_dict['k'], model_dict['seed']))
    except Exception as e:
        model.load_state_dict(args.saved_model_path)
    
    
    print("=================TEST =================")
    result_path = os.path.join(os.path.dirname(args.saved_model_path), 'predict_result.txt')

    results = eval(args, model, test_dataloader, type='test', save_path=result_path)
    print("Results:", results)



if __name__ == "__main__":
    main()