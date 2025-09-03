'''
Author: Sicen Liu
Date: 2025-05-16 15:04:22
LastEditTime: 2025-06-11 18:16:30
FilePath: 
Description: args config

Copyright (c) 2025 by ${liusicen_cs@outlook.com}, All Rights Reserved. 
'''

import os
import numpy as np
import torch
from tqdm import tqdm, trange

DATASET_PATH = "/home/liusicen/methods/DynamicFusion/Datasets/datasets/"
FEATURE_PATH = "/home/liusicen/methods/DynamicFusion/Datasets/"
EXPERT_PATH = "/home/liusicen/methods/DynamicFusion/saved_models/expertModels/"

import argparse
def seed(s):
    if s.isdigit():
        s = int(s)
        if 0 <= s <= 9999:
            return s
        else:
            raise argparse.ArgumentTypeError("Seed must be between 0 and 9999")
    elif s == "random":
        return np.random.randint(0, 9999)
    else:
        raise ValueError("Invalid seed value")

def parse_args():
    parser = argparse.ArgumentParser(description='Hybirds')
    
    # Task
    parser.add_argument("--model_name", type=str, default="FL_DynamicFusion")
    parser.add_argument("--aaIndex", type=int, default=0, help="Utilize aa-index feature")
    parser.add_argument("--pssm", type=int, default=0, help="Utilize pssm feature")
    parser.add_argument("--energy", type=int, default=0, help='Utilize energy feature')
    parser.add_argument("--t5", type=int, default=0, help="Utilize T5 feature")
    parser.add_argument("--drBERT", type=int, default=0, help="Utilize DrBERT feature")
    parser.add_argument("--esm2", type=int, default=0, help="Utilize esm2 feature")
    parser.add_argument("--ontoProtein", type=int, default=0, help="Utilize OntoProtein feature")
    
    # Dataset parameters
    parser.add_argument("--train_dataset_path", type=str, default=DATASET_PATH + "DM4229_training.fasta")
    parser.add_argument("--test_dataset", type=str, default="DISORDER723_test", help="[DISORDER723_test,DisProt832_test,S1_test,CASP,SL329]")
    parser.add_argument("--train_dataset", type=str, default= "DM4229_training")
    parser.add_argument("--evo_path", type=str, default=FEATURE_PATH + "Evo_features")
    parser.add_argument("--preT_feature_path", type=str, default=FEATURE_PATH + "PreT_features")
    parser.add_argument("--cut_seq", type=bool, default=False, help="Whether to cut input seq dim")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default='0')
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--hard_gate", type=bool, default=True, help="Whether to use hard gate")
    parser.add_argument("--moe_loss_weight", type=float, default=0.1, help="Loss weight for MoE loss")
    parser.add_argument("--freeze", type=bool, default=True, help="Whether to freeze the model parameters")
    parser.add_argument("--debug", type=bool, default=False)
    
    
    # Architecture parameters
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=seed, default='random', help="random seed")
    parser.add_argument("--output_dim", type=int, default=500)
    
    # Evaluation parameters
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--max_seq_length", type=int, default=500)
    
    # Dropout parameters
    
    # Save parameters
    parser.add_argument("--save_result", type=str, default="/home/liusicen/methods/DynamicFusion/saved_models")
    parser.add_argument("--saved_embedding", type=int, default=1000)
    parser.add_argument("--saved_model_path", type=str, default="/home/liusicen/models")
    
    test_data_dict = {
        "DISORDER723_test": DATASET_PATH + "DISORDER723_test.fasta",
        "DisProt832_test": DATASET_PATH +  "DisProt832_test.fasta",
        "S1_test": DATASET_PATH + "S1_test.fasta",
        "CASP": DATASET_PATH + "CASP.fasta",
        "Disprot504": DATASET_PATH + "Disprot504.fasta",
        "SL329": DATASET_PATH + "SL329.fasta",
        "MXD494": DATASET_PATH + "MXD494.fasta",
    }

    args = parser.parse_args()
    
    args.expert1_path = os.path.join(EXPERT_PATH, args.test_dataset, "expert1") + "/model.pth"
    args.expert2_path = os.path.join(EXPERT_PATH, args.test_dataset, "expert2") + "/model.pth"
    args.expert3_path = os.path.join(EXPERT_PATH, args.test_dataset, "expert3") + "/model.pth"
    
    test_dataset_path = test_data_dict[args.test_dataset]
    args.test_dataset_path = test_dataset_path
    
    return args