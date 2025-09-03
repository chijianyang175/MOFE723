'''
Author: Sicen Liu
Date: 2025-05-16 15:42:32
LastEditTime: 2025-06-18 20:02:39
FilePath: /liusicen/shared_files/MOFE/src/Trainer.py
Description: 

Copyright (c) 2025 by ${liusicen_cs@outlook.com}, All Rights Reserved. 
'''

import os
import torch
import shutil
import numpy as np
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, TensorDataset

from collections import defaultdict

from src.evaluate import eval, predict


import json
import pickle
from collections import OrderedDict



"""adjust_learning_rate"""
def lr_poly(base_lr, iter, max_iter, power):
    # ratio_length = 1 - (float(current_length) / 30) 
    # iter = iter + ratio_length
    # iter = iter + current_length
    if iter > max_iter:
        iter = iter % max_iter
    return base_lr * ((1 - float(iter) / max_iter) ** (power))#+ (float(current_length) / 30) ** (power))
    # return base_lr * (((1 - float(iter) / max_iter) ** (power))+ 0.1*((1 - (float(current_length) / 30) ** (power))))

def adjust_learning_rate(optimizer, i_iter, args):
    lr = lr_poly(args.lr, i_iter, 100000, 0.9)
    optimizer.param_groups[0]['lr'] = np.min(np.around(lr,8))
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
    return lr

class Trainer:

    def __init__(
            self, args, 
            model, 
            train_dataloader:DataLoader, 
            val_dataloader:DataLoader, 
            test_dataloader
    ):
        self.args = args
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = args.optimizer
    
    def train(self):
        print("Start Training...")
        patients, t_patients = 0, 0
        model = self.model
        global_step = 0
        nb_tr_setp = 0
        tr_loss = 0
        val_loss = 0
        dev_auc_best = 0
        test_auc_best = 0
        best_results = defaultdict(list)
        cu_iter = 0
        # for epoch in trange(int(self.args.epochs), desc="Epoch:", ncols=100):
        for epoch in range(int(self.args.epochs)):
            print("\n--------K: {} epoch: {} seed: {} model_name:{} test_data:{} --------".format(self.args.k, epoch, self.args.seed, self.args.model_name, self.args.test_dataset))
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            prog_iter = tqdm(self.train_dataloader, desc='Training', ncols=100, colour='blue')
            model = model.to(self.args.device)
            model.train()
            for _, batch in enumerate(prog_iter):
                self.optimizer.zero_grad()

                id, seq, ori_labels, input_features1, input_features2, input_mask, labels = batch
                # Move tensors to GPU
                try:
                    input_features1 = input_features1.to(self.args.device)
                except:
                    input_features1 = None
                try:
                    input_features2 = input_features2.to(self.args.device)
                except:
                    input_features2 = None
                input_mask = input_mask.to(self.args.device)
                labels = labels.to(self.args.device)

                # cu_iter +=1
                # adjust_learning_rate(self.optimizer, cu_iter, self.args)
                if self.args.model_name == 'FL_DynamicFusion':
                    logits, moe_loss = model(input_features1, input_features2)
                else: logits = model(input_features1, input_features2)
                # padding 的部分不计算loss
                # logits = logits * input_mask
                loss = self.args.criterion(logits, labels)
                loss = (loss * input_mask).mean()
                if self.args.model_name == 'FL_DynamicFusion':
                    loss = loss + moe_loss * self.args.moe_loss_weight
                
                #清零梯度
                self.optimizer.zero_grad()
                
                loss.backward()
                self.optimizer.step()

                # train_results = cal_batch(self.args, labels, logits, residue_masks, sequences_masks)

                tr_loss += loss.item()
                nb_tr_examples += 1
                nb_tr_steps += 1

                # Display loss
                # if self.is_local_0 == 0:
                prog_iter.set_postfix(train_loss='%.4f' % (tr_loss / nb_tr_steps))
                # prog_iter.set_postfix(train_loss='%.4f' % (loss.item()))

                # 可以在每个epoch或者batch结束后使用torch.cuda.empty_cache()
                torch.cuda.empty_cache()
           
        
            # Validation 
            print("=================VALIDATION =================")
            val_results = eval(self.args, model, self.val_dataloader)
            
            feature_params = {"t5":self.args.t5, "esm2":self.args.esm2, "drBERT":self.args.drBERT, "ontoProtein":self.args.ontoProtein, "aaIndex":self.args.aaIndex, "pssm":self.args.pssm, "energy":self.args.energy}
            feature_name = [k for k, v in feature_params.items() if v == 1]

            if val_results['AUC'] > dev_auc_best:
                patients = 0
                dev_auc_best = val_results['AUC']
                save_path = os.path.join(self.args.save_result, "{}".format(self.args.model_name), "{}".format(self.args.test_dataset), "SEED_{}".format(self.args.seed), "{}".format("-".join(feature_name)), "K_{}".format(self.args.k), "EPOCH_{}".format(epoch))
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save({
                    "epoch" : epoch,
                    "k" : self.args.k,
                    "seed" : self.args.seed,
                    "model_state_dict" : model.state_dict()
                }, os.path.join(save_path,"model.pth"))
            else:
                patients += 1
                if patients == 10:
                    print("Early stopping at epoch {}.".format(epoch))
                    break
                
            