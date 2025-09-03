'''
Author: Sicen Liu
Date: 2024-12-13 07:37:56
LastEditTime: 2025-06-18 20:08:37
FilePath: /liusicen/shared_files/MOFE/src/evaluate.py
Description: 

Copyright (c) 2024 by ${liusicen_cs@outlook.com}, All Rights Reserved. 
'''
import pickle
import numpy as np
from sklearn import metrics
import torch
from src.metrics import get_results, predict_get_matrics
import wandb
from collections import defaultdict
from tqdm import tqdm
import os
import json

# evaluate
def eval(args, model, eval_dataloader, type='val', save_path=None):
    model.eval()
    eval_loss = 0
    # batch_results = defaultdict(list)
    data_ids, data_seqs, data_ori_labels, data_labels, data_logits, data_residue_masks, data_sequences_masks = [], [], [], [], [], [], []
    data_gt_masks = []
    data_gt_seqs = []
    data_gt_labels = []
    val_prog_iter = tqdm(eval_dataloader, desc='Val', ncols=100, colour='green')
    nb_val_steps = 0
    with torch.no_grad():
        for idx, batch in enumerate(val_prog_iter):
            id, seq, ori_labels, input_features1, input_features2, input_mask, labels = batch
            # Move tensors to GPU
            try:
                input_features1 = input_features1.to(args.device)
            except:
                input_features1 = None
            try:
                input_features2 = input_features2.to(args.device)
            except:
                input_features2 = None
            input_mask = input_mask.to(args.device)
            labels = labels.to(args.device)
            if args.model_name == 'FL_DynamicFusion':
                logits, moe_loss = model(input_features1, input_features2)
            else:
                logits = model(input_features1, input_features2)
            e_loss = args.criterion(logits, labels)
            e_loss = (e_loss * input_mask).mean()
            eval_loss += e_loss.item()
            if not args.model_name == "DeepDRP":
                logits = torch.sigmoid(logits)
            
            nb_val_steps += 1
            val_prog_iter.set_postfix(val_loss='%.4f' % (eval_loss / nb_val_steps))
            # val_prog_iter.set_postfix(val_loss='%.4f' % (e_loss.item()))
            torch.cuda.empty_cache()
            
            # # 计算batch数据的auc, F1, mcc, acc
            # batch_result = cal_batch(args, labels, logits, residue_masks, sequences_masks)
            # for k, v in batch_result.items():
            #     batch_results[k].append(v)
            data_ids.append(id)
            data_seqs.append(seq)
            data_ori_labels.extend(ori_labels)
            data_labels.extend(labels.detach().cpu().numpy())
            data_logits.extend(logits.detach().cpu().numpy())
            data_gt_masks.extend(input_mask.detach().cpu().numpy())
    
    reuslts = get_results(args, data_labels, data_logits, data_gt_masks)

    if type == 'test':
        # 保存预测的结果
        result_path = os.path.join(save_path, 'predict_result.txt')
        with open(result_path, 'w') as file:
            new_results = {}
            for  k, v in reuslts.items():
                new_results[k] = float(v)
            json.dump(new_results, file)  # 写入字典
            file.write('\n')
            file.write('====='*5) 
            for id, seq, label, logit in zip(data_ids, data_seqs, data_labels, data_logits):
                file.write(f'\n>{id}:\n')
                for sq, ll , lt in zip(seq, label, logit):
                    file.write("{}\t{}\t{}\n".format(sq, ll, lt))

    return reuslts

def preict_get_results(args, ids, seqs, labels, logits, input_mask):
    true_label, pre_label, pre_prob = [], [], []
    for i, logit in enumerate(logits):
        true_label.append(labels[i])
        pre_label.append((logit >= args.threshold).astype(int))
    results, sorted_total_samples = predict_get_matrics(ids, seqs, true_label, pre_label, logits, input_mask)
    return results, sorted_total_samples

# prediction part
def predict(args, model, eval_dataloader):
    model.eval()
    eval_loss = 0
    # batch_results = defaultdict(list)
    data_ids, data_seqs, data_ori_labels, data_labels, data_logits, data_residue_masks, data_sequences_masks = [], [], [], [], [], [], []
    data_gt_masks = []
    data_gt_seqs = []
    data_gt_labels = []
    tri_feature_list = []
    preT_feature_list = []
    fused_feature_list = []
    val_prog_iter = tqdm(eval_dataloader, desc='Predict', ncols=100, colour='green')
    nb_val_steps = 0
    with torch.no_grad():
        for idx, batch in enumerate(val_prog_iter):
            id, seq, ori_labels, input_tri_features, input_preT_features, input_mask, labels = batch
            # Move tensors to GPU
            try:
                input_tri_features = input_tri_features.to(args.device)
            except:
                input_tri_features = None
            try:
                input_preT_features = input_preT_features.to(args.device)
            except:
                    input_preT_features = None
            input_mask = input_mask.to(args.device)
            labels = labels.to(args.device)
            if args.model_name == 'FL_DynamicFusion':
                logits, moe_loss = model(input_tri_features, input_preT_features)
            else:
                logits = model(input_tri_features, input_preT_features)
            e_loss = args.criterion(logits, labels)
            e_loss = (e_loss * input_mask).mean()
            eval_loss += e_loss.item()
            if not args.model_name == "DeepDRP":
                logits = torch.sigmoid(logits)
            
            nb_val_steps += 1
            val_prog_iter.set_postfix(val_loss='%.4f' % (eval_loss / nb_val_steps))
            # val_prog_iter.set_postfix(val_loss='%.4f' % (e_loss.item()))
            torch.cuda.empty_cache()
            
            # # 计算batch数据的auc, F1, mcc, acc
            # batch_result = cal_batch(args, labels, logits, residue_masks, sequences_masks)
            # for k, v in batch_result.items():
            #     batch_results[k].append(v)
            data_ids.append(id)
            data_seqs.append(seq)
            data_ori_labels.extend(ori_labels)
            data_labels.extend(labels.detach().cpu().numpy())
            data_logits.extend(logits.detach().cpu().numpy())
            data_gt_masks.extend(input_mask.detach().cpu().numpy())
    
    results, sorted_total_samples = preict_get_results(args, data_ids, data_seqs, data_labels, data_logits, data_gt_masks)
                
        
    return results, sorted_total_samples