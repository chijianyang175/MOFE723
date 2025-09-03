'''
Author: Sicen Liu
Date: 2024-09-10 06:52:43
LastEditTime: 2025-06-18 20:08:28
FilePath: /liusicen/shared_files/MOFE/src/metrics.py
Description: 

Copyright (c) 2024 by ${liusicen_cs@outlook.com}, All Rights Reserved. 
'''
import numpy as np
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, f1_score, roc_auc_score, auc, precision_recall_curve, roc_curve, average_precision_score, average_precision_score
from math import *
from collections import OrderedDict


def data2numpy(data):
    return data.cpu().detach().numpy()


def get_matrics(true_label, pre_label, pred_logit, mask):
        results = {}
        total_true_label, total_pre_logit = [], []
        TP = 0
        FN = 0
        TN = 0
        FP = 0
        other = 0

        for i in range(len(pre_label)):
            for j in range(len(pre_label[i])):
                if mask[i][j] == 1:
                    if pre_label[i][j] == 0. and true_label[i][j] == 1.:
                        FN = FN + 1
                    elif pre_label[i][j] == 1. and true_label[i][j] == 1.:
                        TP = TP + 1
                    elif pre_label[i][j] == 0. and true_label[i][j] == 0.:
                        TN = TN + 1
                    elif pre_label[i][j] == 1. and true_label[i][j] == 0.:
                        FP = FP + 1
                    elif true_label[i][j] == '2':
                        other = other + 1
                    total_true_label.append(true_label[i][j])
                    total_pre_logit.append(pred_logit[i][j])
                else: continue

        # print('TP =', TP)
        # print('FP =', FP)
        # print('TN =', TN)
        # print('FN =', FN)
        # print('other =', other)
        # print('if the test set is whole ordered, the FPR is adopted!')
        # print('the residue FPR is :', round(FP/(1.0*(FP + TN)), 5))

        if (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) != 0:
            Sn = TP / float(TP + FN)
            Sp = TN / float(TN + FP )
            Bacc = (TP / float(TP + FN) + TN / float(TN + FP)) / 2.0
            MCC = ((TP * TN) - (FP * FN)) / float(sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))
            ACC = (TP + TN )/((TP + TN + FP + FN)*1.0)
            F1 = (2. * TP ) / (2 * TP + FP + FN)
            Prec = TP / float(TP + FP)
        else:
            Sn = 0
            Sp = 0
            Bacc = 0
            MCC = 0
            ACC = 0
            F1 = 0
            Prec = 0
        Sn = round(Sn, 4)
        Sp = round(Sp, 4)
        Bacc = round(Bacc, 4)
        ACC = round(ACC, 4)
        MCC = round(MCC, 4)
        F1 = round(F1, 4)
        Prec = round(Prec, 4)
        results["Sn"] = Sn
        results["Sp"] = Sp
        results["Bacc"] = Bacc
        results["ACC"] = ACC
        results["MCC"] = MCC
        results["F1"] = F1
        results["Prec"] = Prec
        
        AP = average_precision_score(np.array(total_true_label), np.array(total_pre_logit))
        results["AP"] = AP
        
        AUC = roc_auc_score(np.array(total_true_label), np.array(total_pre_logit))
        results["AUC"] = AUC
        
        # Fmax
        # 计算不同阈值下的Precision、Recall
        precisions, recalls, thresholds = precision_recall_curve(np.array(total_true_label), np.array(total_pre_logit))

        # 计算每个阈值对应的F1分数，选择最大值
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)  # 避免除零
        f1_max = np.nanmax(f1_scores)  # 忽略NaN值
        optimal_threshold = thresholds[np.nanargmax(f1_scores)]
        results["F_max"] = f1_max
        results["F_max_threshold"] = optimal_threshold
        
        # 计算 PR-AUC
        pr_auc = auc(recalls, precisions)
        results["APS-PRAUC"] = pr_auc
        
        for k, v in results.items():
            print("{}:{}".format(k, v))
        
        return results

def get_results(args, labels, logits, input_mask):
    true_label, pre_label = [], []
    for i, logit in enumerate(logits):
        true_label.append(labels[i])
        pre_label.append((logit >= args.threshold).astype(int))
    results = get_matrics(true_label, pre_label, logits, input_mask)
    return results

def predict_get_matrics(ids, seqs, true_label, pre_label, pred_logit, mask):
    results = {}
    total_samples = {}
    total_true_label, total_pre_logit = [], []
    TP = 0
    FN = 0
    TN = 0
    FP = 0
    other = 0

    for i in range(len(pre_label)):
        for j in range(len(pre_label[i])):
            if mask[i][j] == 1:
                if pre_label[i][j] == 0. and true_label[i][j] == 1.:
                    FN = FN + 1
                elif pre_label[i][j] == 1. and true_label[i][j] == 1.:
                    TP = TP + 1
                elif pre_label[i][j] == 0. and true_label[i][j] == 0.:
                    TN = TN + 1
                elif pre_label[i][j] == 1. and true_label[i][j] == 0.:
                    FP = FP + 1
                elif true_label[i][j] == '2':
                    other = other + 1
                total_true_label.append(true_label[i][j])
                total_pre_logit.append(pred_logit[i][j])
            else: continue

    for i in range(len(pre_label)):
            id = ids[i]
            mask_i = mask[i]
            label_i = true_label[i]
            pred_i = pred_logit[i] #[round(l,4) for l in pred_logit[i]] # 保留小数点后四位
            seq = seqs[i]
            total_samples['{}'.format(id)] = ((seq, label_i, pred_i))
        
    sorted_total_samples = {key: total_samples[key] for key in ids if key in total_samples}
    # print('TP =', TP)
    # print('FP =', FP)
    # print('TN =', TN)
    # print('FN =', FN)
    # print('other =', other)
    # print('if the test set is whole ordered, the FPR is adopted!')
    # print('the residue FPR is :', round(FP/(1.0*(FP + TN)), 5))

    if (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) != 0:
        Sn = TP / float(TP + FN)
        Sp = TN / float(TN + FP )
        Bacc = (TP / float(TP + FN) + TN / float(TN + FP)) / 2.0
        MCC = ((TP * TN) - (FP * FN)) / float(sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))
        ACC = (TP + TN )/((TP + TN + FP + FN)*1.0)
        F1 = (2. * TP ) / (2 * TP + FP + FN)
        Prec = TP / float(TP + FP)
    else:
        Sn = 0
        Sp = 0
        Bacc = 0
        MCC = 0
        ACC = 0
        F1 = 0
        Prec = 0
    Sn = round(Sn, 4)
    Sp =  round(Sp, 4)
    Bacc =  round(Bacc, 4)
    ACC =  round(ACC, 4)
    MCC =  round(MCC, 4)
    F1 =  round(F1, 4)
    Prec =  round(Prec, 4)
    results["Sn"] = Sn
    results["Sp"] = Sp
    results["Bacc"] = Bacc
    results["ACC"] = ACC
    results["MCC"] = MCC
    results["F1"] = F1
    results["Prec"] = Prec
    
    AP = average_precision_score(np.array(total_true_label), np.array(total_pre_logit))
    results["AP"] = AP
    
    AUC = roc_auc_score(np.array(total_true_label), np.array(total_pre_logit))
    results["AUC"] = AUC
    
    # Fmax
    # 计算不同阈值下的Precision、Recall
    precisions, recalls, thresholds = precision_recall_curve(np.array(total_true_label), np.array(total_pre_logit))

    # 计算每个阈值对应的F1分数，选择最大值
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)  # 避免除零
    f1_max = np.nanmax(f1_scores)  # 忽略NaN值
    optimal_threshold = thresholds[np.nanargmax(f1_scores)]
    results["F_max"] = f1_max
    results["F_max_threshold"] = optimal_threshold
    
    # 计算 PR-AUC
    pr_auc = auc(recalls, precisions)
    results["APS-PRAUC"] = pr_auc
    
    for k, v in results.items():
        print("{}:{}".format(k, v))
    
    return results, sorted_total_samples
    
def predict_get_matrics1(ids, seqs, true_label, pre_label, pred_logit, mask):
        results = {}
        total_samples = {}
        total_true_label, total_pre_logit = [], []
        TP = 0
        FN = 0
        TN = 0
        FP = 0
        other = 0

        for i in range(len(pre_label)):
            for j in range(len(pre_label[i])):
                if mask[i][j] == 1:
                    if pre_label[i][j] == 0. and true_label[i][j] == 1.:
                        FN = FN + 1
                    elif pre_label[i][j] == 1. and true_label[i][j] == 1.:
                        TP = TP + 1
                    elif pre_label[i][j] == 0. and true_label[i][j] == 0.:
                        TN = TN + 1
                    elif pre_label[i][j] == 1. and true_label[i][j] == 0.:
                        FP = FP + 1
                    elif true_label[i][j] == '2':
                        other = other + 1
                    total_true_label.append(true_label[i][j])
                    total_pre_logit.append(pred_logit[i][j])
                else: continue
        for i in range(len(pre_label)):
            id = ids[i]
            mask_i = mask[i]
            label_i = true_label[i]
            pred_i = pred_logit[i] #[round(l,4) for l in pred_logit[i]] # 保留小数点后四位
            seq = seqs[i]
            total_samples['{}'.format(id)] = ((seq, label_i, pred_i))
        
        sorted_total_samples = {key: total_samples[key] for key in ids if key in total_samples}
        

        # print('TP =', TP)
        # print('FP =', FP)
        # print('TN =', TN)
        # print('FN =', FN)
        # print('other =', other)
        # print('if the test set is whole ordered, the FPR is adopted!')
        # print('the residue FPR is :', round(FP/(1.0*(FP + TN)), 5))

        if (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) != 0:
            Sn = TP / float(TP + FN)
            Sp = TN / float(TN + FP )
            Bacc = (TP / float(TP + FN) + TN / float(TN + FP)) / 2.0
            MCC = ((TP * TN) - (FP * FN)) / float(sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))
            ACC = (TP + TN )/((TP + TN + FP + FN)*1.0)
            F1 = (2. * TP ) / (2 * TP + FP + FN)
            Prec = TP / float(TP + FP)
        else:
            Sn = 0
            Sp = 0
            Bacc = 0
            MCC = 0
            ACC = 0
            F1 = 0
            Prec = 0
        Sn = round(Sn, 4)
        Sp = round(Sp, 4)
        Bacc = round(Bacc, 4)
        ACC = round(ACC, 4)
        MCC = round(MCC, 4)
        F1 = round(F1, 4)
        Prec = round(Prec, 4)
        results["Sn"] = Sn
        results["Sp"] = Sp
        results["Bacc"] = Bacc
        results["ACC"] = ACC
        results["MCC"] = MCC
        results["F1"] = F1
        results["Prec"] = Prec
        
        AP = average_precision_score(np.array(total_true_label), np.array(total_pre_logit))
        results["AP"] = AP
        
        AUC = roc_auc_score(np.array(total_true_label), np.array(total_pre_logit))
        results["AUC"] = AUC
        for k, v in results.items():
            print("{}:{}".format(k, v))
        
        return results, sorted_total_samples