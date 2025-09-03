'''
Author: Sicen Liu
Date: 2024-12-10 09:12:48
LastEditTime: 2025-07-15 16:13:44
FilePath: /liusicen/shared_files/MOFE/utils/data_processing.py
Description: Data processing code

Copyright (c) 2024 by ${liusicen_cs@outlook.com}, All Rights Reserved. 
'''
import sys, getopt
import torch
import numpy as np
import re
from transformers import AutoModel, T5Tokenizer, T5Model
from transformers import AutoTokenizer, AutoModelForMaskedLM, FeatureExtractionPipeline
import os
import textwrap
from tqdm import tqdm
import gc

from utils.features.OntoProtein import encode_by_onto
from utils.features.traditionalFeature import encode_by_pssm
from torch.utils.data import Dataset

from sklearn.model_selection import KFold

class fastaExample(object):
    def __init__(self, id, name, seq, label):
        self.id = id
        self.name = name
        self.seq = seq
        self.label = label

class CAID3_fastaExample(object):
    def __init__(self, id, name, seq, label, unexcluded_label_idx):
        self.id = id
        self.name = name
        self.seq = seq
        self.label = label
        self.unexcluded_label_idx = unexcluded_label_idx

def parseFasta_with_label(input_file):
    samples = []
    load_f = []
    with open(input_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line=line.strip('\n')
            load_f.append(line)
    for i in range(len(load_f)):
        if i % 3 == 0:
            id = load_f[i].strip('>').strip('\n\r')
            seq = load_f[i+1]
            label = load_f[i+2]
            samples.append(fastaExample(id, id, seq, label))
    return samples 

def parseCAID3_Fasta_with_label(input_file):
    samples = []
    load_f = []
    with open(input_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line=line.strip('\n')
            load_f.append(line)
    for i in range(len(load_f)):
        if i % 3 == 0:
            id = load_f[i].strip('>').strip('\n\r')
            seq = load_f[i+1]
            label = load_f[i+2]
            
            unexcluded_label_idx = []
            temp_label = []
            for i, (s, l) in enumerate(zip(seq, label)):
                if l == "-" or l== '2':
                    temp_label.append('0')
                    continue
                else:
                    temp_label.append(l)
                    unexcluded_label_idx.append(i)
            temp_label = ''.join(temp_label)
            samples.append(CAID3_fastaExample(id, id, seq, temp_label, unexcluded_label_idx))
    return samples 


"""--------------------------------------------------------"""

device = "cuda" if torch.cuda.is_available() else "cpu"
print("loading ESM-2...")
# model_esm2, alphabet_esm2 = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
model_esm2, alphabet_esm2 = torch.hub.load("/home/liusicen/methods/FusionLSTM/DeepDRP-my/preTrainModels/esm-main", "esm2_t33_650M_UR50D", source="local")
gc.collect()
"""--------------------------------------------------------"""
print("loading DR-BERT...")
# model_name_drbert = "lib/DR-BERT"
model_name_drbert = "/home/liusicen/methods/FusionLSTM/DeepDRP-my/preTrainModels/DR-BERT"
tokenizer_drbert = AutoTokenizer.from_pretrained(model_name_drbert)
model_drbert = AutoModel.from_pretrained(model_name_drbert).to(device).eval()
gc.collect()
"""--------------------------------------------------------"""
def encode_by_drbert(seq):

    def get_hidden_states(encoded, model):
        with torch.no_grad():
            output = model(**encoded)
        # Get last hidden state
        return output.last_hidden_state

    # if len(seq) > 1022, cut it    
    seq = seq.upper()
    if len(seq) > 1022:
        embedding_list = []
        seq_list = textwrap.wrap(seq, 1022)
        for seq in seq_list:
            encoded = tokenizer_drbert.encode_plus(seq, return_tensors="pt").to(device)
            embedding = get_hidden_states(encoded, model_drbert).detach().cpu().numpy()#[0, 1:-1, :embedding_dim]
            embedding_list.append(embedding[0, 1:-1])
        embedding = np.concatenate(embedding_list, axis=0)
    else:
        encoded = tokenizer_drbert.encode_plus(seq, return_tensors="pt").to(device)
        embedding = get_hidden_states(encoded, model_drbert).detach().cpu().numpy()[0, 1:-1]#[0, 1:-1, :embedding_dim]

    # return embedding[:maxsize]
    return embedding

def encode_by_esm2(seq):
    
    batch_converter = alphabet_esm2.get_batch_converter()
    model_esm2.eval()  # disables dropout for deterministic results
    # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
    data = [
        ("tmpDeepDRP", seq.upper()),
    ]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    if len(seq)>2000:#len(seq)>3088:
        cut_num = 1000
    else:
        cut_num = 2000
    # if seq len > 2000, cut it
    if len(seq) > cut_num:
        # 定义切分间隔
        interval = cut_num
        # 切分张量
        tensors = [batch_tokens[:, i:i + interval] for i in range(0, batch_tokens.size(1), interval)]
        results = []
        for tensor in tensors:
            with torch.no_grad():
                result = model_esm2(tensor, repr_layers=[33], return_contacts=True)
                gc.collect()
            token_representation = result["representations"][33] 
            results.append(token_representation)
        token_representations = torch.cat(results, dim=1)
        token_representations = token_representations.detach().cpu().numpy()[0][1:-1,:]
    else:
        with torch.no_grad():
            results = model_esm2(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]
        token_representations = token_representations.detach().cpu().numpy()[0][1:-1,:]

    # embedding = 600
    # maxsize = 500

    # return token_representations[:maxsize, :embedding]
    return token_representations


class InputFastaExample(object):
    """A single example feature of input data"""    
    def __init__(self, id, name, seq, feature1, feature2, label):
        self.id = id
        self.name = name
        self.seq = seq
        self.feature1 = feature1
        self.feature2 = feature2
        self.label = label

def convert_example_to_features(args, examples, data_type='train'):
    """Loads a data file into a list of InputFastaExample."""
    datasets = []
    if args.debug:
        if data_type in ['SL329', 'DISORDER723_test', 'CASP']:
            examples = examples
        else:
            examples = examples[:50]
    for i, example in tqdm(enumerate(examples), desc="Example:"):
        # Perform preprocessing on the sequence
        id = example.id
        seq = example.seq
        label = example.label
        features = []
            
        if args.pssm:
            pssm_feature = encode_by_pssm(args, data_type, id)
            features.append(pssm_feature)
        
        # 由于计算预训练的feature时间过长，需要对计算过程进行单独保持
        # 按照氨基酸的id进行预训练特征的保存

        if args.esm2:
            preT_path = os.path.join(args.preT_feature_path, data_type, "esm2", "{}.npy".format(id))
            if os.path.exists(preT_path):
                preT_esm2_feature = np.load(preT_path)
            else:
                dir_name = os.path.dirname(preT_path)
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                preT_esm2_feature = encode_by_esm2(seq)
                np.save(preT_path, preT_esm2_feature) # save as numpy datatype
            features.append(preT_esm2_feature)

        if args.drBERT:
            preT_path = os.path.join(args.preT_feature_path, data_type, "drBERT", "{}.npy".format(id))
            if os.path.exists(preT_path):
                preT_drbert_feature = np.load(preT_path)
            else:
                dir_name = os.path.dirname(preT_path)
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                preT_drbert_feature = encode_by_drbert(seq)
                np.save(preT_path, preT_drbert_feature) # save as numpy datatype
            features.append(preT_drbert_feature)

        
        if args.feature1 and args.feature2 and len(features) == 2:
            feature1 = features[0]
            feature2 = features[1]
        elif args.feature1 and not args.feature2:
            feature1 = features[0]
            feature2 = None
        

        datasets.append(InputFastaExample(
            id = id,
            name = example.name,
            seq = seq,
            feature1 = feature1,
            feature2 = feature2,
            label = label
        ))
    return datasets    


class CAID3_InputFastaExample(object):
    """A single example feature of input data"""    
    def __init__(self, id, name, seq, feature1, feature2, label, unexcluded_label_idx):
        self.id = id
        self.name = name
        self.seq = seq
        self.feature1 = feature1
        self.feature2 = feature2
        self.label = label
        self.unexcluded_label_idx = unexcluded_label_idx
        
def ensure_numpy_array(obj):
    if not isinstance(obj, np.ndarray):
        obj = np.array(obj)
    return obj

def CAID3_convert_example_to_features(args, examples, data_type='train'):
    """Loads a data file into a list of InputFastaExample."""
    datasets = []
    if args.debug:
        if data_type in ['SL329', 'DISORDER723_test', 'CASP']:
            examples = examples
        else:
            examples = examples[:50]
    for i, example in tqdm(enumerate(examples), desc="Example:"):
        # Perform preprocessing on the sequence
        id = example.id
        seq = example.seq
        label = example.label
        unexcluded_label_idx = example.unexcluded_label_idx
        features = []
        
        if args.pssm:
            pssm_feature = encode_by_pssm(args, data_type, id)
            features.append(pssm_feature)
        
        # 由于计算预训练的feature时间过长，需要对计算过程进行单独保持
        # 按照氨基酸的id进行预训练特征的保存
        if args.esm2:
            preT_path = os.path.join(args.preT_feature_path, data_type, "esm2", "{}.npy".format(id))
            if os.path.exists(preT_path):
                preT_esm2_feature = np.load(preT_path)
            else:
                dir_name = os.path.dirname(preT_path)
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                preT_esm2_feature = encode_by_esm2(seq)
                np.save(preT_path, preT_esm2_feature) # save as numpy datatype
            features.append(preT_esm2_feature)

        if args.drBERT:
            preT_path = os.path.join(args.preT_feature_path, data_type, "drBERT", "{}.npy".format(id))
            if os.path.exists(preT_path):
                preT_drbert_feature = np.load(preT_path)
            else:
                dir_name = os.path.dirname(preT_path)
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                preT_drbert_feature = encode_by_drbert(seq)
                np.save(preT_path, preT_drbert_feature) # save as numpy datatype
            features.append(preT_drbert_feature)

        
        if args.feature1 and args.feature2 and len(features) == 2:
            feature1 = features[0]
            feature2 = features[1]
        elif args.feature1:
            feature1 = features[0]
            feature2 = None
        
        datasets.append(CAID3_InputFastaExample(
            id = id,
            name = example.name,
            seq = seq,
            feature1 = feature1,
            feature2 = feature2,
            label = label,
            unexcluded_label_idx = unexcluded_label_idx
        ))
    return datasets 

def load_dataset(args, fn, dataset_type='train', label=1):
    '''
    Author: Sicen Liu
    Description: process fasta file
    param {*} args : args
    param {*} fn : source file path of fasta format with label
    param {*} label : default label for disordered residues
    return {
        ids: name list
        seqs: residue list
        labels: label list
    }
    '''
    if dataset_type in ['CAID3_disorder_pdb', 'CAID3_disorder_nox','CASP','SL329']:
        prot_list = parseCAID3_Fasta_with_label(fn)
        dataset = CAID3_convert_example_to_features(args, prot_list, data_type=dataset_type)
    else:
        prot_list = parseFasta_with_label(fn)
        dataset = convert_example_to_features(args, prot_list, data_type=dataset_type)
    return dataset

class proteinDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    def get_id(self, index):
        return self.samples[index].id
    def get_seq(self, index):
        return self.samples[index].seq
    
    
def kfold_split(dataset, k=5, seed=42):
    train_datasets = []
    val_datasets = []
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    for train_index, test_index in kf.split(dataset):
        train_data = [dataset[i] for i in train_index]
        val_data = [dataset[i] for i in test_index]
        train_datasets.append(train_data)
        val_datasets.append(val_data)
    return train_datasets, val_datasets
        
        
def collate_fn_batch(batchs, args):
    ids = []
    seqs = []
    ori_labels = []
    labels = []
    features1 = []
    features2 = []
    batch_size = len(batchs)
    max_seq_len = max([len(seq.label) for seq in batchs])
    labels = torch.full((batch_size, max_seq_len), 0.)
    input_mask = torch.full((batch_size, max_seq_len), 0.)

    if args.feature1:
        feature1_dim = batchs[0].feature1.shape[-1]
        input_features1 = torch.full((batch_size, max_seq_len, feature1_dim), 0.)
    else:
        input_features1 = None
    if args.feature2:
        feature2_dim = batchs[0].feature2.shape[-1]
        input_features2 = torch.full((batch_size, max_seq_len, feature2_dim), 0.)
    else:
        input_features2 = None
    for b, batch in enumerate(batchs):
        id = batch.id
        name = batch.name
        seq = batch.seq
        feature1 = batch.feature1
        feature2 = batch.feature2
        label = batch.label
        if args.feature1:
            for s, fea1 in enumerate(feature1):
                input_features1[b, s, :] = torch.from_numpy(fea1)
        if args.feature2:
            for s, fea2 in enumerate(feature2):
                input_features2[b, s, :] = torch.from_numpy(fea2)
        
        for s in range(len(label)):
            input_mask[b, s] = 1.
            labels[b, s] = 1. if label[s] == '1' else 0
            
        ids.append(id)
        seqs.append(id)
        ori_labels.append(label)        
       
    return  id, seq, ori_labels, input_features1, input_features2, input_mask, labels

def CAID3_collate_fn_batch(batchs, args):
    ids = []
    seqs = []
    ori_labels = []
    labels = []
    tri_features = []
    preT_features = []
    batch_size = len(batchs)
    
    max_seq_len = max([len(seq.label) for seq in batchs])
    if args.feature1:
        feature1_dim = batchs[0].feature1.shape[-1]
        input_features1 = torch.full((batch_size, max_seq_len, feature1_dim), 0.)
    else:
        input_features1 = None
    if args.feature2:
        feature2_dim = batchs[0].feature2.shape[-1]
        input_features2 = torch.full((batch_size, max_seq_len, feature2_dim), 0.)
    else:
        input_features2 = None
    labels = torch.full((batch_size, max_seq_len), 0.)
    input_mask = torch.full((batch_size, max_seq_len), 0.)
    for b, batch in enumerate(batchs):
        id = batch.id
        name = batch.name
        seq = batch.seq
        feature1 = batch.feature1
        feature2 = batch.feature2
        label = batch.label
        unexcluded_label_idx = batch.unexcluded_label_idx
        if args.feature1:
            for s, fea1 in enumerate(feature1):
                input_features1[b, s, :] = torch.from_numpy(fea1)
        if args.feature2:
            for s, fea2 in enumerate(feature2):
                input_features2[b, s, :] = torch.from_numpy(fea2)
        
        for s in range(len(label)):
            if s in unexcluded_label_idx:
                input_mask[b, s] = 1.
            labels[b, s] = 1. if label[s] == '1' else 0
            
        ids.append(id)
        seqs.append(id)
        ori_labels.append(label)        
       
    return  id, seq, ori_labels, input_features1, input_features2, input_mask, labels