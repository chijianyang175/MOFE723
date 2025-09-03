'''
Author: Sicen Liu
Date: 2024-12-10 03:55:23
LastEditTime: 2025-06-18 20:07:06
FilePath: /liusicen/shared_files/MOFE/utils/GetPSSM.py
Description: 

Copyright (c) 2024 by ${liusicen_cs@outlook.com}, All Rights Reserved. 
'''
import sys, os, subprocess
import argparse
sys.path.append('Utility')
from multiprocessing import Pool
from Bio import SeqIO
import time
import pickle as pkl
from functools import partial

complet_n=0

class fastaExample(object):
    def __init__(self, id, name, seq, label):
        self.id = id
        self.name = name
        self.seq = seq
        self.label = label

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
            
def generateFasta(input_file):
    print('Generating fasta files...')
    fasta_dir = Profile_HOME+'/fasta' #生成fasta文件夹，存储fasta序列信息
    if not os.path.isdir(fasta_dir):
        os.makedirs(fasta_dir)
    # prot_list = list(SeqIO.parse(input_file, 'fasta')) #保存为序列条目列表 Note：如果只有标题行和序列行可以这样使用
    prot_list = parseFasta_with_label(input_file) # 保存id name seq label
    with open(Profile_HOME+'/id_list.txt','w') as f:
        for protein in prot_list:
            f.write(protein.id + '\n')

    names = [] # 保存所有的protein的名字
    seqs = [] # 保存所有的protein的序列
    for protein in prot_list:
        tname = protein.name.split('|')
        if len(tname) > 1:
            name = tname[1]
        else:
            name = tname[0] #走这一行运行的，其他的都没有作用
        tname = name.split('.')
        if len(tname) > 1:
            name = str(tname[0])+str(tname[1])
        else:
            name = tname[0] #走这一行运行的，输出结果为字符串，及protein.name
        fasta_file = fasta_dir + '/' + name + '.fasta'
        with open(fasta_file, 'w') as wf:
            wf.write('>' + name + '\n')
            wf.write(str(protein.seq) + '\n')
        names.append(name)
        seqs.append(str(protein.seq))
    return names, seqs

def run_simple_search(fd):
    protein_name = fd.split('.')[0]
    global complet_n
    complet_n += 1
    print('Processing:%s---%d' % (protein_name, complet_n*4))
    outfmt_type = 5
    num_iter = 3
    evalue_threshold = 0.001
    num_thred = 4 #20
    fasta_file = Profile_HOME + '/fasta/' + protein_name + '.fasta'
    xml_file = Profile_HOME + '/xml/' + protein_name + '.xml'
    pssm_file = Profile_HOME + '/pssm/' + protein_name + '.pssm'
    if os.path.isfile(pssm_file):
        pass
    else:
        # join是一个字符串操作函数
        # 将join后括号内的每个成员以字符‘ ’分隔开，再拼接成一个整体的字符串
        cmd = ' '.join([BLAST,
                        '-query ' + fasta_file,
                        '-db ' + BLAST_DB,
                        '-out ' + xml_file,
                        '-evalue ' + str(evalue_threshold),
                        '-num_iterations ' + str(num_iter),
                        '-outfmt ' + str(outfmt_type),
                        '-out_ascii_pssm ' + pssm_file,  # Write the pssm file
                        '-num_threads ' + str(num_thred)]
                       )
        subprocess.call(cmd, shell=True)
        
def check(names, target_folder):
    processed = os.listdir(target_folder)
    processed = [p[:-5] for p in processed] # 只保存蛋白质名称

    rest_names = []
    for i in range(len(names)):
        name = names[i]
        fname = name.replace('|', '_') #[1:]
        if fname not in processed:
            rest_names.append(fname)
    return rest_names

def read_blosum(blosum_dir):
    """Read blosum dict and delete some keys and values."""
    blosum_dict = {}
    blosum_reader = open(blosum_dir, 'r')
    count = 0
    for line in blosum_reader:
        count = count + 1
        if count <= 7:
            continue
        line = line.strip('\r').split()
        blosum_dict[line[0]] = [float(x) for x in line[1:21]]

    blosum_dict.pop('*')
    # blosum_dict.pop('B')
    # blosum_dict.pop('Z')
    # blosum_dict.pop('X')
    
    return blosum_dict

def gen_pssm_by_blosum(seq, blosum_dir, trg):

    blosum = read_blosum(blosum_dir)
    enc = []
    for aa in seq:
        if aa in blosum:
            enc.append(blosum[aa])
        else:
            enc.append([0.]*20)
    with open(trg, 'w') as f:
        for i in range(3): f.write('\n')
        for i, s in enumerate(seq):
            str_list = map(str, enc[i])
            f.write(' ' + str(i) + ' ' + s + ' ')
            f.write(' '.join(str_list))
            f.write('\n')
            
            
def generateMSA(file_path):
    Names, Seqs = generateFasta(file_path)
    print('Generating PSSM:')
    fasta_dir = Profile_HOME + '/fasta'
    seq_DIR = os.listdir(fasta_dir) #fasta文件夹内的所有fasta文件名

    pssm_dir = Profile_HOME + '/pssm'
    if not os.path.isdir(pssm_dir):
        os.makedirs(pssm_dir)

    xml_dir = Profile_HOME + '/xml'
    if not os.path.isdir(xml_dir):
        os.makedirs(xml_dir)

    pool = Pool(8) #定义一个进程池，最大进程数为8，并行运行
    results = pool.map(run_simple_search, seq_DIR) #第一个参数传函数，第二个参数传数据列表
    pool.close() #关闭进程池，不再接收新的请求
    pool.join() #等待所有子进程执行完成
    

if __name__ == '__main__':
     # datasets_list = ['DM4845_training', 'DISORDER723_test', 'DisProt832_test', 'S1_test']
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default="/home/liusicen/models/DeepDRP-my/datasets/")
    parser.add_argument('--profile_path', type=str, default="/home/liusicen/methods/DynamicFusion/Datasets/Evo_features/")
    parser.add_argument('--dataset', type=str, default="DM4845_training", help='Dataset name.')
    args = parser.parse_args()
    
    dataset = args.dataset

    global BLAST
    global BLAST_DB
    BLAST = '/home/liusicen/app/ncbi-blast-2.16.0+/bin/psiblast'
    BLAST_DB = '/home/liusicen/app/nrdb90/nrdb90'
    blosum_path = '/home/liusicen/methods/DynamicFusion/utils/psiblast/blosum62.txt'

    file_path = args.file_path + dataset + ".fasta"
    global Profile_HOME
    
    Profile_HOME = args.profile_path + dataset
    if not os.path.isdir(Profile_HOME): #如果没有该路径目录
        os.makedirs(Profile_HOME) #创建路径目录
        
    generateMSA(file_path)
    file_list = os.listdir(Profile_HOME + '/pssm')
    print(f'Total PSSM files for {dataset}: {len(file_list)}')
    
    
    Names, Seqs = [], []
    prot_list = parseFasta_with_label(file_path)
    for prot in prot_list:
        Names.append(prot.name)
        Seqs.append(str(prot.seq))
        
        
    # check 没有生成pssm的序列
    rest_names = check(Names, Profile_HOME + '/pssm')
    print(f'Not generated PSSM: {len(rest_names)}')
    
    for i in range(len(Names)):
        name = Names[i]#[1:]
        if name in rest_names:
            gen_pssm_by_blosum(Seqs[i], blosum_path, Profile_HOME + '/pssm/' + name + '.pssm')
    
    # check final pssm file
    print(f'>>>Total PSSM files for {dataset}: {len(file_list)}')
    
    