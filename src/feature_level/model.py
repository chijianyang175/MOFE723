'''
Author: Sicen Liu
Date: 2025-05-16 14:37:10
LastEditTime: 2025-06-19 15:45:36
FilePath: /liusicen/shared_files/MOFE/src/feature_level/model.py
Description: model part

Copyright (c) 2025 by ${liusicen_cs@outlook.com}, All Rights Reserved. 
'''

import torch 
import torch.nn as nn
from src.feature_level.modules import TransformerEncoder

from collections import OrderedDict
   
class Encoder(nn.Module):
    def __init__(self,feature1_dim, feature2_dim, hidden_dim, heads, n_layers=2, dropout=0.1):
        super(Encoder, self).__init__()
        self.feature1_dim = feature1_dim
        self.feature2_dim = feature2_dim
        self.hidden_dim = hidden_dim
        self.trans = nn.Linear(feature1_dim+feature2_dim, hidden_dim)
        self.encoder = TransformerEncoder(hidden_dim, heads=heads, depth=n_layers, dropout=dropout)
        self.output_layer = nn.Linear(hidden_dim, 1)
        
        
    def forward(self, evo_input, preT_input):
        if evo_input is not None:
            inputs = evo_input
        elif preT_input is not None:
            inputs = preT_input
        elif evo_input is not None and preT_input is not None:
            inputs = torch.cat([evo_input, preT_input], dim=-1)

        batch_size, seq_len, _ = inputs.size()

        inputs = self.trans(inputs)
        output = self.encoder(inputs)
        output = self.output_layer(output)
        output = output.squeeze(-1)
        # output = torch.sigmoid(output)
        return output     

class BiLSTM(nn.Module):
    def __init__(self, feature1_dim, feature2_dim, hidden_dim, n_layers=2, dropout=0.1):
        super(BiLSTM, self).__init__()
        self.feature1_dim = feature1_dim
        self.feature2_dim = feature2_dim
        self.hidden_dim = hidden_dim
        self.bilstm = nn.LSTM(input_size=feature1_dim+feature2_dim, hidden_size=hidden_dim, num_layers=1,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.output_layer = nn.Linear(hidden_dim*2, 1)
        
        
    def forward(self, input1, input2):
        if input1 is not None and input2 is not None:
            inputs = torch.cat([input1, input2], dim=-1)
        elif input1 is not None:
            inputs = input1
        elif input2 is not None:
            inputs = input2
        

        batch_size, seq_len, _ = inputs.size()

        output, _ = self.bilstm(inputs)
        
        output = self.output_layer(output)
        output = output.squeeze(-1)
        return output 
    

def DiffSoftmax(logits, tau=1.0, hard=False, dim=-1):
    # y_soft = (logits / tau).softmax(dim)
    gumbel_noise = -torch.empty_like(logits).exponential_().log()
    noisy_logits = (logits + gumbel_noise) / tau
    y_soft = nn.functional.softmax(noisy_logits, dim=-1)
    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

class FL_DynamicFusion(nn.Module):
    def __init__(self, feature1_dim, feature2_dim, hidden_dim, hard_gate=True, dropout=0.1, freeze=True, expert1_path=None, expert2_path=None, expert3_path=None,infer_mode=0):
        super(FL_DynamicFusion, self).__init__()
        self.feature1_dim = feature1_dim
        self.feature2_dim = feature2_dim
        self.hidden_dim = hidden_dim

        self.expert1 = self.model_init(expert1_path, feature1_dim, 0, hidden_dim)  
        self.expert2 = self.model_init(expert2_path, 0, feature2_dim, hidden_dim) 
        self.expert3 = self.model_init(expert3_path, feature1_dim, feature2_dim, hidden_dim) # Load the  expert1+epert2 model here, 

        if freeze:
            self.freeze_branch(self.expert1)
            self.freeze_branch(self.expert2)
            self.freeze_branch(self.expert3)

        # gating nework
        self.gate = nn.Sequential(
            nn.Linear(feature1_dim + feature2_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # 3 experts
        )
        self.temp = 1
        self.hard_gate = hard_gate
        self.weight_list = torch.Tensor()
        self.store_weight = False
        self.infer_mode = infer_mode
        self.flop = torch.Tensor([1.25261, 10.86908])

    def model_init(self, model_path, feature1_dim, feature2_dim, hidden_dim, dropout=0.1):
        bilstm = BiLSTM(feature1_dim=feature1_dim, feature2_dim=feature2_dim, hidden_dim=hidden_dim, n_layers=1, dropout=dropout)
        expert = self.load_model(bilstm, model_path)
        return expert
    
    def load_model(self, model, saved_model_path):
        try:
            model_dict = torch.load(saved_model_path, map_location=torch.device('cpu'))
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
        except Exception as e:
            model.load_state_dict(saved_model_path)
        return model


    def freeze_branch(self, branch):
        for param in branch.parameters():
            param.requires_grad = False
    
    def reset_weight(self):
        self.weight_list = torch.Tensor()
        self.store_weight = True

    def weight_stat(self):
        print(self.weight_list)
        tmp = torch.mean(self.weight_list, dim=0)
        print(f'mean branch weight {tmp[0].item():.4f}, {tmp[1].item():.4f}')
        self.store_weight = False
        return tmp[1].item()

    def cal_flop(self):
        tmp = torch.mean(self.weight_list, dim=0)
        total_flop = (self.flop * tmp).sum()
        print(f'Total Flops {total_flop.item():.2f}M')
        return total_flop.item()
        
        
    def forward(self, input1, input2):

        inputs = torch.cat([input1, input2], dim=-1)
        
        weight = DiffSoftmax(self.gate(inputs), tau=self.temp, hard=self.hard_gate)

        if self.store_weight:
            self.weight_list = torch.cat((self.weight_list, weight.cpu()))

        pred_list = [self.expert1(input1, input2=None), self.expert2(input1=None, input2=input2), self.expert3(input1, input2)]
        # if self.infer_mode > 0:
        #     return pred_list[self.infer_mode - 1], 0

        output = weight[:, :, 0:1].squeeze(-1) * pred_list[0] + weight[:, :, 1:2].squeeze(-1) * pred_list[1] + weight[:, :, 2:3].squeeze(-1) * pred_list[2]
        # gate: (B, L, N)
        B, L, N = weight.shape

        # 所有 token 分配到各个 expert 的总权重
        importance = weight.sum(dim=(0, 1))  # shape: (N,)
        # 统计每个 expert 被激活的 token 数
        load = (weight > 0).float().sum(dim=(0, 1))  # shape: (N,)
        importance = importance / importance.sum()  # shape: (N,)
        load = load / load.sum()                    # shape: (N,)
        # Gshard-style
        aux_loss = N * (importance * load).sum()

        return output, aux_loss
    