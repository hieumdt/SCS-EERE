import torch
torch.manual_seed(1741)
import random
random.seed(1741)
import numpy as np
np.random.seed(1741)
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel
from utils.constant import *
import os.path as path


class ECIRobertaJointTask(nn.Module):
    def __init__(self, mlp_size, roberta_type, datasets,
                finetune=True, pos_dim=None, loss=None, sub=True, mul=True, fn_activate='relu',
                negative_slope=0.2, drop_rate=0.5, task_weights=None, kg_emb_dim=300, lstm=False):
        super().__init__()
        
        if path.exists("/vinai/hieumdt/pretrained_models/models/{}".format(roberta_type)):
            print("Loading pretrain model from local ......")
            self.roberta = RobertaModel.from_pretrained("/vinai/hieumdt/pretrained_models/models/{}".format(roberta_type), output_hidden_states=True)
        else:
            print("Loading pretrain model ......")
            self.roberta = RobertaModel.from_pretrained(roberta_type, output_hidden_states=True)
        
        if roberta_type == 'roberta-base':
            self.roberta_dim = 768
        if roberta_type == 'roberta-large':
            self.roberta_dim = 1024

        self.sub = sub
        self.mul = mul
        self.finetune = finetune
        
        if pos_dim != None:
            self.is_pos_emb = True
            pos_size = len(pos_dict.keys())
            self.pos_emb = nn.Embedding(pos_size, pos_dim)
            self.mlp_in = self.roberta_dim + pos_dim
        else:
            self.is_pos_emb = False
            self.mlp_in = self.roberta_dim
        
        self.mlp_size = mlp_size
        
        self.drop_out = nn.Dropout(drop_rate)
        
        if fn_activate=='relu':
            self.relu = nn.LeakyReLU(negative_slope, True)
        elif fn_activate=='tanh':
            self.relu = nn.Tanh()
        elif fn_activate=='relu6':
            self.relu = nn.ReLU6()
        elif fn_activate=='silu':
            self.relu = nn.SiLU()
        elif fn_activate=='hardtanh':
            self.relu = nn.Hardtanh()
        
        self.max_num_class = 0

        module_dict = {}
        loss_dict = {}
        for dataset in datasets:
            if dataset == "HiEve":
                num_classes = 4
                if self.max_num_class < num_classes:
                    self.max_num_class = num_classes
                
                if sub==True and mul==True:
                    fc1 = nn.Linear(self.mlp_in*4, int(self.mlp_size*2))
                    fc2 = nn.Linear(int(self.mlp_size*2), num_classes)
                
                if (sub==True and mul==False) or (sub==False and mul==True):
                    fc1 = nn.Linear(self.mlp_in*3, int(self.mlp_size*1.5))
                    fc2 = nn.Linear(int(self.mlp_size*1.5), num_classes)
                
                if sub==False and mul==False:
                    fc1 = nn.Linear(self.mlp_in*2, int(self.mlp_size))
                    fc2 = nn.Linear(int(self.mlp_size), num_classes)

                weights = [993.0/333, 993.0/349, 933.0/128, 933.0/453]
                weights = torch.tensor(weights)
                loss = nn.CrossEntropyLoss(weight=weights)

                module_dict['1'] = nn.Sequential(OrderedDict([
                                                ('dropout1',self.drop_out), 
                                                ('fc1', fc1), 
                                                ('dropout2', self.drop_out), 
                                                ('relu', self.relu), 
                                                ('fc2',fc2) ]))
                loss_dict['1'] = loss
            
            if dataset == "MATRES":
                num_classes = 4
                if self.max_num_class < num_classes:
                    self.max_num_class = num_classes
                
                if sub==True and mul==True:
                    fc1 = nn.Linear(self.mlp_in*4, int(self.mlp_size*2))
                    fc2 = nn.Linear(int(self.mlp_size*2), num_classes)
                
                if (sub==True and mul==False) or (sub==False and mul==True):
                    fc1 = nn.Linear(self.mlp_in*3, int(self.mlp_size*1.5))
                    fc2 = nn.Linear(int(self.mlp_size*1.5), num_classes)
                
                if sub==False and mul==False:
                    fc1 = nn.Linear(self.mlp_in*2, int(self.mlp_size))
                    fc2 = nn.Linear(int(self.mlp_size), num_classes)
                
                weights = [30.0/412, 30.0/263, 30.0/30, 30.0/113,]
                weights = torch.tensor(weights)
                loss = nn.CrossEntropyLoss(weight=weights)

                module_dict['2'] = nn.Sequential(OrderedDict([
                                                ('dropout1',self.drop_out), 
                                                ('fc1', fc1), 
                                                ('dropout2', self.drop_out), 
                                                ('relu', self.relu), 
                                                ('fc2',fc2) ]))
                loss_dict['2'] = loss
            
            if dataset == "I2B2":
                num_classes = 3
                if self.max_num_class < num_classes:
                    self.max_num_class = num_classes
                
                if sub==True and mul==True:
                    fc1 = nn.Linear(self.mlp_in*4, int(self.mlp_size*2))
                    fc2 = nn.Linear(int(self.mlp_size*2), num_classes)
                
                if (sub==True and mul==False) or (sub==False and mul==True):
                    fc1 = nn.Linear(self.mlp_in*3, int(self.mlp_size*1.5))
                    fc2 = nn.Linear(int(self.mlp_size*1.5), num_classes)
                
                if sub==False and mul==False:
                    fc1 = nn.Linear(self.mlp_in*2, int(self.mlp_size))
                    fc2 = nn.Linear(int(self.mlp_size), num_classes)
                
                weights = [213.0/368, 213.0/213, 213.0/1013]
                weights = torch.tensor(weights)
                loss = nn.CrossEntropyLoss(weight=weights)

                module_dict['3'] = nn.Sequential(OrderedDict([
                                                ('dropout1',self.drop_out), 
                                                ('fc1', fc1), 
                                                ('dropout2', self.drop_out), 
                                                ('relu', self.relu), 
                                                ('fc2',fc2) ]))
                loss_dict['3'] = loss
            
            if dataset == "TBD":
                num_classes = 6
                if self.max_num_class < num_classes:
                    self.max_num_class = num_classes
                
                if sub==True and mul==True:
                    fc1 = nn.Linear(self.mlp_in*4, int(self.mlp_size*2))
                    fc2 = nn.Linear(int(self.mlp_size*2), num_classes)
                
                if (sub==True and mul==False) or (sub==False and mul==True):
                    fc1 = nn.Linear(self.mlp_in*3, int(self.mlp_size*1.5))
                    fc2 = nn.Linear(int(self.mlp_size*1.5), num_classes)
                
                if sub==False and mul==False:
                    fc1 = nn.Linear(self.mlp_in*2, int(self.mlp_size))
                    fc2 = nn.Linear(int(self.mlp_size), num_classes)
                
                weights = [41.0/387, 41.0/287, 41.0/64, 41.0/74, 41.0/41, 41.0/642]
                weights = torch.tensor(weights)
                loss = nn.CrossEntropyLoss(weight=weights)

                module_dict['4'] = nn.Sequential(OrderedDict([
                                                ('dropout1',self.drop_out), 
                                                ('fc1', fc1), 
                                                ('dropout2', self.drop_out), 
                                                ('relu', self.relu), 
                                                ('fc2',fc2) ]))
                loss_dict['4'] = loss

            if "TDD" in dataset:
                num_classes = 5
                if self.max_num_class < num_classes:
                    self.max_num_class = num_classes
                
                if sub==True and mul==True:
                    fc1 = nn.Linear(self.mlp_in*4, int(self.mlp_size*2))
                    fc2 = nn.Linear(int(self.mlp_size*2), num_classes)
                
                if (sub==True and mul==False) or (sub==False and mul==True):
                    fc1 = nn.Linear(self.mlp_in*3, int(self.mlp_size*1.5))
                    fc2 = nn.Linear(int(self.mlp_size*1.5), num_classes)
                
                if sub==False and mul==False:
                    fc1 = nn.Linear(self.mlp_in*2, int(self.mlp_size))
                    fc2 = nn.Linear(int(self.mlp_size), num_classes)
                
                weights = [41.0/387, 41.0/287, 41.0/64, 41.0/74, 41.0/41]
                weights = torch.tensor(weights)
                loss = nn.CrossEntropyLoss(weight=weights)

                module_dict['5'] = nn.Sequential(OrderedDict([
                                                ('dropout1',self.drop_out), 
                                                ('fc1', fc1), 
                                                ('dropout2', self.drop_out), 
                                                ('relu', self.relu), 
                                                ('fc2',fc2) ]))
                loss_dict['5'] = loss
            
        self.module_dict = nn.ModuleDict(module_dict)
        self.loss_dict = nn.ModuleDict(loss_dict)
        
        self.task_weights = task_weights
        if self.task_weights != None:
            assert len(self.task_weights)==len(datasets), "Length of weight is difference number datasets: {}".format(len(self.task_weights))

    def forward(self, sent, sent_mask, x_position, y_position, xy, flag, sent_pos=None):
        batch_size = sent.size(0)

        if self.finetune:
            output = self.roberta(sent, sent_mask)[2]
        else:
            with torch.no_grad():
                output = self.roberta(sent, sent_mask)[2]
        
        output = torch.max(torch.stack(output[-4:], dim=0), dim=0)[0]
        
        if sent_pos != None:
            pos = self.pos_emb(sent_pos)
            output = torch.cat([output, pos], dim=2)

        output = self.drop_out(output)
        # print(output_x.size())
        output_A = torch.cat([output[i, x_position[i], :].unsqueeze(0) for i in range(0, batch_size)])
        
        output_B = torch.cat([output[i, y_position[i], :].unsqueeze(0) for i in range(0, batch_size)])
        
        if self.sub and self.mul:
            sub = torch.sub(output_A, output_B)
            mul = torch.mul(output_A, output_B)
            presentation = torch.cat([output_A, output_B, sub, mul], 1)
        if self.sub==True and self.mul==False:
            sub = torch.sub(output_A, output_B)
            presentation = torch.cat([output_A, output_B, sub], 1)
        if self.sub==False and self.mul==True:
            mul = torch.mul(output_A, output_B)
            presentation = torch.cat([output_A, output_B, mul], 1)
        if self.sub==False and self.mul==False:
            presentation = torch.cat([output_A, output_B], 1)
    
        loss = 0.0
        logits = []
        for i in range(0, batch_size):
            typ = str(flag[i].item())
            logit = self.module_dict[typ](presentation[i])
            pad_logit = torch.zeros((1,self.max_num_class))
            pad_logit = pad_logit - 1000
            pad_logit[:, :len(logit)] = logit
            logit = logit.unsqueeze(0)
            target = xy[i].unsqueeze(0)
            if self.training:
                if self.task_weights == None:
                    loss += self.loss_dict[typ](logit, target)
                else:
                    loss += self.task_weights[typ]*self.loss_dict[typ](logit, target)
            
            logits.append(pad_logit)
        return torch.cat(logits, 0), loss
