import torch
torch.manual_seed(1741)
import random
random.seed(1741)
import numpy as np
np.random.seed(1741)
from os import path
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from utils.constant import CUDA


class SelectorModel(nn.Module):
    def __init__(self, mlp_dim, hidden_dim , roberta_type, is_finetune):
        super().__init__()
        self.roberta_type = roberta_type
        self.is_finetune  = is_finetune
        if path.exists("./pretrained_models/models/{}".format(roberta_type)):
            print("Loading pretrain model from local ......")
            self.encoder = AutoModel.from_pretrained("./pretrained_models/models/{}".format(self.roberta_type), output_hidden_states=True)
        else:
            print("Loading pretrain model ......")
            self.encoder = AutoModel.from_pretrained(self.roberta_type, output_hidden_states=True)
        
        if 'base' in roberta_type:
            self.in_dim = 768
        if 'large' in roberta_type:
            self.in_dim = 1024
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.selector = LSTMSelector(self.in_dim, self.hidden_dim, self.mlp_dim)
    
    def forward(self, target, target_len, ctx, ctx_len, n_step):
        if self.is_finetune == False:
            with torch.no_grad():
                target_emb = self.encode(target)
                ctx_emb = torch.stack(
                    [self.encode(ctx[i]) for i in range(ctx.size(0))], dim=0
                )
        else:
            target_emb = self.encode(target)
            ctx_emb = torch.stack(
                [self.encode(ctx[i]) for i in range(ctx.size(0))], dim=0
            )
        outputs, dist, log_prob = self.selector(target_emb, ctx_emb, target_len, ctx_len, n_step)
        return outputs, dist, log_prob
    
    def encode(self, input_ids):
        return self.encoder(input_ids)[0][:, 0]


class LSTMSelector(nn.Module):
    def __init__(self, in_dim, hidden_dim, mlp_dim, fn_activate='tanh'):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.lstm_cell = nn.LSTMCell(self.in_dim, self.hidden_dim)

        self.mlp_dim = mlp_dim
        
        self.drop_out = nn.Dropout(0.5)

        self.mlp1 = nn.Linear(self.hidden_dim+self.in_dim, self.mlp_dim)
        
        if fn_activate=='relu':
            self.fn_activate1 = nn.LeakyReLU(0.2, True)
        elif fn_activate=='tanh':
            self.fn_activate1 = nn.Tanh()
        elif fn_activate=='relu6':
            self.fn_activate1 = nn.ReLU6()
        elif fn_activate=='silu':
            self.fn_activate1 = nn.SiLU()
        elif fn_activate=='hardtanh':
            self.fn_activate1 = nn.Hardtanh()
        
        self.mlp2 = nn.Linear(self.mlp_dim, 1)
        
        self.fn_activate2 = nn.Sigmoid()
    
    def forward(self, target_emb: torch.Tensor, ctx_emb: torch.Tensor, target_len, ctx_len, n_step):
        outputs = []
        dist = []

        lstm_in = target_emb # bs x dim
        
        bs = lstm_in.size(0)
        ns = ctx_emb.size(1)
        
        h_0 = torch.zeros((bs, self.hidden_dim))
        c_0 = torch.zeros((bs, self.hidden_dim))
        
        # mask for selected sentences
        mask = torch.zeros((bs, ns))
        for i in range(bs):
            if len(ctx_len[i]) < ns:
                mask[i, len(ctx_len[i]):] = -100000
        
        log_probs = torch.zeros((bs))
        if CUDA:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()
            mask = mask.cuda()
            log_probs = log_probs.cuda()
        lstm_state = (h_0, c_0)
        
        for _ in range(n_step):
            lstm_in = self.drop_out(lstm_in)
            h, c = self.lstm_cell(lstm_in, lstm_state) # h: bs x hidden_dim
            
            _ctx_emb = self.drop_out(ctx_emb)
            h = self.drop_out(h)
            
            sc = self.mlp1(torch.cat([_ctx_emb, h.unsqueeze(1).expand((-1, ns, -1))], dim=-1)) # bs x ns x mlp_dim
            sc = self.drop_out(sc)
            sc = self.mlp2(self.fn_activate1(sc)) # bs x ns x 1
            sc = self.fn_activate2(sc.squeeze()) 
            sc = sc + mask

            if self.training:
                probs = F.softmax(sc, dim=-1) # bs x ns
                probs = torch.distributions.Categorical(probs=probs)
                # out = sc.max(dim=-1)[1] # bs x 1: index of selected sentence in this step
                out = probs.sample()
                log_probs = log_probs + probs.log_prob(out)
                dist.append(probs)
                outputs.append(out)
            else:
                out = sc.max(dim=-1)[1] # bs x 1: index of selected sentence in this step
                outputs.append(out)
            
            for i in range(bs):
                mask[i, out[i]] = -100000 # mask selected sentent
            
            lstm_in = torch.gather(ctx_emb, dim=1, index=out.unsqueeze(1).unsqueeze(2).expand(bs, 1, self.in_dim))
            lstm_in = lstm_in.squeeze(1)
            lstm_state = (h, c)
        
        return outputs, dist, log_probs
