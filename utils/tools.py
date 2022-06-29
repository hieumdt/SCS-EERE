import datetime
from typing import List
import numpy as np
np.random.seed(1741)
import torch
torch.manual_seed(1741)
import random
random.seed(1741)
import spacy
from transformers import RobertaTokenizer
from utils.constant import *


tokenizer = RobertaTokenizer.from_pretrained('/vinai/hieumdt/pretrained_models/tokenizers/roberta-base', unk_token='<unk>')
nlp = spacy.load("en_core_web_sm")


# Padding function
def padding(sent, pos = False, max_sent_len = 194):
    if pos == False:
        one_list = [1] * max_sent_len # pad token id
        mask = [0] * max_sent_len
        one_list[0:len(sent)] = sent
        mask[0:len(sent)] = [1] * len(sent)
        return one_list, mask
    else:
        one_list = [0] * max_sent_len # none id 
        one_list[0:len(sent)] = sent
        return one_list
      
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def RoBERTa_list(content, token_list = None, token_span_SENT = None):
    encoded = tokenizer.encode(content)
    roberta_subword_to_ID = encoded
    # input_ids = torch.tensor(encoded).unsqueeze(0)  # Batch size 1
    # outputs = model(input_ids)
    # last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
    roberta_subwords = []
    roberta_subwords_no_space = []
    for index, i in enumerate(encoded):
        r_token = tokenizer.decode([i])
        if r_token != " ":
            roberta_subwords.append(r_token)
            if r_token[0] == " ":
                roberta_subwords_no_space.append(r_token[1:])
            else:
                roberta_subwords_no_space.append(r_token)

    roberta_subword_span = tokenized_to_origin_span(content, roberta_subwords_no_space[1:-1]) # w/o <s> and </s>
    roberta_subword_map = []
    if token_span_SENT is not None:
        roberta_subword_map.append(-1) # "<s>"
        for subword in roberta_subword_span:
            roberta_subword_map.append(token_id_lookup(token_span_SENT, subword[0], subword[1]))
        roberta_subword_map.append(-1) # "</s>" 
        return roberta_subword_to_ID, roberta_subwords, roberta_subword_span, roberta_subword_map
    else:
        return roberta_subword_to_ID, roberta_subwords, roberta_subword_span, -1

def tokenized_to_origin_span(text: str, token_list: List[str]):
    token_span = []
    pointer = 0
    for token in token_list:
        start = text.find(token, pointer)
        if start != -1:
            end = start + len(token) - 1
            pointer = end + 1
            token_span.append([start, end])
            assert text[start: end+1] == token, f"token: {token} - text:{text}"
        else:
            token_span.append([-100, -100])
    return token_span

def sent_id_lookup(my_dict, start_char, end_char = None):
    for sent_dict in my_dict['sentences']:
        if end_char is None:
            if start_char >= sent_dict['sent_start_char'] and start_char <= sent_dict['sent_end_char']:
                return sent_dict['sent_id']
        else:
            if start_char >= sent_dict['sent_start_char'] and end_char <= sent_dict['sent_end_char']:
                return sent_dict['sent_id']

def token_id_lookup(token_span_SENT, start_char, end_char):
    for index, token_span in enumerate(token_span_SENT):
        if start_char >= token_span[0] and end_char <= token_span[1]:
            return index

def span_SENT_to_DOC(token_span_SENT, sent_start):
    token_span_DOC = []
    #token_count = 0
    for token_span in token_span_SENT:
        start_char = token_span[0] + sent_start
        end_char = token_span[1] + sent_start
        #assert my_dict["doc_content"][start_char] == sent_dict["tokens"][token_count][0]
        token_span_DOC.append([start_char, end_char])
        #token_count += 1
    return token_span_DOC

def id_lookup(span_SENT, start_char):
    # this function is applicable to RoBERTa subword or token from ltf/spaCy
    # id: start from 0
    token_id = -1
    for token_span in span_SENT:
        token_id += 1
        if token_span[0] <= start_char and token_span[1] >= start_char:
            return token_id
    raise ValueError("Nothing is found. \n span sentence: {} \n start_char: {}".format(span_SENT, start_char))

def pos_to_id(sent_pos):
    id_pos_sent =  [pos_dict.get(pos) if pos_dict.get(pos) != None else 0 
                    for pos in sent_pos]
    return id_pos_sent
            
def pad_to_max_ns(ctx_augm_emb):
    max_ns = 0
    ctx_augm_emb_paded = []
    for ctx in ctx_augm_emb:
        # print(ctx.size())
        max_ns  = max(max_ns, ctx.size(0))
    
    for ctx in ctx_augm_emb:
        pad = torch.zeros((max_ns, 768))
        if ctx.size(0) < max_ns:
            pad[:ctx.size(0), :] = ctx
            ctx_augm_emb_paded.append(pad)
        else:
            ctx_augm_emb_paded.append(ctx)
    return ctx_augm_emb_paded

def word_dropout(seq_id, position, is_word=True, dropout_rate=0.05):
    if is_word==True:
        drop_sent = [3 if np.random.rand() < dropout_rate and i not in position else seq_id[i] for i in range(len(seq_id))]
    if is_word==False:
        drop_sent = [20 if np.random.rand() < dropout_rate and i not in position else seq_id[i] for i in range(len(seq_id))]
    # print(drop_sent)
    return drop_sent

def create_target(x_sent, y_sent, x_sent_id, y_sent_id, x_position, y_position):
    if x_sent_id < y_sent_id:
        sent = x_sent + y_sent[1:] # <s> x_sent </s> y_sent </s>
        y_position_new = y_position + len(x_sent) - 1
        x_position_new = x_position
        assert y_sent[y_position] == sent[y_position_new]
        assert x_sent[x_position] == sent[x_position_new]
    elif x_sent_id == y_sent_id:
        assert x_sent == y_sent
        sent = x_sent
        x_position_new = x_position
        y_position_new = y_position
        assert y_sent[y_position] == sent[y_position_new]
        assert x_sent[x_position] == sent[x_position_new]
    else:
        sent = y_sent + x_sent[1:]
        y_position_new = y_position
        x_position_new = x_position + len(y_sent) - 1
        assert y_sent[y_position] == sent[y_position_new]
        assert x_sent[x_position] == sent[x_position_new]
    return sent, x_position_new, y_position_new

def make_predictor_input(x_sent, y_sent, x_sent_pos, y_sent_pos, x_sent_id, y_sent_id, x_possition, y_possition, ctx, pos_ctx, ctx_id, doc_id, dropout_rate=0.05, is_test=False):
    bs = len(x_sent)
    assert len(ctx) == bs and len(x_sent) == bs and len(x_possition) == bs, 'Each element must be same batch size'
    augm_target = []
    augm_target_mask = []
    augm_pos_target = []
    _augm_target = []
    _augm_pos_target = []
    x_augm_position = []
    y_augm_position = []
    max_len = 0
    for i in range(bs):
        if ctx_id == 'all':
            selected_ctx = list(range(len(ctx[i])))
        elif ctx_id == 'warming':
            selected_ctx = []
        else:
            selected_ctx = [step[i].cpu().item() for step in ctx_id]
        augment, x_possition_new, y_possition_new = augment_target(x_sent[i], y_sent[i], x_sent_id[i], y_sent_id[i], x_possition[i], y_possition[i], 
                                                                ctx[i], selected_ctx, doc_id[i])
        pos_augment, x_pos_possition_new, y_pos_possition_new = augment_target(x_sent_pos[i], y_sent_pos[i], x_sent_id[i], y_sent_id[i], 
                                                                            x_possition[i], y_possition[i], pos_ctx[i], selected_ctx, doc_id[i], is_pos=True)
        # assert x_possition_new == x_pos_possition_new
        # assert y_possition_new == y_pos_possition_new
        if is_test == False:
            augment = word_dropout(augment, [x_possition_new, y_possition_new], dropout_rate=dropout_rate)
            pos_augment = word_dropout(pos_augment, [x_possition_new, y_possition_new], is_word=False, dropout_rate=dropout_rate)
        # print(len(augment))
        max_len = max(len(augment), max_len)
        
        x_augm_position.append(x_possition_new)
        y_augm_position.append(y_possition_new)
        _augm_target.append(augment)
        _augm_pos_target.append(pos_augment)

    for i in range(bs):
        _augment = _augm_target[i]
        _pos_augment = _augm_pos_target[i]
        pad, mask = padding(_augment, max_sent_len=max_len)
        augm_target.append(pad)
        augm_target_mask.append(mask)
        augm_pos_target.append(padding(_pos_augment, pos=True, max_sent_len=max_len))

    augm_target = torch.tensor(augm_target)
    augm_target_mask = torch.tensor(augm_target_mask)
    augm_pos_target = torch.tensor(augm_pos_target)
    x_augm_position = torch.tensor(x_augm_position)
    y_augm_position = torch.tensor(y_augm_position)
    return augm_target, augm_target_mask, augm_pos_target, x_augm_position, y_augm_position

def augment_target(x_sent, y_sent, x_sent_id, y_sent_id, x_possition, y_possition, ctx, ctx_id, doc_id, is_pos=False):
    id_left = []
    id_cent = []
    id_right = []
    if x_sent_id < y_sent_id:
        for id in ctx_id:
            assert doc_id[id] != x_sent_id
            assert doc_id[id] != y_sent_id
            if doc_id[id] < x_sent_id:
                id_left.append(id)
            elif doc_id[id] > y_sent_id:
                id_right.append(id)
            else:
                id_cent.append(id)
    elif x_sent_id == y_sent_id:
        for id in ctx_id:
            assert doc_id[id] != x_sent_id
            assert doc_id[id] != y_sent_id
            assert x_sent == y_sent
            if doc_id[id] < x_sent_id:
                id_left.append(id)
            elif doc_id[id] > x_sent_id:
                id_right.append(id)
    else:
        for id in ctx_id:
            assert doc_id[id] != x_sent_id
            assert doc_id[id] != y_sent_id
            if doc_id[id] < y_sent_id:
                id_left.append(id)
            elif doc_id[id] > x_sent_id:
                id_right.append(id)
            else:
                id_cent.append(id)
    
    sent_left = []
    for id in sorted(id_left):
        sent_left += ctx[id][1:-1]
    sent_cent = []
    for id in sorted(id_cent):
        sent_cent += ctx[id][1:-1]
    sent_right = []
    for id in sorted(id_right):
        sent_right += ctx[id][1:-1]
    
    sent = []
    if x_sent_id < y_sent_id:
        if is_pos:
            sent = [0] + sent_left + [0] + x_sent[1:] + sent_cent + [0] + y_sent[1:] + sent_right + [0]
        else:
            sent = [0] + sent_left + [2] + x_sent[1:] + sent_cent + [2] + y_sent[1:] + sent_right + [2]
        x_possition_new = 1 + x_possition + len(sent_left)
        y_possition_new = 1 + len(sent_left) + len(x_sent) + len(sent_cent) + y_possition
    elif x_sent_id == y_sent_id:
        if is_pos:
            sent = [0] + sent_left + [0] + x_sent[1:] + sent_right + [0]
        else:
            sent = [0] + sent_left + [2] + x_sent[1:] + sent_right + [2]
        x_possition_new = 1 + x_possition + len(sent_left)
        y_possition_new = 1 + y_possition + len(sent_left)
    else:
        if is_pos:
            sent = [0] + sent_left + [0] + y_sent[1:] + sent_cent + [0] + x_sent[1:] + sent_right + [0]
        else:
            sent = [0] + sent_left + [2] + y_sent[1:] + sent_cent + [2] + x_sent[1:] + sent_right + [2]
        y_possition_new = 1 + y_possition + len(sent_left)
        x_possition_new = 1 + len(sent_left) + len(y_sent) + len(sent_cent) + x_possition
    
    assert sent[x_possition_new] == x_sent[x_possition]
    assert sent[y_possition_new] == y_sent[y_possition]

    return sent, x_possition_new, y_possition_new

def processing_vague(logits, threshold, vague_id):
    bs = logits.size(0)
    predicts = []
    for i in range(bs):
        logit = logits[i].detach()
        logit = torch.softmax(logit, dim=0)
        entropy = - torch.sum(logit * torch.log(logit)).cpu().item()
        if entropy > threshold:
            predict = vague_id
        else:
            predict = torch.max(logit.unsqueeze(0), 1).indices.cpu().item()
        predicts.append(predict)
    # print(predicts)
    return predicts


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def collate_fn(batch):
    return tuple(zip(*batch))
