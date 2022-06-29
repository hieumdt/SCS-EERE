from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
import torch

import tqdm
from data_loader.EventDataset import EventDataset
from data_loader.loader import loader
from torch.utils.data.dataloader import DataLoader
from models.predictor_model import ECIRobertaJointTask
from models.selector_model import LSTMSelector
from transformers import RobertaConfig
from utils.constant import CUDA, temp_num_map
from utils.tools import collate_fn, make_predictor_input, pad_to_max_ns


def inference(params, file):

    RobertaConfig._to_dict_new = RobertaConfig.to_dict

    selector = LSTMSelector(768, params['s_hidden_dim'], params['s_mlp_dim'])
    predictor =ECIRobertaJointTask(mlp_size=params['p_mlp_dim'], roberta_type=roberta_type, datasets=datasets, pos_dim=16, 
                                    fn_activate=params['fn_activative'], drop_rate=params['drop_rate'], task_weights=None)
    
    selector.load_state_dict(torch.load('MATRES_model/selector.pt'))
    predictor.load_state_dict(torch.load('MATRES_model/predictor.pt'))
    
    selector.eval()
    predictor.eval()
    if CUDA:
        selector = selector.cuda()
        predictor = predictor.cuda()
    selector.zero_grad()
    predictor.zero_grad()

    train, test, validate, train_short, test_short, validate_short = loader(dataset='infer', min_ns=4, file_type='mulerx', file_path=file, label_type=2) # 2 means MATRES
    if len(test_short) == 0:
            short_dataloader = None
    else:
        short_dataloader = DataLoader(EventDataset(test_short), batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
    if len(test) == 0:
        dataloader = None
    else:
        dataloader = DataLoader(EventDataset(test), batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
    
    selector.eval()
    predictor.eval()
    pred = []
    if short_dataloader != None:
        for step, batch in tqdm.tqdm(enumerate(short_dataloader), desc="Processing for short doc", total=len(short_dataloader)):
            x, y, x_sent, y_sent, x_sent_id, y_sent_id, x_sent_pos, y_sent_pos, x_position, y_position, x_ev_embs, y_ev_embs, x_kg_ev_emb, \
            y_kg_ev_emb, doc_id, target, target_emb, target_len, ctx, ctx_emb, ctx_ev_embs, num_ev_sents, ctx_ev_kg_embs, ctx_len, ctx_pos, flag, xy = batch
                            
            augm_target, augm_target_mask, augm_pos_target, x_augm_position, y_augm_position = make_predictor_input(x_sent, y_sent, x_sent_pos, y_sent_pos, x_sent_id, y_sent_id, x_position, y_position, ctx, ctx_pos, 'all', doc_id, dropout_rate=0, is_test=True)
            xy = torch.tensor(xy)
            flag = torch.tensor(flag)
            if CUDA:
                augm_target = augm_target.cuda() 
                augm_target_mask = augm_target_mask.cuda()
                augm_pos_target = augm_pos_target.cuda()
                x_augm_position = x_augm_position.cuda() 
                y_augm_position = y_augm_position.cuda()
                xy = xy.cuda()
                flag = flag.cuda()
            logits, p_loss = predictor(augm_target, augm_target_mask, x_augm_position, y_augm_position, xy, flag, augm_pos_target)
            y_pred = torch.max(logits, 1).indices.cpu().numpy()
            for i in range(len(x)):
                pred.append(((x[i], y[i]), y_pred[i]))
    if dataloader !=  None:
        for step, batch in tqdm.tqdm(enumerate(dataloader), desc="Processing for long doc", total=len(dataloader)):
            x, y, x_sent, y_sent, x_sent_id, y_sent_id, x_sent_pos, y_sent_pos, x_position, y_position, x_ev_embs, y_ev_embs, x_kg_ev_emb, \
            y_kg_ev_emb, doc_id, target, target_emb, target_len, ctx, ctx_emb, ctx_ev_embs, num_ev_sents, ctx_ev_kg_embs, ctx_len, ctx_pos, flag, xy = batch
            target_emb = torch.stack(target_emb, dim=0)
            ctx_emb = torch.stack(pad_to_max_ns(ctx_emb), dim=0)
            if CUDA:
                target_emb = target_emb.cuda()
                ctx_emb = ctx_emb.cuda()

            ctx_selected, dist, log_prob = selector(target_emb, ctx_emb, target_len, ctx_len, params['num_ctx_select'])
            
            augm_target, augm_target_mask, augm_pos_target, x_augm_position, y_augm_position = make_predictor_input(x_sent, y_sent, x_sent_pos, y_sent_pos, x_sent_id, y_sent_id, x_position, y_position, ctx, ctx_pos, ctx_selected, doc_id, dropout_rate=0, is_test=True)
            xy = torch.tensor(xy)
            flag = torch.tensor(flag)
            if CUDA:
                augm_target = augm_target.cuda() 
                augm_target_mask = augm_target_mask.cuda()
                augm_pos_target = augm_pos_target.cuda()
                x_augm_position = x_augm_position.cuda() 
                y_augm_position = y_augm_position.cuda()
                xy = xy.cuda()
                flag = flag.cuda()
            logits, p_loss = predictor(augm_target, augm_target_mask, x_augm_position, y_augm_position, xy, flag, augm_pos_target)
            
            y_pred = torch.max(logits, 1).indices.cpu().numpy()
            for i in range(len(x)):
                pred.append(((x[i], y[i]), y_pred[i]))
    
    return pred


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='SEED', default=1741, type=int)
    parser.add_argument('--dataset', help="Name of dataset", action='append', required=True)
    parser.add_argument('--roberta_type', help="base or large", default='roberta-base', type=str)
    parser.add_argument('--bs', help='batch size', default=16, type=int)
    # parser.add_argument('--num_select', help='number of select sentence', default=3, type=int)

    args = parser.parse_args()
    seed = args.seed
    roberta_type  = args.roberta_type
    batch_size = args.bs
    datasets = args.dataset
    print(datasets)

    params = {
        's_hidden_dim': 512,
        # 512,
        's_mlp_dim': 512,
        # 512,
        'p_mlp_dim': 1024,
        # 512, 
        'n_head': 16,
        "epoches": 7,
        "warming_epoch": 1,
        "task_weights": {
            '1': 1, # 1 is HiEve
            '2': 1, # 2 is MATRES.
            # '3': trial.suggest_float('I2B2_weight', 0.4, 1, step=0.2),
        },
        'num_ctx_select': 3,
        's_lr': 1e-4,
        'b_lr': 7e-6,
        'm_lr': 5e-5,
        'b_lr_decay_rate': 0.5,
        'word_drop_rate': 0.05,
        'task_reward': 'logit',
        'perfomance_reward_weight': 0.7,
        'ctx_sim_reward_weight': 0.003,
        'knowledge_reward_weight': 0.7,
        'drop_rate': 0.5,
        'fn_activative': 'tanh',
        'is_mul': True,
        # trial.suggest_categorical('is_mul', [True, False])
        'is_sub': True
    }
    # file = 'datasets/mulerx/subevent-en-10/test/aviation_accidents-week2-nhung-108257_chunk_80.ann.tsvx'
    # predicts = inference(params, file)
    # print(predicts)
    for set in ['dev']:
        dir_name = f'datasets/mulerx/subevent-en-10/{set}/'
        predict_dir = 'datasets/temporal-mulerx/'
        onlyfiles = [f for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f)) and f.endswith('.tsvx')]
        i = 0
        for file in onlyfiles:
            # file = 'datasets/mulerx/subevent-en-10/test/aviation_accidents-week2-nhung-108257_chunk_80.ann.tsvx'
            predicts = inference(params, dir_name+file)
            i = i + 1
            print(predicts)
            # with open(predict_dir + f'temporal-{set}-{i}-' + file, 'w', encoding='UTF-8') as f:
            #     for pred in predicts:
            #         f.write(f"{pred[0][0]}\t{pred[0][1]}\t{temp_num_map[pred[1]]}\n")

        
