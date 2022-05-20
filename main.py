import datetime
import os
from exp import EXP
from utils.constant import CUDA
from models.predictor_model import ECIRobertaJointTask
from models.selector_model import LSTMSelector
import torch
torch.manual_seed(1741)
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import random
random.seed(1741)
import numpy as np
np.random.seed(1741)
import optuna
from torch.utils.data.dataloader import DataLoader
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from data_loader.loader import loader
from data_loader.EventDataset import EventDataset
import gc


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def collate_fn(batch):
    return tuple(zip(*batch))

def objective(trial: optuna.Trial):
    params = {
        's_hidden_dim': 512, 
        's_mlp_dim': 512, 
        'p_mlp_dim': 1024, 
        'epoches': 7, 
        'warming_epoch': 1, 
        'num_ctx_select': 3, 
        's_lr': 0.0001, 
        'b_lr': 7e-06, 
        'm_lr': 5e-05, 
        'b_lr_decay_rate': 0.5, 
        'word_drop_rate': 0.05, 
        'task_reward': 'logit', 
        'perfomance_reward_weight': 0.7, 
        'ctx_sim_reward_weight': 0.003, 
        'knowledge_reward_weight': 0.7, 
        'fn_activate': 'tanh', 
        'seed': 1741
        }
    torch.manual_seed(1741)
    np.random.seed(params['seed'])
    random.seed(params['seed'])

    drop_rate = 0.5
    fn_activative = params['fn_activate']
    is_mul = True
    # trial.suggest_categorical('is_mul', [True, False])
    is_sub = True
    # trial.suggest_categorical('is_sub', [True, False])
    num_select = params['num_ctx_select']

    train_set = []
    train_short_set = []
    validate_dataloaders = {}
    test_dataloaders = {}
    validate_short_dataloaders = {}
    test_short_dataloaders = {}
    for dataset in datasets:
        train, test, validate, train_short, test_short, validate_short = loader(dataset, 5)
        train_set.extend(train)
        train_short_set.extend(train_short)
        validate_dataloader = DataLoader(EventDataset(validate), batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
        test_dataloader = DataLoader(EventDataset(test), batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
        validate_dataloaders[dataset] = validate_dataloader
        test_dataloaders[dataset] = test_dataloader
        if len(validate_short) == 0:
            validate_short_dataloader = None
        else:
            validate_short_dataloader = DataLoader(EventDataset(validate_short), batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
        if len(test_short) == 0:
            test_short_dataloader = None
        else:
            test_short_dataloader = DataLoader(EventDataset(test_short), batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
        validate_short_dataloaders[dataset] = validate_short_dataloader
        test_short_dataloaders[dataset] = test_short_dataloader
    if len(train_short_set) == 0:
        train_short_dataloader = None
    else:
        train_short_dataloader = DataLoader(EventDataset(train_short_set), batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
    train_dataloader = DataLoader(EventDataset(train_set), batch_size=batch_size, shuffle=True,collate_fn=collate_fn)

    print("Hyperparameter will be use in this trial: \n {}".format(params))

    selector = LSTMSelector(768, params['s_hidden_dim'], params['s_mlp_dim'])
    predictor =ECIRobertaJointTask(mlp_size=params['p_mlp_dim'], roberta_type=roberta_type, datasets=datasets, pos_dim=16, 
                                    fn_activate=fn_activative, drop_rate=drop_rate, task_weights=None)
    
    if CUDA:
        selector = selector.cuda()
        predictor = predictor.cuda()
    selector.zero_grad()
    predictor.zero_grad()
    print("# of parameters:", count_parameters(selector) + count_parameters(predictor))
    epoches = params['epoches'] + 5
    total_steps = len(train_dataloader) * epoches
    print("Total steps: [number of batches] x [number of epochs] =", total_steps)

    exp = EXP(selector, predictor, epoches, params['num_ctx_select'], train_dataloader, validate_dataloaders, test_dataloaders,
            train_short_dataloader, test_short_dataloaders, validate_short_dataloaders, 
            params['s_lr'], params['b_lr'], params['m_lr'], params['b_lr_decay_rate'],  params['epoches'], params['warming_epoch'],
            best_path, word_drop_rate=params['word_drop_rate'], reward=[params['task_reward']], perfomance_reward_weight=params['perfomance_reward_weight'],
            ctx_sim_reward_weight=params['ctx_sim_reward_weight'], kg_reward_weight=params['knowledge_reward_weight'])
    F1, CM, matres_F1, test_f1 = exp.train()
    # test_f1 = exp.evaluate(is_test=True)
    print("Result: Best micro F1 of interaction: {}".format(F1))
    with open(result_file, 'a', encoding='UTF-8') as f:
        f.write("\n -------------------------------------------- \n")
        # f.write("\nNote: use lstm in predictor \n")
        f.write("{}\n".format(roberta_type))
        f.write("Hypeparameter: \n{}\n ".format(params))
        f.write("Test F1: {}\n".format(test_f1))
        f.write("Seed: {}\n".format(seed))
        # f.write("Drop rate: {}\n".format(drop_rate))
        # f.write("Batch size: {}\n".format(batch_size))
        f.write("Activate function: {}\n".format(fn_activative))
        f.write("Sub: {} - Mul: {}".format(is_sub, is_mul))
        # f.write("\n Best F1 MATRES: {} \n".format(matres_F1))
        for i in range(0, len(datasets)):
            f.write("{} \n".format(dataset[i]))
            f.write("F1: {} \n".format(F1[i]))
            f.write("CM: \n {} \n".format(CM[i]))
        f.write("Time: {} \n".format(datetime.datetime.now()))
    if test_f1 > 0.834:
        os.rename(best_path[1], best_path[1]+'.{}'.format(test_f1))
        os.rename(best_path[0], best_path[0]+'.{}'.format(test_f1))

    del exp
    del selector
    del predictor
    gc.collect()

    return test_f1


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='SEED', default=1741, type=int)
    parser.add_argument('--dataset', help="Name of dataset", action='append', required=True)
    parser.add_argument('--roberta_type', help="base or large", default='roberta-base', type=str)
    parser.add_argument('--best_path', help="Path for save model", type=str)
    parser.add_argument('--log_file', help="Path of log file", type=str)
    parser.add_argument('--bs', help='batch size', default=16, type=int)
    # parser.add_argument('--num_select', help='number of select sentence', default=3, type=int)

    args = parser.parse_args()
    seed = args.seed
    datasets = args.dataset
    print(datasets)
    roberta_type  = args.roberta_type
    best_path = args.best_path
    best_path = [best_path+"selector.pth", best_path+"predictor.pth"]
    result_file = args.log_file
    batch_size = args.bs
    pre_processed_dir = "./" + "_".join(datasets) + "/"

    sampler = optuna.samplers.TPESampler(seed=1741)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=50)
    trial = study.best_trial

    print('Accuracy: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))

