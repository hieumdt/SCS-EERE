import torch
torch.manual_seed(1741)
import random
random.seed(1741)
import numpy as np
np.random.seed(1741)
from collections import defaultdict
import pickle
import os
import tqdm
from itertools import combinations
from data_loader.reader import mulerx_tsvx_reader, tbd_tml_reader, tdd_tml_reader, tml_reader, tsvx_reader
from utils.tools import create_target, padding, pos_to_id
from sklearn.model_selection import train_test_split
from utils.SentenceEncoder import SentenceEncoder
import gc


class Reader(object):
    def __init__(self, type) -> None:
        super().__init__()
        self.type = type
    
    def read(self, dir_name, file_name):
        if self.type == 'tsvx':
            return tsvx_reader(dir_name, file_name)
        elif self.type == 'tml':
            return tml_reader(dir_name, file_name)
        elif self.type == 'mulerx':
            return mulerx_tsvx_reader(dir_name, file_name)
        elif self.type == 'tbd_tml':
            return tbd_tml_reader(dir_name, file_name)
        elif self.type == 'tdd_man':
            return tdd_tml_reader(dir_name, file_name, type_doc='man')
        elif self.type == 'tdd_auto':
            return tdd_tml_reader(dir_name, file_name, type_doc='auto')
        else:
            raise ValueError("We have not supported {} type yet!".format(self.type))


class C2V(object):
    def __init__(self, emb_file:str) -> None:
        super().__init__()
        with open(emb_file, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
        self.c2v = {}
        for line in lines:
            tokens = line.split(" ")
            concept = tokens[0]
            emb = [float(tok) for tok in tokens[1:]]
            self.c2v[concept] = emb

    def get_emb(self, concept):
        _concept = "_".join(concept.split(" ")).lower()
        # print(cp)
        try:
            return self.c2v[_concept]
        except:
            return [0.0]*300


def load_dataset(dir_name, type):
    reader = Reader(type)
    onlyfiles = [f for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))]
    corpus = []
    # i = 0
    for file_name in tqdm.tqdm(onlyfiles):
        # if i == 1:
        #     break
        # i = i + 1
        if type == 'i2b2_xml':
            if file_name.endswith('.xml'):
                my_dict = reader.read(dir_name, file_name)
                if my_dict != None:
                    corpus.append(my_dict)
        else:
            my_dict = reader.read(dir_name, file_name)
            if my_dict != None:
                corpus.append(my_dict)
    return corpus

global_sent_encoder = SentenceEncoder('roberta-base')
global_c2v = C2V('./datasets/numberbatch-en-19.08.txt')

def loader(dataset, min_ns, file_type=None, file_path=None, label_type=None):
    sent_encoder = global_sent_encoder
    c2v = global_c2v
    def get_data_point(my_dict, flag):
        data = []
        eids = my_dict['event_dict'].keys()
        pair_events = list(combinations(eids, 2))

        ctx_id_augm = []
        for pair in pair_events:
            x, y = pair
            
            x_sent_id = my_dict['event_dict'][x]['sent_id']
            y_sent_id = my_dict['event_dict'][y]['sent_id']
            ctx_id = list(range(len( my_dict["sentences"])))
            if  x_sent_id != y_sent_id:
                ctx_id.remove(x_sent_id)
                ctx_id.remove(y_sent_id)
            else:
                ctx_id.remove(x_sent_id)
            id_augm = [tuple(sorted([x_sent_id, y_sent_id, id])) for id in ctx_id]
            ctx_id_augm.extend(id_augm)
        ctx_id_augm = list(set(ctx_id_augm))
        # print(len(ctx_id_augm))

        ctx_augm = []
        ctx_augm_mask = []
        _augm_emb = []
        for ids in ctx_id_augm:
            ids = set(ids)
            sent = []
            for id in ids:
                sent = sent + my_dict["sentences"][id]["roberta_subword_to_ID"][1:]
            sent = [0] + sent
            pad, mask = padding(sent, max_sent_len=512)
            _augm_emb.append(sent_encoder.encode(pad, mask, is_ctx=True))
            ctx_augm.append(pad)
            ctx_augm_mask.append(mask)
        _augm_emb = torch.cat(_augm_emb, dim=0)
        if len(ctx_augm) == 0:
            _augm_emb = []
        _ctx_augm_emb = {}
        for i in range(len(ctx_id_augm)):
            _ctx_augm_emb[ctx_id_augm[i]] = _augm_emb[i]

        doc = []
        doc_mask = []
        doc_emb = []
        for sent_id in list(range(len( my_dict["sentences"]))):
            sent = my_dict["sentences"][sent_id]["roberta_subword_to_ID"]
            pad, mask = padding(sent, max_sent_len=512)
            doc_emb.append(sent_encoder.encode(pad, mask))
            doc.append(pad)
            doc_mask.append(mask)
        doc_emb = torch.cat(doc_emb, dim=0)

        sent_ev = defaultdict(list)
        sent_ev_ids = defaultdict(list)
        ev_kg_emb = {}
        # print(eids)
        for eid in eids:
            sent_id = my_dict['event_dict'][eid]['sent_id']
            e_possition = my_dict["event_dict"][eid]["roberta_subword_id"]
            sent_ev[sent_id].append(e_possition)
            sent_ev_ids[sent_id].append(eid)
            mention = my_dict["event_dict"][eid]["mention"] ######
            # print(mention)
            kg_emb = c2v.get_emb(mention)
            # print(kg_emb)
            ev_kg_emb[eid] = kg_emb
        sent_ev = dict(sent_ev)
        sent_ev_ids = dict(sent_ev_ids)
        # print(ev_kg_emb)
        
        for pair in pair_events:
            x, y = pair
            
            x_sent_id = my_dict['event_dict'][x]['sent_id']
            y_sent_id = my_dict['event_dict'][y]['sent_id']

            x_sent = my_dict["sentences"][x_sent_id]["roberta_subword_to_ID"]
            y_sent = my_dict["sentences"][y_sent_id]["roberta_subword_to_ID"]

            x_position = my_dict["event_dict"][x]["roberta_subword_id"]
            y_position = my_dict["event_dict"][y]["roberta_subword_id"]

            x_sent_pos = pos_to_id(my_dict["sentences"][x_sent_id]["roberta_subword_pos"])
            y_sent_pos = pos_to_id(my_dict["sentences"][y_sent_id]["roberta_subword_pos"])
            
            target, x_position_new, y_position_new = create_target(x_sent, y_sent, x_sent_id, y_sent_id, x_position, y_position)
            target_encode = sent_encoder.encode(target)
            target_emb = target_encode[:, 0].squeeze()
            target_len = len(target)

            x_ev_embs = target_encode[:, x_position_new].squeeze()
            y_ev_embs = target_encode[:, y_position_new].squeeze()

            x_kg_ev_emb = torch.tensor(ev_kg_emb[x])
            y_kg_ev_emb = torch.tensor(ev_kg_emb[y])
            # print(x_kg_ev_emb)

            ctx = []
            _ctx_emb = []
            ctx_pos = []
            ctx_len = []
            _ctx_ev_embs = []
            _ctx_ev_kg_embs = []
            ctx_id = list(range(len( my_dict["sentences"])))
            num_ev_sents = []
            if  x_sent_id != y_sent_id:
                ctx_id.remove(x_sent_id)
                ctx_id.remove(y_sent_id)
            else:
                ctx_id.remove(x_sent_id)
            id_mapping = {}
            if len(ctx_id) != 0:
                i = 0
                for sent_id in ctx_id:
                    id_mapping[i] = sent_id
                    sent = my_dict["sentences"][sent_id]['roberta_subword_to_ID']
                    sent_pos = pos_to_id(my_dict["sentences"][sent_id]['roberta_subword_pos'])
                    ctx.append(sent)
                    ctx_pos.append(sent_pos)
                    ctx_len.append(len(sent))
                    sent_emb = _ctx_augm_emb[tuple(sorted([x_sent_id, y_sent_id, sent_id]))]
                    assert sent_emb != None
                    _ctx_emb.append(sent_emb)
                    e_possitions = sent_ev.get(sent_id)
                    if e_possitions == None:
                        e_possitions = []
                    num_ev_sents.append(len(e_possitions))
                    if len(e_possitions) != 0:
                        ev_embs = torch.max(doc_emb[sent_id, e_possitions, :], dim=0)[0] # 768
                        _ctx_ev_embs.append(ev_embs)

                    eids = sent_ev_ids.get(sent_id)
                    if eids == None:
                        eids = []
                    # print(eids)
                    if len(eids) != 0:
                        sent_ev_kg_emb = [ev_kg_emb[eid] for eid in eids]
                        # print("before: ", torch.tensor(sent_ev_kg_emb))
                        sent_ev_kg_emb = torch.max(torch.tensor(sent_ev_kg_emb), dim=0)[0] # 300
                        # print("max: ", sent_ev_kg_emb)
                        _ctx_ev_kg_embs.append(sent_ev_kg_emb)
                    else:
                        # print("Sent no ev")
                        _ctx_ev_embs.append(torch.ones(768)*-1000.0)
                        _ctx_ev_kg_embs.append(torch.ones(300)*-1000.0)
                    i = i + 1
                # print(ctx_ev_kg_embs)
                ctx_ev_kg_embs =torch.stack(_ctx_ev_kg_embs, dim=0)
                ctx_ev_embs = torch.stack(_ctx_ev_embs, dim=0)
                ctx_emb = torch.stack(_ctx_emb, dim=0) # ns x 768
            # print(ctx_emb.size())
            xy = my_dict["relation_dict"].get((x, y))
            yx = my_dict["relation_dict"].get((y, x))

            candidates = [
                [str(x), str(y), x_sent, y_sent, x_sent_id, y_sent_id, x_sent_pos, y_sent_pos, x_position, y_position, x_ev_embs, y_ev_embs, x_kg_ev_emb, 
                y_kg_ev_emb, id_mapping, target, target_emb, target_len, ctx, ctx_emb, ctx_ev_embs, num_ev_sents, ctx_ev_kg_embs, ctx_len, ctx_pos, flag, xy],
                [str(y), str(x), y_sent, x_sent, y_sent_id, x_sent_id, y_sent_pos, x_sent_pos, y_position, x_position, y_ev_embs, x_ev_embs, y_kg_ev_emb,
                x_kg_ev_emb, id_mapping, target, target_emb, target_len, ctx, ctx_emb, ctx_ev_embs, num_ev_sents, ctx_ev_kg_embs, ctx_len, ctx_pos, flag, yx],
            ]
            for item in candidates:
                if item[-1] != None:
                    data.append(item)
        return data

    train_set = []
    train_short = []
    test_set = []
    test_short = []
    validate_set = []
    validate_short = []
    if dataset == "MATRES":
        print("MATRES Loading .......")
        aquaint_dir_name = "./datasets/MATRES/TBAQ-cleaned/AQUAINT/"
        timebank_dir_name = "./datasets/MATRES/TBAQ-cleaned/TimeBank/"
        platinum_dir_name = "./datasets/MATRES/te3-platinum/"
        validate = load_dataset(aquaint_dir_name, 'tml')
        train = load_dataset(timebank_dir_name, 'tml')
        test = load_dataset(platinum_dir_name, 'tml')
        _tt = train + validate
        _tt = list(sorted(_tt, key=lambda x: x["doc_id"]))
        train, validate = train_test_split(_tt, test_size=0.1, train_size=0.9)
        
        processed_dir = "./datasets/MATRES/docEvR_processed_kg/"
        if not os.path.exists(processed_dir):
            os.mkdir(processed_dir)

        for my_dict in tqdm.tqdm(train):
            file_name = my_dict["doc_id"] + ".pkl"
            if not os.path.exists(processed_dir+file_name):
                data = get_data_point(my_dict, 2)
                with open(processed_dir+file_name, 'wb') as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            else:
                with open(processed_dir+file_name, 'rb') as f:
                    data = pickle.load(f)
            for item in data:
                if len(item[-4]) >= min_ns:
                    train_set.append(item)
                if len(item[-4]) < min_ns:
                    train_short.append(item)

        for my_dict in tqdm.tqdm(validate):
            file_name = my_dict["doc_id"] + ".pkl"
            if not os.path.exists(processed_dir+file_name):
                data = get_data_point(my_dict, 2)
                with open(processed_dir+file_name, 'wb') as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            else:
                with open(processed_dir+file_name, 'rb') as f:
                    data = pickle.load(f)
            for item in data:
                if len(item[-4]) >= min_ns:
                    validate_set.append(item)
                if len(item[-4]) < min_ns:
                    validate_short.append(item)
        
        for my_dict in tqdm.tqdm(test):
            file_name = my_dict["doc_id"] + ".pkl"
            if not os.path.exists(processed_dir+file_name):
                data = get_data_point(my_dict, 2)
                with open(processed_dir+file_name, 'wb') as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            else:
                with open(processed_dir+file_name, 'rb') as f:
                    data = pickle.load(f)
            for item in data:
                if len(item[-4]) >= min_ns:
                    test_set.append(item)
                if len(item[-4]) < min_ns:
                    test_short.append(item)

        print("Train_size: {}".format(len(train_set) + len(train_short)))
        print("Test_size: {}".format(len(test_set) + len(test_short)))
        print("Validate_size: {}".format(len(validate_set) + len(validate_short)))
    
    if dataset == "HiEve":
        print("HiEve Loading .....")
        dir_name = "./datasets/hievents_v2/processed/"
        corpus = load_dataset(dir_name, 'tsvx')
        corpus = list(sorted(corpus, key=lambda x: x["doc_id"]))
        train, test = train_test_split(corpus, train_size=0.8, test_size=0.2)
        train, validate = train_test_split(train, train_size=0.75, test_size=0.25)
        sample = 0.4

        processed_dir = "./datasets/hievents_v2/processed/docEvR_processed_kg/"
        if not os.path.exists(processed_dir):
            os.mkdir(processed_dir)

        for my_dict in tqdm.tqdm(train):
            file_name = my_dict["doc_id"] + ".pkl"
            if os.path.exists(processed_dir+file_name):
                with open(processed_dir+file_name, 'rb') as f:
                    data = pickle.load(f)
            else:
                data = get_data_point(my_dict, 1)
                with open(processed_dir+file_name, 'wb') as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            for item in data:
                if item[-1] == 3:
                    if random.uniform(0, 1) < sample:
                        if len(item[-4]) >= min_ns:
                            train_set.append(item)
                        else:
                            train_short.append(item)
                else:
                    if len(item[-4]) >= min_ns:
                            train_set.append(item)
                    else:
                        train_short.append(item)
        
        for my_dict in tqdm.tqdm(test):
            file_name = my_dict["doc_id"] + ".pkl"
            if os.path.exists(processed_dir+file_name):
                with open(processed_dir+file_name, 'rb') as f:
                    data = pickle.load(f)
            else:
                data = get_data_point(my_dict, 1)
                with open(processed_dir+file_name, 'wb') as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            for item in data:
                if item[-1] == 3:
                    if random.uniform(0, 1) < sample:
                        if len(item[-4]) >= min_ns:
                            test_set.append(item)
                        else:
                            test_short.append(item)
                else:
                    if len(item[-4]) >= min_ns:
                            test_set.append(item)
                    else:
                        test_short.append(item)
        
        for my_dict in tqdm.tqdm(validate):
            file_name = my_dict["doc_id"] + ".pkl"
            if os.path.exists(processed_dir+file_name):
                with open(processed_dir+file_name, 'rb') as f:
                    data = pickle.load(f)
            else:
                data = get_data_point(my_dict, 1)
                with open(processed_dir+file_name, 'wb') as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            for item in data:
                if item[-1] == 3:
                    if random.uniform(0, 1) < sample:
                        if len(item[-4]) >= min_ns:
                            validate_set.append(item)
                        else:
                            validate_short.append(item)
                else:
                    if len(item[-4]) >= min_ns:
                            validate_set.append(item)
                    else:
                        validate_short.append(item)
        
        print("Train_size: {}".format(len(train_set) + len(train_short)))
        print("Test_size: {}".format(len(test_set) + len(test_short)))
        print("Validate_size: {}".format(len(validate_set) + len(validate_short)))

    if dataset == 'TBD':
        print("Timebank Dense Loading .....")
        train_dir = "./datasets/TimeBank-dense/train/"
        test_dir = "./datasets/TimeBank-dense/test/"
        validate_dir = "./datasets/TimeBank-dense/dev/"
        train = load_dataset(train_dir, 'tbd_tml')
        test = load_dataset(test_dir, 'tbd_tml')
        validate = load_dataset(validate_dir, 'tbd_tml')
        _tt = train + validate
        _tt = list(sorted(_tt, key=lambda x: x["doc_id"]))
        train, validate = train_test_split(_tt, test_size=0.1, train_size=0.9)

        processed_dir = "./datasets/TimeBank-dense/docEvR_processed_kg/"
        if not os.path.exists(processed_dir):
            os.mkdir(processed_dir)

        for my_dict in tqdm.tqdm(train):
            file_name = my_dict["doc_id"] + ".pkl"
            if not os.path.exists(processed_dir+file_name):
                data = get_data_point(my_dict, 4)
                with open(processed_dir+file_name, 'wb') as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            else:
                with open(processed_dir+file_name, 'rb') as f:
                    data = pickle.load(f)
            for item in data:
                if len(item[-4]) >= min_ns:
                    train_set.append(item)
                if len(item[-4]) < min_ns:
                    train_short.append(item)

        for my_dict in tqdm.tqdm(test):
            file_name = my_dict["doc_id"] + ".pkl"
            if not os.path.exists(processed_dir+file_name):
                data = get_data_point(my_dict, 4)
                with open(processed_dir+file_name, 'wb') as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            else:
                with open(processed_dir+file_name, 'rb') as f:
                    data = pickle.load(f)
            for item in data:
                if len(item[-4]) >= min_ns:
                    test_set.append(item)
                if len(item[-4]) < min_ns:
                    test_short.append(item)
            
        for my_dict in tqdm.tqdm(validate):
            file_name = my_dict["doc_id"] + ".pkl"
            if not os.path.exists(processed_dir+file_name):
                data = get_data_point(my_dict, 4)
                with open(processed_dir+file_name, 'wb') as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            else:
                with open(processed_dir+file_name, 'rb') as f:
                    data = pickle.load(f)
            for item in data:
                if len(item[-4]) >= min_ns:
                    validate_set.append(item)
                if len(item[-4]) < min_ns:
                    validate_short.append(item)

        print("Train_size: {}".format(len(train_set) + len(train_short)))
        print("Test_size: {}".format(len(test_set) + len(test_short)))
        print("Validate_size: {}".format(len(validate_set) + len(validate_short)))
        
    if dataset == 'TDD_man':
        print("TDD_man Loading .....")
        train_dir = "./datasets/TimeBank-dense/train/"
        test_dir = "./datasets/TimeBank-dense/test/"
        validate_dir = "./datasets/TimeBank-dense/dev/"
        train = load_dataset(train_dir, 'tdd_man')
        test = load_dataset(test_dir, 'tdd_man')
        validate = load_dataset(validate_dir, 'tdd_man')
        # _tt = train + validate
        # _tt = list(sorted(_tt, key=lambda x: x["doc_id"]))
        # train, validate = train_test_split(_tt, test_size=0.1, train_size=0.9)

        processed_dir = "./datasets/TDDiscourse/TDDMan/docEvR_processed_kg/"
        if not os.path.exists(processed_dir):
            os.mkdir(processed_dir)

        for my_dict in tqdm.tqdm(train):
            file_name = my_dict["doc_id"] + ".pkl"
            if not os.path.exists(processed_dir+file_name):
                data = get_data_point(my_dict, 5)
                with open(processed_dir+file_name, 'wb') as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            else:
                with open(processed_dir+file_name, 'rb') as f:
                    data = pickle.load(f)
            for item in data:
                if len(item[-4]) >= min_ns:
                    train_set.append(item)
                if len(item[-4]) < min_ns:
                    train_short.append(item)

        for my_dict in tqdm.tqdm(test):
            file_name = my_dict["doc_id"] + ".pkl"
            if not os.path.exists(processed_dir+file_name):
                data = get_data_point(my_dict, 5)
                with open(processed_dir+file_name, 'wb') as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            else:
                with open(processed_dir+file_name, 'rb') as f:
                    data = pickle.load(f)
            for item in data:
                if len(item[-4]) >= min_ns:
                    test_set.append(item)
                if len(item[-4]) < min_ns:
                    test_short.append(item)
            
        for my_dict in tqdm.tqdm(validate):
            file_name = my_dict["doc_id"] + ".pkl"
            if not os.path.exists(processed_dir+file_name):
                data = get_data_point(my_dict, 5)
                with open(processed_dir+file_name, 'wb') as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            else:
                with open(processed_dir+file_name, 'rb') as f:
                    data = pickle.load(f)
            for item in data:
                if len(item[-4]) >= min_ns:
                    validate_set.append(item)
                if len(item[-4]) < min_ns:
                    validate_short.append(item)

        print("Train_size: {}".format(len(train_set) + len(train_short)))
        print("Test_size: {}".format(len(test_set) + len(test_short)))
        print("Validate_size: {}".format(len(validate_set) + len(validate_short)))
    
    if dataset == 'TDD_auto':
        print("TDD_auto Loading .....")
        train_dir = "./datasets/TimeBank-dense/train/"
        test_dir = "./datasets/TimeBank-dense/test/"
        validate_dir = "./datasets/TimeBank-dense/dev/"
        train = load_dataset(train_dir, 'tdd_auto')
        test = load_dataset(test_dir, 'tdd_auto')
        validate = load_dataset(validate_dir, 'tdd_auto')
        # _tt = train + validate
        # _tt = list(sorted(_tt, key=lambda x: x["doc_id"]))
        # train, validate = train_test_split(_tt, test_size=0.1, train_size=0.9)

        processed_dir = "./datasets/TDDiscourse/TDDAuto/docEvR_processed_kg/"
        if not os.path.exists(processed_dir):
            os.mkdir(processed_dir)

        for my_dict in tqdm.tqdm(train):
            file_name = my_dict["doc_id"] + ".pkl"
            if not os.path.exists(processed_dir+file_name):
                data = get_data_point(my_dict, 5)
                with open(processed_dir+file_name, 'wb') as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            else:
                with open(processed_dir+file_name, 'rb') as f:
                    data = pickle.load(f)
            for item in data:
                if len(item[-4]) >= min_ns:
                    train_set.append(item)
                if len(item[-4]) < min_ns:
                    train_short.append(item)

        for my_dict in tqdm.tqdm(test):
            file_name = my_dict["doc_id"] + ".pkl"
            if not os.path.exists(processed_dir+file_name):
                data = get_data_point(my_dict, 5)
                with open(processed_dir+file_name, 'wb') as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            else:
                with open(processed_dir+file_name, 'rb') as f:
                    data = pickle.load(f)
            for item in data:
                if len(item[-4]) >= min_ns:
                    test_set.append(item)
                if len(item[-4]) < min_ns:
                    test_short.append(item)
            
        for my_dict in tqdm.tqdm(validate):
            file_name = my_dict["doc_id"] + ".pkl"
            if not os.path.exists(processed_dir+file_name):
                data = get_data_point(my_dict, 5)
                with open(processed_dir+file_name, 'wb') as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            else:
                with open(processed_dir+file_name, 'rb') as f:
                    data = pickle.load(f)
            for item in data:
                if len(item[-4]) >= min_ns:
                    validate_set.append(item)
                if len(item[-4]) < min_ns:
                    validate_short.append(item)

        print("Train_size: {}".format(len(train_set) + len(train_short)))
        print("Test_size: {}".format(len(test_set) + len(test_short)))
        print("Validate_size: {}".format(len(validate_set) + len(validate_short)))

    if dataset=='infer':
        reader = Reader(file_type)
        print(f'Reading file {file_path} ....')
        my_dict = reader.read('', file_path)
        data = get_data_point(my_dict, label_type)
        for item in data:
            if len(item[-4]) >= min_ns:
                test_set.append(item)
            else:
                test_short.append(item)
        print("Train_size: {}".format(len(train_set)))
        print("Test_size: {}".format(len(test_set)))
        print("Validate_size: {}".format(len(validate_set)))
        print("Train_size: {}".format(len(train_short)))
        print("Test_size: {}".format(len(test_short)))
        print("Validate_size: {}".format(len(validate_short)))

    del sent_encoder
    del c2v
    gc.collect()

    return train_set, test_set, validate_set, train_short, test_short, validate_short
