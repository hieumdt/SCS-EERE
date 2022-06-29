import numpy as np
np.random.seed(1741)
import torch
torch.manual_seed(1741)
import random
random.seed(1741)


CUDA = torch.cuda.is_available()

corpus_mapping = {'HiEve': '1', 'MATRES': '2', 'I2B2': '3', 'TBD':'4'}

pos_tags = ["ADJ", "ADP", "ADV", "AUX", "CONJ", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "SPACE"]
pos_dict = {"None": 0, "ADJ": 1, "ADP": 2, "ADV": 3, "AUX": 4, "CONJ": 5, "CCONJ": 6, "DET": 7, 
            "INTJ": 8, "NOUN": 9, "NUM": 10, "PART": 11, "PRON": 12, "PROPN": 13, "PUNCT": 14, 
            "SCONJ": 15, "SYM": 16, "VERB": 17, "X": 18, "SPACE": 19, "UNK":20}

hieve_label_dict={"SuperSub": 0, "SubSuper": 1, "Coref": 2, "NoRel": 3}
hieve_num_dict = {0: "SuperSub", 1: "SubSuper", 2: "Coref", 3: "NoRel"}

mypath_TB = './datasets/MATRES/TBAQ-cleaned/TimeBank/' # after correction
mypath_AQ = './datasets/MATRES/TBAQ-cleaned/AQUAINT/' 
mypath_PL = './datasets/MATRES/te3-platinum/'

MATRES_timebank = './datasets/MATRES/timebank.txt'
MATRES_aquaint = './datasets/MATRES/aquaint.txt'
MATRES_platinum = './datasets/MATRES/platinum.txt'

TDD_man_train = "./datasets/TDDiscourse/TDDMan/TDDManTrain.tsv"
TDD_man_test = "./datasets/TDDiscourse/TDDMan/TDDManTest.tsv"
TDD_man_val = "./datasets/TDDiscourse/TDDMan/TDDManDev.tsv"

TDD_auto_train = "./datasets/TDDiscourse/TDDAuto/TDDAutoTrain.tsv"
TDD_auto_test = "./datasets/TDDiscourse/TDDAuto/TDDAutoTest.tsv"
TDD_auto_val = "./datasets/TDDiscourse/TDDAuto/TDDAutoDev.tsv"

temp_label_map = {"BEFORE": 0, "AFTER": 1, "EQUAL": 2, "VAGUE": 3}
temp_num_map = {0: "BEFORE", 1: "AFTER", 2: "EQUAL", 3: "VAGUE"}

i2b2_label_dict = {'BEFORE': 0, 'AFTER': 1, 'SIMULTANEOUS': 2, 'OVERLAP': 2, 'simultaneous': 2, 
                    'BEGUN_BY': 1, 'ENDED_BY': 0, 'DURING': 2,'BEFORE_OVERLAP': 0}
tbd_label_dict = { 'BEFORE': 0, 'AFTER': 1, 'INCLUDES': 2, 'IS_INCLUDED': 3, 'SIMULTANEOUS': 4, 'NONE': 5}

tdd_label_dict = {'b': 0, 'a': 1, 'i': 2, 'ii': 3, 's': 4}

mulerx_label_dict={"SuperSub": 0, "SubSuper": 1, "NoRel": 2}
mulerx_num_dict = {0: "SuperSub", 1: "SubSuper", 2: "NoRel"}


