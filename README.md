# Selecting Optimal Context Sentences for Event-Event Relation Extraction
==========

This repo contains our PyTorch implementation for the paper [Selecting Optimal Context Sentences for Event-Event Relation Extraction](https://www.aaai.org/AAAI22Papers/AAAI-3912.ManH.pdf). 


## SCS-EERE

1. Create and activate a new environment:
```
docker build -t hieumdt/ie_env -f information-extraction-env.dockerfile .
```
2. Prepare data\
Download the desired corpus (HiEve, MATRES, TDD) and put it in folder ```.\datasets```\
Download [numberbatch w2v](https://github.com/commonsense/conceptnet-numberbatch) and move it to folder ```.\datasets```
```
wget https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-en-19.08.txt.gz
gunzip numberbatch-en-19.08.txt.gz
mv numberbatch-en-19.08.txt ./datasets
```
3. Train model:
```
python main.py --seed <your_seed> --dataset <datataset> --roberta_type <roberta_type> --best_path <path_to_save_model> --log_file <log> --bs <batch_size>
```
- dataset chooses from HiEve, MATRES, TDD_man, TDD_auto
- roberta_type chooses from roberta_base, roberta_large
4. Example commands:
Training HiEve
```
python main.py --seed 1741 --dataset HiEve --roberta_type roberta_large --best_path /rst_HiEve/ --log_file HiEve_result.txt --bs 16
```
Note \
Need to ensure the minimal number of sentences of the doc is always larger than the number of selected sentence.
## License

All work contained in this package is licensed under the Apache License, Version 2.0.

# Reference
Bibtex:
```
@article{trong2022selecting,
  title={Selecting Optimal Context Sentences for Event-Event Relation Extraction},
  author={Trong, Hieu Man Duc and Trung, Nghia Ngo and Van Ngo, Linh and Nguyen, Thien Huu},
  year={2022}
}
```
Email: hieuman2708@gmail.com; v.hieumdt@vinai.io

