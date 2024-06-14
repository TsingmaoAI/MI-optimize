
import numpy as np
import torch
import random
import logging

from itertools import islice
import yaml

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from datasets import load_dataset

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def get_wikitext2(tokenizer, nsamples=128, seqlen=2048, seed=42, dataset_config=[]):
    print("get_wikitext2")
    traindata = load_dataset(dataset_config['wikitext_data_path'], 'wikitext-2-raw-v1', split='train')  
    testdata = load_dataset(dataset_config['wikitext_data_path'], 'wikitext-2-raw-v1', split='test')

    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt') 
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        trainloader.append(inp)

    nsamples = len(testenc['input_ids'][0]) // seqlen + 1
    testloader = []
    for i in range(nsamples):
        testloader.append(testenc['input_ids'][:, i * seqlen: (i + 1) * seqlen])
    
    return trainloader, testloader


def get_c4(tokenizer, nsamples=128, seqlen=2048, seed=42, dataset_config=[]):
    print("get_c4")
    traindata = load_dataset(path=dataset_config['c4_data_path'], name='default', data_files={'train': 'c4-train.00000-of-01024.json.gz'}, split='train')
    valdata = load_dataset(path=dataset_config['c4_data_path'], name='default', data_files={'validation': 'c4-validation.00000-of-00008.json.gz'}, split='validation')
    
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        trainloader.append(inp)
    testenc = tokenizer(" ".join(valdata[:1100]['text']), return_tensors='pt')    
    nsamples = len(testenc['input_ids'][0]) // seqlen + 1
    testloader = []
    for i in range(nsamples):
        testloader.append(testenc['input_ids'][:, i * seqlen: (i + 1) * seqlen])
    return trainloader, testloader

def get_ptb(tokenizer, nsamples=128, seqlen=2048, seed=42, dataset_config=[]):
    print("get_ptb")
    print('2', dataset_config['ptb_data_path'])
    traindata = load_dataset(dataset_config['ptb_data_path'], 'penn_treebank', split='train')
    testdata  = load_dataset(dataset_config['ptb_data_path'], 'penn_treebank', split='test')

    trainenc = tokenizer(" ".join(traindata["sentence"]), return_tensors="pt")
    testenc = tokenizer(" ".join(testdata ["sentence"]), return_tensors="pt")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append(inp)
    testloader = [] 
    testenc = testenc.input_ids
    nsamples = testenc.numel() // seqlen
    for i in range(nsamples):
        testloader.append(testenc[:, (i * seqlen):((i + 1) * seqlen)])
    return trainloader, testloader

def get_loader(dataset_name, tokenizer, seqlen=2048, nsamples=128, seed=42, dataset_path_config=[]):
    logging.info(f"Dataset: {dataset_name}")
    logging.info(f"Sequence length: {seqlen}")
    logging.info(f"Number of samples: {nsamples}")
    
    if dataset_name == 'wikitext2':
        return get_wikitext2(tokenizer, nsamples=nsamples, seqlen=seqlen, seed=seed, dataset_config=dataset_path_config)
    elif dataset_name == 'c4':
        return get_c4(tokenizer, nsamples=nsamples, seed=seed, seqlen=seqlen, dataset_config=dataset_path_config)
    elif dataset_name == 'ptb':
        return get_ptb(tokenizer, nsamples=nsamples, seed=seed, seqlen=seqlen, dataset_config=dataset_path_config)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def get_calibrate_dataset(calibrate_name, tokenizer, nsamples, seqlen):
    if calibrate_name in ['wikitext2', 'c4', 'ptb']:
        with open('configs/datasets_path.yaml', 'r') as file:
            dataset_path_config = yaml.safe_load(file)

        from mi_optimize.datasets import get_loader
        calibrate_data, _= get_loader(tokenizer=tokenizer, nsamples=nsamples, dataset_name=calibrate_name, seqlen=seqlen, dataset_path_config=dataset_path_config)

    elif calibrate_name in ['ceval', 'ceval_hm', 'ceval_st', 'ceval_ss']:
        with open('configs/ceval_calibration_config', 'r') as file:
            calibrate_config = yaml.safe_load(file)
            
        logging.info("ceval calibration Config: %s", calibrate_config)
        
        calibration_split = calibrate_config['calibration_split']
        calibrate_nums = calibrate_config['ceval'][calibrate_name]
        from datasets.load_ceval import get_calibrate_ceval
        calibrate_data = get_calibrate_ceval(calibrate_name, calibration_split, calibrate_nums, answer=True, path=calibrate_config['ceval_data_path'])
        
    elif calibrate_name in ['cmmlu', 'cmmlu_hm', 'cmmlu_st', 'cmmlu_ss']:
        from mi_optimize.datasets import get_calibrate_cmmlu
        with open('configs/cmmlu_calibration_config', 'r') as file:
            calibrate_config = yaml.safe_load(file)
            
        logging.info("ceval calibration Config: %s", calibrate_config)
        calibration_split = calibrate_config['calibration_split']
        calibrate_nums = calibrate_config['cmmlu'][calibrate_name]
        calibrate_data = get_calibrate_cmmlu(calibrate_name, calibration_split, calibrate_nums, answer=True, path=calibrate_config['ceval_data_path'])
    elif calibrate_name in ['NaturalLanguageInference_mnli']: 
        from datasets.load_ceval import get_calibrate_boss
        calibrate_dataset = calibrate_name.split("_")
        calibrate_task_name = calibrate_dataset[0]
        calibrate_dataset_name = calibrate_dataset[1]

        calibrate_data = get_calibrate_boss(calibrate_task_name, calibrate_dataset_name, nsamples=208, split='train', shuffle=False, seed=42)
    else:
        raise ValueError(f'not support {calibrate_name}')

    return calibrate_data