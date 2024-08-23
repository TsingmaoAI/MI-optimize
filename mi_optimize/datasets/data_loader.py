import random
import logging

from .load_ceval import get_calibrate_ceval
from .load_cmmlu import get_calibrate_cmmlu
from .load_boss import get_calibrate_boss

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from datasets import load_dataset


def get_wikitext2(tokenizer, split='test', nsamples=128, seqlen=2048, seed=42, **kwargs):
    if split=='train':
        logging.info("get_wikitext2_train")
        traindata = load_dataset("mi_optimize/datasets/wikitext", 'wikitext-2-raw-v1', split='train')  
        trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt') 
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            trainloader.append(inp)
        return trainloader
    
    if split=='test':
        logging.info("get_wikitext2_test")
        testdata = load_dataset("mi_optimize/datasets/wikitext", 'wikitext-2-raw-v1', split='test')
        testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
        if nsamples=='all':
            nsamples = len(testenc['input_ids'][0]) // seqlen + 1
        testloader = []
        for i in range(nsamples):
            testloader.append(testenc['input_ids'][:, i * seqlen: (i + 1) * seqlen])
        return testloader
    
    raise ValueError(f'not support wikitext2 {split} split')
    
    
def get_c4(tokenizer, split='validation', nsamples=128, seqlen=2048, seed=42, **kwargs):
    if split=='train':
        logging.info("get_c4_train")
        traindata = load_dataset("mi_optimize/datasets/c4/", name='default', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
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
    
    if split=='validation':
        logging.info("get_c4_validation")
        valdata = load_dataset("mi_optimize/datasets/c4/", name='default', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
        testenc = tokenizer(" ".join(valdata[:1100]['text']), return_tensors='pt')  
        if nsamples=='all':  
            nsamples = len(testenc['input_ids'][0]) // seqlen + 1
        testloader = []
        for i in range(nsamples):
            testloader.append(testenc['input_ids'][:, i * seqlen: (i + 1) * seqlen])
        return testloader
    
    raise ValueError(f'not support c4 {split} split')


def get_ptb(tokenizer, split='test', nsamples=128, seqlen=2048, seed=42, **kwargs):
    if split=='train':
        logging.info("get_ptb_train")
        traindata = load_dataset("mi_optimize/datasets/ptb_text_only", 'penn_treebank', split='train')
        trainenc = tokenizer(" ".join(traindata["sentence"]), return_tensors="pt")
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append(inp)
        return trainloader
    
    if split=='test':
        logging.info("get_ptb_test")
        testdata  = load_dataset("mi_optimize/datasets/ptb_text_only", 'penn_treebank', split='test')
        testenc = tokenizer(" ".join(testdata ["sentence"]), return_tensors="pt")    
        testloader = [] 
        testenc = testenc.input_ids
        nsamples = testenc.numel() // seqlen
        for i in range(nsamples):
            testloader.append(testenc[:, (i * seqlen):((i + 1) * seqlen)])
        return testloader
    
    raise ValueError(f'not support ptb {split} split')

def get_test_loader(dataset_name, tokenizer, seqlen=2048, nsamples=128, seed=42, split='test'):
    logging.info(f"Dataset: {dataset_name}")
    logging.info(f"Sequence length: {seqlen}")
    logging.info(f"Number of samples: {nsamples}")
    
    if dataset_name == 'wikitext2':
        return get_wikitext2(tokenizer, nsamples=nsamples, seqlen=seqlen, seed=seed, split=split)
    if dataset_name == 'c4':
        return get_c4(tokenizer, nsamples=nsamples, seed=seed, seqlen=seqlen)
    if dataset_name == 'ptb':
        return get_ptb(tokenizer, nsamples=nsamples, seed=seed, seqlen=seqlen)
    
    raise ValueError(f"Unknown dataset: {dataset_name}")

def get_calibrate_loader(tokenizer, calibrate_config):
    calibrate_name = calibrate_config['name']
    
    if calibrate_name=='wikitext2':
        return get_wikitext2(tokenizer, **calibrate_config)
    
    if calibrate_name=='c4':
        return get_c4(tokenizer, **calibrate_config)
    
    if calibrate_name=='ptb':
        return get_ptb(tokenizer, **calibrate_config)
    
    if calibrate_name=='ceval':
        return get_calibrate_ceval(tokenizer, **calibrate_config)
    
    if calibrate_name=='cmmlu':
        return get_calibrate_cmmlu(tokenizer, **calibrate_config)
        
    if calibrate_name=='boss':
        return get_calibrate_boss(tokenizer, **calibrate_config)
    
    raise ValueError(f'not support calibrate name:{calibrate_name}')
