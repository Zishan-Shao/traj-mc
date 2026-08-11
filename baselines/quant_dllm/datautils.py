import random

import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
import os

from paths import DATA_DIR, PROJECT_DIR


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

'''
Generate tokenizer and return it to preload datasets by converting them to embedded vectors instead of natural words
'''
def get_tokenizer(model):
    if "dream" in model.lower():
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    return tokenizer

def get_wikitext2(nsamples, seed, seqlen, model, tokenizer):
    dataset_dir = DATA_DIR / "wikitext"
    if dataset_dir.is_dir():
        traindata = load_from_disk(dataset_dir / "traindata")
        testdata = load_from_disk(dataset_dir / "testdata")
    else:
        traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_ptb(nsamples, seed, seqlen, model, tokenizer):
    dataset_dir = DATA_DIR / "ptb"
    if dataset_dir.is_dir():
        traindata = load_from_disk(dataset_dir / "traindata")
        testdata = load_from_disk(dataset_dir / "testdata")
    else:
        traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
        testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

def get_c4(nsamples, seed, seqlen, model, tokenizer):
    dataset_dir = DATA_DIR / "c4"
    if dataset_dir.is_dir():
        traindata = load_from_disk(dataset_dir / "traindata")
        valdata = load_from_disk(dataset_dir / "valdata")
    else:
        data_files = {
            'train': 'en/c4-train.00000-of-01024.json.gz',
            'validation': 'en/c4-validation.00000-of-00008.json.gz',
        }
        traindata = load_dataset(
            'allenai/c4',
            'en',
            data_files=data_files,
            split='train',
            verification_mode='no_checks',
        )
        valdata = load_dataset(
            'allenai/c4',
            'en',
            data_files=data_files,
            split='validation',
            verification_mode='no_checks',
        )

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]


    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc

def get_loaders(name, nsamples=128, seed=0, seqlen=2048, model=''):
    model_name = os.path.basename(os.path.normpath(model))
    cache_file = (
        PROJECT_DIR / "cache" / f"{name}_{nsamples}_{seed}_{seqlen}_{model_name}.pt"
    )
    try:
        # These cache files are generated locally by this module and contain
        # TokenizerWrapper, so PyTorch 2.6+ cannot use weights_only=True.
        return torch.load(cache_file, weights_only=False)
    except (FileNotFoundError, RuntimeError):
        pass

    tokenizer = get_tokenizer(model)

    if 'wikitext2' in name:
        loaders= get_wikitext2(nsamples, seed, seqlen, model, tokenizer)
    if 'ptb' in name:
        loaders= get_ptb(nsamples, seed, seqlen, model, tokenizer)
    if 'c4' in name:
        loaders= get_c4(nsamples, seed, seqlen, model, tokenizer)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    torch.save(loaders,cache_file)
    return loaders
