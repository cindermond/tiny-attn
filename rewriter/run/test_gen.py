from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
import torch
import os.path

from torch import nn
import sys
import ast
import random
import nltk

from rewriter.model.bart_exp import BartForConditionalGenerationBL


def train(dataset: str="xsum", nhead: int=4, d_hid: int=256, nlayers: int=1, dropout: float=0.1, is_rewriter: bool=True, output_nlayers: int=1, cache_dir: str='data', seed: int=1234, load_name:str = "weight-best-code=0201=dataset=xsum-nlayers=1-d_hid=256-nhead=4-lr=4e-05-is_rewriter=True-output_nlayers=1-weight_decay=0.01-seed=42-warmup_steps=100000-epoch_num=20-scheduler_type=cosine-encoder_attn_size=64-decoder_attn_size=64", encoder_attn_size = 64, decoder_attn_size = 64, code = '0201') -> None:
    #reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    #initializes
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large", cache_dir=cache_dir)

    raw_datasets = load_dataset(dataset, cache_dir=cache_dir)
    devloader = torch.utils.data.DataLoader(raw_datasets['test'], batch_size=1, shuffle=False)

    def preprocess_fn(examples, label_max_length=128):
        result = tokenizer(examples["document"], padding=True, truncation='longest_first', max_length=512, return_tensors='pt')
        label = tokenizer(examples["summary"], padding=True, truncation='longest_first', max_length=label_max_length, return_tensors='pt')["input_ids"]
        return result, label

    metric = load_metric("rouge")
    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.

    model = BartForConditionalGenerationBL.from_pretrained("facebook/bart-large", output_nlayers=output_nlayers, is_rewriter=is_rewriter, rewriter_nhead=nhead, rewriter_d_hid=d_hid, rewriter_dropout=dropout, rewriter_nlayers=nlayers, encoder_attn_size=encoder_attn_size, decoder_attn_size=decoder_attn_size, code=code)
    last_cp = torch.load(os.path.abspath(f'log/result/{load_name}.pt'))
    model.load_state_dict(last_cp['state_dict'])

    model = model.to(device)
    print(eval(model, preprocess_fn, devloader, metric, tokenizer))


    
@torch.no_grad()
def eval(model: nn.Module, preprocess_fn, dataloader: torch.utils.data.DataLoader, metric, tokenizer) -> float:
    #initialize
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.eval()
    for index, batch in enumerate(dataloader):
        inputs, labels = preprocess_fn(batch,512)
        inputs.to(device)
        preds = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], num_beams=6, max_length=100, min_length=10)
        
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
        
        metric.add_batch(predictions=preds, references=labels)

        if (index+1)%100 == 0:
            print(index)
        
    result = metric.compute(use_stemmer = True)
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    return result["rouge2"], result

if __name__=='__main__':
    if len(sys.argv) > 1:
        raw_arguments = sys.argv[1]
        arguments = ast.literal_eval(raw_arguments)
        print(arguments)
        train(**arguments)
    else:
        train()
