from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup,get_cosine_schedule_with_warmup
import torch
import os.path

from torch import nn
import sys
import ast
import random
import math
import nltk
from torch.optim import AdamW

from rewriter.model.bart_gen import BartForConditionalGenerationBL
from rewriter.utils.oom import chunk_batch


def train(dataset: str="xsum", lr: float=0.00003, batch_size: int=8, epoch_num: int=20, nhead: int=4, d_hid: int=256, nlayers: int=1, dropout: float=0.1, is_shuffled: bool=True, is_rewriter: bool=True, output_nlayers: int=1, weight_decay: float=0, cache_dir: str='data', seed: int=1234, warmup_steps: int=0, load_name:str = "weight-best-dataset=xsum-nlayers=1-d_hid=256-nhead=4-lr=0.0002-is_rewriter=True-output_nlayers=1-weight_decay=0.01-seed=42-warmup_steps=10000-epoch_num=5-scheduler_type=linear-encoder_attn_size=32-decoder_attn_size=32", scheduler_type:str = "constant", eval_times:int = 1, encoder_attn_size = 32, decoder_attn_size = 32) -> None:
    #reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    #initializes
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    save_name = f'dataset={dataset}-nlayers={nlayers}-d_hid={d_hid}-nhead={nhead}-lr={lr}-is_rewriter={is_rewriter}-output_nlayers={output_nlayers}-weight_decay={weight_decay}-seed={seed}-warmup_steps={warmup_steps}-epoch_num={epoch_num}-scheduler_type={scheduler_type}-encoder_attn_size={encoder_attn_size}-decoder_attn_size={decoder_attn_size}'
    print(save_name)
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large", cache_dir=cache_dir)

    raw_datasets = load_dataset(dataset, cache_dir=cache_dir)
    devloader = torch.utils.data.DataLoader(raw_datasets['validation'].select(range(1600)), batch_size=batch_size, shuffle=False)

    def preprocess_fn(examples):
        result = tokenizer(examples["document"], padding=True, truncation='longest_first', max_length=512, return_tensors='pt')
        label = tokenizer(examples["summary"], padding=True, truncation='longest_first', max_length=128, return_tensors='pt')["input_ids"]
        return result, label

    metric = load_metric("rouge")
    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.

    model = BartForConditionalGenerationBL.from_pretrained("facebook/bart-large", output_nlayers=output_nlayers, is_rewriter=is_rewriter, rewriter_nhead=nhead, rewriter_d_hid=d_hid, rewriter_dropout=dropout, rewriter_nlayers=nlayers, encoder_attn_size=encoder_attn_size, decoder_attn_size=decoder_attn_size)

    last_cp = torch.load(os.path.abspath(f'log/result/{load_name}.pt'))
    model.load_state_dict(last_cp['state_dict'])

    for p in model.model.shared.parameters():
        p.requires_grad = False
    for p in model.model.encoder.layers.parameters():
        p.requires_grad = False
    if output_nlayers == 0:
        for p in model.model.decoder.layers.parameters():
            p.requires_grad = False
    else:
        for p in model.model.decoder.layers[:-output_nlayers].parameters():
            p.requires_grad = False

    model = model.to(device)
    print(eval(model, preprocess_fn, devloader, metric, tokenizer))


    
@torch.no_grad()
def eval(model: nn.Module, preprocess_fn, dataloader: torch.utils.data.DataLoader, metric, tokenizer) -> float:
    #initialize
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.eval()
    for batch in dataloader:
        inputs, labels = preprocess_fn(batch)
        inputs.to(device)
        preds = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], num_beams=6, max_length=60, min_length=10)
        
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
        
        metric.add_batch(predictions=preds, references=labels)
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
