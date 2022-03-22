from collections import defaultdict
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoConfig
from transformers.optimization import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup,get_cosine_schedule_with_warmup
import torch
import os.path

from torch import nn
import sys
import ast
import random
import math
from torch.optim import AdamW

from rewriter.model.bart_gen import BartForConditionalGenerationBL
from rewriter.utils.oom import chunk_batch, search_num_chunks

def train(dataset: str="xsum", lr: float=0.00005, batch_size: int=2, epoch_num: int=20, nhead: int=4, d_hid: int=512, nlayers: int=1, dropout: float=0.1, is_shuffled: bool=True, is_rewriter: bool=True, output_nlayers: int=1, weight_decay: float=0, cache_dir: str='data', seed: int=1234, warmup_steps: int=0, load_name:str = "None", scheduler_type:str = "linear", eval_times:int = 10000) -> None:
    #reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    #initializes
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    save_name = f'dataset={dataset}-nlayers={nlayers}-d_hid={d_hid}-nhead={nhead}-lr={lr}-is_rewriter={is_rewriter}-output_nlayers={output_nlayers}-weight_decay={weight_decay}-seed={seed}-warmup_steps={warmup_steps}-epoch_num={epoch_num}-scheduler_type={scheduler_type}'
    print(save_name)
    
    tokenizer = AutoTokenizer.from_pretrained("roberta-large", cache_dir=cache_dir)

    raw_datasets = load_dataset(dataset, cache_dir=cache_dir)
    trainloader = torch.utils.data.DataLoader(raw_datasets['train'], batch_size=batch_size, shuffle=is_shuffled)
    devloader = torch.utils.data.DataLoader(raw_datasets['validation'], batch_size=batch_size, shuffle=False)

    def preprocess_fn(examples):
        result = tokenizer(examples["document"], padding=True, truncation='longest_first', max_length=(tokenizer.model_max_length), return_tensors='pt')
        label = tokenizer(examples["summary"], padding=True, truncation='longest_first', max_length=(tokenizer.model_max_length), return_tensors='pt')["input_ids"]
        return result, label

    metric = load_metric("rouge")
    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def metric_fn(preds, labels):
        result = metric.compute(predictions=preds, references=labels)
        result["rouge1"]=result["rouge1"].mid.fmeasure
        result["rouge2"]=result["rouge2"].mid.fmeasure
        result["rougeL"]=result["rougeL"].mid.fmeasure
        del result["rougeLsum"]
        score = result["rouge2"]
        return score, result

    start_epoch = 0
    max_dev_metric = -float('inf')
    def make_cp(pt_obj, current_epoch):
        assert 'state_dict' in dir(pt_obj)
        return {'state_dict': pt_obj.state_dict(),
                'epoch': current_epoch,
                'max_dev_metric': max_dev_metric}

    model = BartForConditionalGenerationBL.from_pretrained("facebook/bart-large", output_nlayers=output_nlayers, is_rewriter=is_rewriter, rewriter_nhead=nhead, rewriter_d_hid=d_hid, rewriter_dropout=dropout, rewriter_nlayers=nlayers)
    if os.path.exists(os.path.abspath(f'log/weight/weight-last-{save_name}.pt')):
        last_cp = torch.load(os.path.abspath(f'log/weight/weight-last-{save_name}.pt'))
        model.load_state_dict(last_cp['state_dict'])
        start_epoch = last_cp['epoch'] + 1
        max_dev_metric = last_cp['max_dev_metric']
    elif load_name != "None":
        last_cp = torch.load(os.path.abspath(f'log/result/{load_name}.pt'))
        model.load_state_dict(last_cp['state_dict'])
        max_dev_metric = last_cp['max_dev_metric']

    for p in model.model.shared.parameters():
        p.requires_grad = False
    for p in model.model.encoder.layers.parameters():
        p.requires_grad = False
    if output_nlayers == 0:
        for p in model.model.decoder.layers[1::2].parameters():
            p.requires_grad = False
    else:
        for p in model.model.decoder.layers[1:-(2*output_nlayers-1):2].parameters():
            p.requires_grad = False

    model = model.to(device)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),lr=lr,weight_decay=weight_decay)
    if os.path.exists(os.path.abspath(f'log/weight/opt-last-{save_name}.pt')):
        last_cp = torch.load(os.path.abspath(f'log/weight/opt-last-{save_name}.pt'))
        optimizer.load_state_dict(last_cp['state_dict'])
    # chunk dataset for oom errors
    num_chunks = 1
    if scheduler_type=="linear":
        scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=warmup_steps,num_training_steps=len(trainloader)*epoch_num,last_epoch = start_epoch-1)
    elif scheduler_type=="constant":
        scheduler = get_constant_schedule_with_warmup(optimizer,num_warmup_steps=warmup_steps,last_epoch = start_epoch-1)
    elif scheduler_type=="cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=warmup_steps,num_training_steps=len(trainloader)*epoch_num,last_epoch = start_epoch-1)


    #training
    model.train()
    model.model.encoder.layers.eval()
    model.model.shared.eval()
    if output_nlayers == 0:
        model.model.decoder.layers[1::2].eval()
    else:
        model.model.decoder.layers[1:-(2*output_nlayers-1):2].eval()
    total_loss = 0
    log_interval = 1000
    eval_interval = math.floor(len(trainloader)/eval_times)

    for epoch in range(start_epoch, epoch_num):
        for index, batch in enumerate(trainloader):
            inputs, label = preprocess_fn(batch)
            inputs.update({'labels': label})
            inputs.to(device)
            optimizer.zero_grad()

            for mbatch in chunk_batch(dict(inputs), num_chunks):
                output = model(mbatch['input_ids'], mbatch['attention_mask'], labels=mbatch['labels'])
                loss = output.loss / num_chunks
                loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            total_loss += loss.item()
            scheduler.step()

            if index % log_interval == log_interval - 1:
                cur_loss = total_loss/log_interval
                print(f'Epoch:{epoch+1}/{epoch_num} Progress:{index+1}/{len(trainloader)} Loss:{cur_loss}')
                total_loss = 0
            
            if index % eval_interval == eval_interval - 1:
                dev_metric, dev_metrics = eval(model, preprocess_fn, devloader, metric_fn)
                print(f'Epoch{epoch+1} dev_metrics: {dev_metrics}')

                if dev_metric > max_dev_metric:
                    max_dev_metric = dev_metric
                    print(f'Epoch{epoch+1} max_dev_metric: {max_dev_metric}')
                    torch.save(make_cp(model, epoch), os.path.abspath(f'log/weight/weight-best-{save_name}.pt'))
                model.train()
                model.model.encoder.layers.eval()
                model.model.shared.eval()
                if output_nlayers == 0:
                    model.model.decoder.layers[1::2].eval()
                else:
                    model.model.decoder.layers[1:-(2*output_nlayers-1):2].eval()

        torch.save(make_cp(model, epoch) ,os.path.abspath(f'log/weight/weight-last-{save_name}.pt'))
        torch.save(make_cp(optimizer, epoch) ,os.path.abspath(f'log/weight/opt-last-{save_name}.pt'))

    
@torch.no_grad()
def eval(model: nn.Module, preprocess_fn, dataloader: torch.utils.data.DataLoader, metric) -> float:
    #initialize
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.eval()
    for batch in dataloader:
        inputs, label = preprocess_fn(batch)
        inputs.to(device)
        output = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], num_beams=5, max_new_tokens=100)
        metric.add_batch(predictions=output, references=label)
    result = metric.compute()
    result["rouge1"]=result["rouge1"].mid.fmeasure
    result["rouge2"]=result["rouge2"].mid.fmeasure
    result["rougeL"]=result["rougeL"].mid.fmeasure
    del result["rougeLsum"]
    return result["rouge2"], result

if __name__=='__main__':
    if len(sys.argv) > 1:
        raw_arguments = sys.argv[1]
        arguments = ast.literal_eval(raw_arguments)
        print(arguments)
        train(**arguments)
    else:
        train()
