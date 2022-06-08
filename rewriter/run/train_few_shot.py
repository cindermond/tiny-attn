from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup,get_cosine_schedule_with_warmup
import torch
import os.path

from torch import nn
import sys
import ast
import random
import math
from torch.optim import AdamW
from yaml import load
from rewriter.model.AlbertMLM import AlbertMLM

task_to_keys = {
    "rte": ("premise", "hypothesis"),
    "cb": ("premise", "hypothesis"),
}

def train(dataset: str="rte", lr: float=0.0005, epoch_num: int=10, weight_decay: float=0, cache_dir: str='data', seed: int=1234, warmup_steps: int=0, load_name:str = "None", scheduler_type:str = "linear", eval_times:int = 1, attention_emd = 1, attention_head = 1) -> None:
    #reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    #initializes
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    save_name = f'dataset={dataset}-lr={lr}-weight_decay={weight_decay}-seed={seed}-warmup_steps={warmup_steps}-epoch_num={epoch_num}-scheduler_type={scheduler_type}-attention_emd={attention_emd}-attention_head={attention_head}'
    print(save_name)
    
    tokenizer = AutoTokenizer.from_pretrained("albert-xxlarge-v2", cache_dir=cache_dir)

    raw_datasets = load_dataset("juny116/few_glue", dataset, cache_dir=cache_dir)
    trainloader = raw_datasets['train']
    devloader = raw_datasets['validation']
    testloader = raw_datasets['test']

    sentence1_key, sentence2_key = task_to_keys[dataset]
    def preprocess_fn(examples, label_word = None):
        result = tokenizer(examples[sentence1_key]+"?[MASK]. "+ examples[sentence2_key], return_tensors='pt')

        if label_word is not None:
            label = tokenizer(examples[sentence1_key]+"?"+label_word+". "+ examples[sentence2_key], return_tensors='pt')["input_ids"]
            label = torch.where(result["input_ids"] == tokenizer.mask_token_id, label, -100)
        else:
            label = None
        return result, label

    start_epoch = 0
    max_dev_metric = -float('inf')
    def make_cp(pt_obj, current_epoch):
        assert 'state_dict' in dir(pt_obj)
        return {'state_dict': pt_obj.state_dict(),
                'epoch': current_epoch,
                'max_dev_metric': max_dev_metric}

    model = AlbertMLM.from_pretrained("albert-xxlarge-v2", attention_emb=attention_emd, attention_head=attention_head)
    if os.path.exists(os.path.abspath(f'log/weight/weight-last-{save_name}.pt')):
        last_cp = torch.load(os.path.abspath(f'log/weight/weight-last-{save_name}.pt'))
        model.load_state_dict(last_cp['state_dict'])
        start_epoch = last_cp['epoch'] + 1
        max_dev_metric = last_cp['max_dev_metric']
    elif load_name != "None":
        last_cp = torch.load(os.path.abspath(f'log/result/{load_name}.pt'))
        model.load_state_dict(last_cp['state_dict'])
        max_dev_metric = last_cp['max_dev_metric']

    for p in model.albert.embeddings.parameters():
        p.requires_grad = False
    for name, p in model.albert.encoder.albert_layer_groups[0].albert_layers.named_parameters():
        if 'tiny_attn' not in name:
            p.requires_grad = False

    total_para = 0
    trainable_para = 0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        total_para += nn
        if p.requires_grad:
            trainable_para += nn

    train_percent = trainable_para/total_para
    print(f'trainable parameters: {train_percent}')

    model = model.to(device)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),lr=lr,weight_decay=weight_decay)
    if os.path.exists(os.path.abspath(f'log/weight/opt-last-{save_name}.pt')):
        last_cp = torch.load(os.path.abspath(f'log/weight/opt-last-{save_name}.pt'))
        optimizer.load_state_dict(last_cp['state_dict'])
    if scheduler_type=="linear":
        scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=warmup_steps,num_training_steps=len(trainloader)*epoch_num,last_epoch = start_epoch-1)
    elif scheduler_type=="constant":
        scheduler = get_constant_schedule_with_warmup(optimizer,num_warmup_steps=warmup_steps,last_epoch = start_epoch-1)
    elif scheduler_type=="cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=warmup_steps,num_training_steps=len(trainloader)*epoch_num,last_epoch = start_epoch-1)




    #training
    model.train()
    model.albert.embeddings.eval()
    model.albert.encoder.albert_layer_groups[0].albert_layers.eval()
    for l in model.albert.encoder.albert_layer_groups[0].albert_layers:
        l.tiny_attn.train()
    total_loss = 0
    log_interval = 100
    eval_interval = math.floor(len(trainloader)/eval_times)
    for epoch in range(start_epoch, epoch_num):
        trainloader.shuffle()
        for index, batch in enumerate(trainloader):
            if dataset == "cb":
                if batch['label'] == 0:
                    label_word = " yes "
                elif batch['label'] == 1:
                    label_word = " no "
                else:
                    label_word = " maybe "
            if dataset == "rte":
                if batch['label'] == 0:
                    label_word = " yes "
                elif batch['label'] == 1:
                    label_word = " no "
            inputs, label = preprocess_fn(batch, label_word)

            inputs.update({'labels': label})
            inputs.to(device)
            optimizer.zero_grad()

            output = model(inputs['input_ids'], inputs['attention_mask'], labels=inputs['labels'])
            loss = output.loss
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
                dev_metric = eval(model, preprocess_fn, devloader, dataset, tokenizer)
                print(f'Epoch{epoch+1} dev_metric: {dev_metric}')

                if dev_metric > max_dev_metric:
                    max_dev_metric = dev_metric
                    print(f'Epoch{epoch+1} max_dev_metric: {max_dev_metric}')
                model.train()
                model.albert.embeddings.eval()
                model.albert.encoder.albert_layer_groups[0].albert_layers.eval()
                for l in model.albert.encoder.albert_layer_groups[0].albert_layers:
                    l.tiny_attn.train()

    test_metric_last = eval(model, preprocess_fn, testloader, dataset, tokenizer)
    print(f'test_metric_last: {test_metric_last}')
        


@torch.no_grad()
def eval(model: nn.Module, preprocess_fn, dataloader, dataset, tokenizer) -> float:
    #initialize
    model.eval()
    device = next(model.parameters()).device
    correct_number = 0
    total_number = 0
    for batch in dataloader:
        inputs, _ = preprocess_fn(batch)
        if dataset == "cb":
            if batch['label'] == 0:
                label_word = "yes"
            elif batch['label'] == 1:
                label_word = "no"
            else:
                label_word = "maybe"
        if dataset == "rte":
            if batch['label'] == 0:
                label_word = "yes"
            elif batch['label'] == 1:
                label_word = "no"
        label_id = tokenizer.convert_tokens_to_ids(label_word)
        inputs.to(device)
        mask_token_index = (inputs['input_ids'] == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
        output = model(inputs['input_ids'], inputs['attention_mask']).logits[0, mask_token_index]
        if dataset == "cb":
            if output[0,label_id]>= output[0,tokenizer.convert_tokens_to_ids("yes")] and output[0,label_id]>= output[0,tokenizer.convert_tokens_to_ids("no")] and output[0,label_id]>= output[0,tokenizer.convert_tokens_to_ids("maybe")]:
                correct_number += 1
        if dataset == "rte":
            if output[0,label_id]>= output[0,tokenizer.convert_tokens_to_ids("yes")] and output[0,label_id]>= output[0,tokenizer.convert_tokens_to_ids("no")]:
                correct_number += 1
        total_number += 1

    return correct_number/total_number

if __name__=='__main__':
    if len(sys.argv) > 1:
        raw_arguments = sys.argv[1]
        arguments = ast.literal_eval(raw_arguments)
        print(arguments)
        train(**arguments)
    else:
        train()
