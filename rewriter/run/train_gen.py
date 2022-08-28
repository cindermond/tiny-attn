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
from rewriter.model.gpt_gen import GPT2LMHeadModelBL
from rewriter.utils.bleu import BLEUScore
import json


def train(lr: float=0.01, batch_size: int=1, epoch_num: int=100, weight_decay: float=0.0, cache_dir: str='data', seed: int=1234, warmup_steps: int=0, load_name:str = "None", scheduler_type:str = "linear", eval_times:int = 1, attn_emb=1, attn_head=1, attn_dropout=0.1, is_sequential=True) -> None:
    #reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    #initializes
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    save_name = f'lr={lr}-weight_decay={weight_decay}-seed={seed}-warmup_steps={warmup_steps}-epoch_num={epoch_num}-scheduler_type={scheduler_type}-attn_emb={attn_emb}-attn_head={attn_head}-is_seq={is_sequential}'
    print(save_name)
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2-medium", cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token

    with open(os.path.abspath('data/e2e/train.jsonl'),'r') as f:
        trainset = [json.loads(x) for x in f]
    with open(os.path.abspath('data/e2e/dev.jsonl'),'r') as f:
        devset = [json.loads(x) for x in f]
    with open(os.path.abspath('data/e2e/test.jsonl'),'r') as f:
        testset = [json.loads(x) for x in f]
    
    for data in trainset:
        data["gt_length"] = len(tokenizer(data["mr"]+"<|endoftext|>")["input_ids"])

    start_epoch = 0
    max_dev_metric = -float('inf')
    def make_cp(pt_obj, current_epoch):
        assert 'state_dict' in dir(pt_obj)
        return {'state_dict': pt_obj.state_dict(),
                'epoch': current_epoch,
                'max_dev_metric': max_dev_metric}

    model = GPT2LMHeadModelBL.from_pretrained("gpt2-medium", attention_emb=attn_emb, attention_head=attn_head, attention_dropout=attn_dropout, is_sequential=is_sequential)

    if os.path.exists(os.path.abspath(f'log/weight/weight-last-{save_name}.pt')):
        last_cp = torch.load(os.path.abspath(f'log/weight/weight-last-{save_name}.pt'))
        model.load_state_dict(last_cp['state_dict'])
        start_epoch = last_cp['epoch'] + 1
        max_dev_metric = last_cp['max_dev_metric']
    elif load_name != "None":
        last_cp = torch.load(os.path.abspath(f'log/result/{load_name}.pt'))
        model.load_state_dict(last_cp['state_dict'])
        max_dev_metric = last_cp['max_dev_metric']

    for name, p in model.named_parameters():
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
    model.eval()
    for block in model.transformer.h:
        block.tiny_attn.train()

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),lr=lr,weight_decay=weight_decay)
    if os.path.exists(os.path.abspath(f'log/weight/opt-last-{save_name}.pt')):
        last_cp = torch.load(os.path.abspath(f'log/weight/opt-last-{save_name}.pt'))
        optimizer.load_state_dict(last_cp['state_dict'])

    if scheduler_type=="linear":
        scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=warmup_steps,num_training_steps=len(trainset)*epoch_num,last_epoch = max(len(trainset)*(start_epoch-1),-1))
    elif scheduler_type=="constant":
        scheduler = get_constant_schedule_with_warmup(optimizer,num_warmup_steps=warmup_steps,last_epoch = max(len(trainset)*(start_epoch-1),-1))
    elif scheduler_type=="cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=warmup_steps,num_training_steps=len(trainset)*epoch_num,last_epoch = max(len(trainset)*(start_epoch-1),-1))


    #training
    total_loss = 0
    log_interval = 100

    for epoch in range(start_epoch, epoch_num):
        random.shuffle(trainset)
        batched_trainset = [trainset[i:i+batch_size] for i in range(0,len(trainset),batch_size)]
        eval_interval = math.floor(len(batched_trainset)/eval_times)
        for index, batch in enumerate(batched_trainset):
            input_seq = [data['mr'] + "<|endoftext|>" + data['ref'] + "<|endoftext|>" for data in batch]
            gt_length = [data['gt_length'] for data in batch]
            inputs = tokenizer(input_seq, return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            labels = torch.where(attention_mask==1, input_ids, -100)
            for i,l in enumerate(gt_length):
                labels[i,:l] = -100
            optimizer.zero_grad()

            output = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = output.loss/len(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            total_loss += loss.item()
            scheduler.step()

            if index % log_interval == log_interval - 1:
                cur_loss = total_loss/log_interval
                print(f'Epoch:{epoch+1}/{epoch_num} Progress:{index+1}/{len(batched_trainset)} Loss:{cur_loss}')
                total_loss = 0
            
            if index % eval_interval == eval_interval - 1:
                dev_metric = eval(model, devset, tokenizer)
                print(f'Epoch{epoch+1} dev_metric: {dev_metric}')

                if dev_metric > max_dev_metric:
                    max_dev_metric = dev_metric
                    print(f'Epoch{epoch+1} max_dev_metric: {max_dev_metric}')
                    torch.save(make_cp(model, epoch), os.path.abspath(f'log/weight/weight-best-{save_name}.pt'))
                for block in model.transformer.h:
                    block.tiny_attn.train()

        torch.save(make_cp(model, epoch) ,os.path.abspath(f'log/weight/weight-last-{save_name}.pt'))
        torch.save(make_cp(optimizer, epoch) ,os.path.abspath(f'log/weight/opt-last-{save_name}.pt'))
    
    best_cp = torch.load(os.path.abspath(f'log/weight/weight-best-{save_name}.pt'))
    model.load_state_dict(best_cp['state_dict'])
    print(f'Test set metric: {eval(model, testset, tokenizer)}')

    
@torch.no_grad()
def eval(model: nn.Module, evalset, tokenizer) -> float:
    #initialize
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.eval()
    metric = BLEUScore()
    temp_label = [evalset[0]["ref"]]
    temp_data = evalset[0]["mr"]
    for data in evalset[1:]:
        if len(data["mr"])>0:
            input_ids = tokenizer(temp_data+"<|endoftext|>", return_tensors="pt")["input_ids"].to(device)            
            generated_result = model.generate(input_ids, max_new_tokens=100, min_length=input_ids.size(dim=1)+5, num_beams=5,eos_token_id=50256,pad_token_id=50256)[0][input_ids.size(dim=1):]
            generated_seq = tokenizer.decode(generated_result, skip_special_tokens=True).strip()
            metric.append(generated_seq, temp_label)
            temp_label = [data["ref"]]
            temp_data = data["mr"]
        else:
            temp_label.append(data["ref"])

    input_ids = tokenizer(temp_data+"<|endoftext|>", return_tensors="pt")["input_ids"].to(device)            
    generated_result = model.generate(input_ids, max_new_tokens=100, min_length=input_ids.size(dim=1)+5, num_beams=5,eos_token_id=50256,pad_token_id=50256)[0][input_ids.size(dim=1):]
    generated_seq = tokenizer.decode(generated_result, skip_special_tokens=True).strip()
    metric.append(generated_seq, temp_label)

    return metric.score()

if __name__=='__main__':
    if len(sys.argv) > 1:
        raw_arguments = sys.argv[1]
        arguments = ast.literal_eval(raw_arguments)
        print(arguments)
        train(**arguments)
    else:
        train()
