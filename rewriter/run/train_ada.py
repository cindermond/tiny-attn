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
from torch.optim import AdamW

from rewriter.model.RobertaForSCAda import RobertaForSCAda
from rewriter.utils.oom import chunk_batch, search_num_chunks

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def train(dataset: str="sst2", lr: float=0.00005, batch_size: int=8, epoch_num: int=20, nhead: int=4, d_hid: int=512, nlayers: int=1, dropout: float=0.1, is_rewriter: bool=False, output_nlayers: int=0, weight_decay: float=0, cache_dir: str='data', seed: int=1234, warmup_steps: int=0, load_name:str = "None", scheduler_type:str = "linear", eval_times:int = 1, model_name = 'roberta-large', attention_emd = 1, attention_head = 1) -> None:
    #reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    #initializes
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    save_name = f'model_name={model_name}-dataset={dataset}-nlayers={nlayers}-d_hid={d_hid}-nhead={nhead}-lr={lr}-is_rewriter={is_rewriter}-output_nlayers={output_nlayers}-weight_decay={weight_decay}-seed={seed}-warmup_steps={warmup_steps}-epoch_num={epoch_num}-scheduler_type={scheduler_type}-attention_emd={attention_emd}-attention_head={attention_head}'
    print(save_name)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    raw_datasets = load_dataset("glue", dataset, cache_dir=cache_dir)
    is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
    if is_regression:
        num_labels = 1
    else:
        label_list = raw_datasets["train"].unique("label")
        label_list.sort()  # Let's sort it for determinism
        num_labels = len(label_list)
    trainloader = torch.utils.data.DataLoader(raw_datasets['train'], batch_size=batch_size, shuffle=True)
    if dataset == "mnli":
        val_word = 'validation_matched'
    else:
        val_word = 'validation'
    devloader = torch.utils.data.DataLoader(raw_datasets[val_word], batch_size=batch_size, shuffle=False)

    sentence1_key, sentence2_key = task_to_keys[dataset]
    def preprocess_fn(examples):
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=True, truncation='longest_first', max_length=(tokenizer.model_max_length), return_tensors='pt')
        return result

    metric = load_metric("glue", dataset)
    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.

    start_epoch = 0
    max_dev_metric = -float('inf')
    def make_cp(pt_obj, current_epoch):
        assert 'state_dict' in dir(pt_obj)
        return {'state_dict': pt_obj.state_dict(),
                'epoch': current_epoch,
                'max_dev_metric': max_dev_metric}

    model = RobertaForSCAda.from_pretrained(model_name, output_nlayers=output_nlayers, is_rewriter=is_rewriter, rewriter_nhead=nhead, rewriter_d_hid=d_hid, rewriter_dropout=dropout, rewriter_nlayers=nlayers, n_labels=num_labels, attention_emd=attention_emd, attention_head=attention_head)
    if os.path.exists(os.path.abspath(f'log/weight/weight-last-{save_name}.pt')):
        last_cp = torch.load(os.path.abspath(f'log/weight/weight-last-{save_name}.pt'))
        model.load_state_dict(last_cp['state_dict'])
        start_epoch = last_cp['epoch'] + 1
        max_dev_metric = last_cp['max_dev_metric']
    elif load_name != "None":
        last_cp = torch.load(os.path.abspath(f'log/result/{load_name}.pt'))
        model.load_state_dict(last_cp['state_dict'])
        max_dev_metric = last_cp['max_dev_metric']

    for p in model.roberta.embeddings.parameters():
        p.requires_grad = False
    for name, p in model.roberta.encoder.layer.named_parameters():
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
    print(f'num:{trainable_para}, trainable parameters: {train_percent}')

    model = model.to(device)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),lr=lr,weight_decay=weight_decay)
    if os.path.exists(os.path.abspath(f'log/weight/opt-last-{save_name}.pt')):
        last_cp = torch.load(os.path.abspath(f'log/weight/opt-last-{save_name}.pt'))
        optimizer.load_state_dict(last_cp['state_dict'])
    # chunk dataset for oom errors
    num_chunks = search_num_chunks(model, preprocess_fn, trainloader.dataset, batch_size)
    print('num chunks to chunk batches into is: ', num_chunks)
    if scheduler_type=="linear":
        scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=warmup_steps,num_training_steps=len(trainloader)*epoch_num,last_epoch = start_epoch-1)
    elif scheduler_type=="constant":
        scheduler = get_constant_schedule_with_warmup(optimizer,num_warmup_steps=warmup_steps,last_epoch = start_epoch-1)
    elif scheduler_type=="cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=warmup_steps,num_training_steps=len(trainloader)*epoch_num,last_epoch = start_epoch-1)




    #training
    model.train()
    model.roberta.embeddings.eval()
    model.roberta.encoder.layer.eval()
    for l in model.roberta.encoder.layer:
        for t in l.tiny_attn:
            t.train()
    total_loss = 0
    log_interval = 100
    eval_interval = math.floor(len(trainloader)/eval_times)
    for epoch in range(start_epoch, epoch_num):
        for index, batch in enumerate(trainloader):
            inputs = preprocess_fn(batch)
            if batch['label'].dtype == torch.double: batch['label'] = batch['label'].float() # for stsb
            inputs.update({'labels': batch['label']})
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
                dev_metric, dev_metrics = eval(model, preprocess_fn, devloader, metric, dataset)
                print(f'Epoch{epoch+1} dev_metrics: {dev_metrics}')

                if dev_metric > max_dev_metric:
                    max_dev_metric = dev_metric
                    print(f'Epoch{epoch+1} max_dev_metric: {max_dev_metric}')
                    torch.save(make_cp(model, epoch), os.path.abspath(f'log/weight/weight-best-{save_name}.pt'))
                model.train()
                model.roberta.embeddings.eval()
                model.roberta.encoder.layer.eval()
                for l in model.roberta.encoder.layer:
                    for t in l.tiny_attn:
                        t.train()

        torch.save(make_cp(model, epoch) ,os.path.abspath(f'log/weight/weight-last-{save_name}.pt'))
        torch.save(make_cp(optimizer, epoch) ,os.path.abspath(f'log/weight/opt-last-{save_name}.pt'))


@torch.no_grad()
def eval(model: nn.Module, preprocess_fn, dataloader: torch.utils.data.DataLoader, metric, dataset) -> float:
    #initialize
    model.eval()
    device = next(model.parameters()).device
    for batch in dataloader:
        inputs = preprocess_fn(batch)
        if batch['label'].dtype == torch.double: batch['label'] = batch['label'].float() # for stsb
        inputs.update({'labels': batch['label']})
        inputs.to(device)
        output = model(inputs['input_ids'], inputs['attention_mask'], labels=inputs['labels'])
        if dataset != "stsb":
            _, pos = torch.max(output.logits, 1)
        else:
            pos = output.logits
        metric.add_batch(predictions=pos, references=inputs['labels'])
    result = metric.compute()
    return list(result.values())[0], result

if __name__=='__main__':
    if len(sys.argv) > 1:
        raw_arguments = sys.argv[1]
        arguments = ast.literal_eval(raw_arguments)
        print(arguments)
        train(**arguments)
    else:
        train()
