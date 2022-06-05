from datasets import load_dataset
from transformers import AutoTokenizer
import torch
import os.path

import sys
import ast
import random
import csv

from rewriter.model.RobertaForSC import RobertaForSC


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli_matched": ("premise", "hypothesis"),
    "mnli_mismatched": ("premise", "hypothesis"),
    "ax": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def train(dataset: str="ax", nhead: int=4, d_hid: int=512, nlayers: int=1, dropout: float=0.1, is_rewriter: bool=False, output_nlayers: int=0, cache_dir: str='data', seed: int=1234, load_name:str = "weight-best-model_name=roberta-large-dataset=mnli-nlayers=1-d_hid=512-nhead=4-lr=0.001-is_rewriter=False-output_nlayers=0-weight_decay=0-seed=42-warmup_steps=20000-epoch_num=20-scheduler_type=linear-attention_emd=1-attention_head=1-structure=m0", model_name = 'roberta-large', attention_emd = 1, attention_head = 1, structure = 'm0') -> None:
    #reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    #initializes
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    raw_datasets = load_dataset("glue", dataset, cache_dir=cache_dir)
    is_regression = raw_datasets["test"].features["label"].dtype in ["float32", "float64"]
    if is_regression:
        num_labels = 1
    else:
        #label_list = raw_datasets["test"].unique("label")
        #label_list.sort()  # Let's sort it for determinism
        #num_labels = len(label_list)
        num_labels = 3
    testloader = torch.utils.data.DataLoader(raw_datasets['test'], batch_size=1, shuffle=False)

    sentence1_key, sentence2_key = task_to_keys[dataset]
    def preprocess_fn(examples):
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=True, truncation='longest_first', max_length=(tokenizer.model_max_length), return_tensors='pt')
        return result

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.

    model = RobertaForSC.from_pretrained(model_name, output_nlayers=output_nlayers, is_rewriter=is_rewriter, rewriter_nhead=nhead, rewriter_d_hid=d_hid, rewriter_dropout=dropout, rewriter_nlayers=nlayers, n_labels=num_labels, attention_emd=attention_emd, attention_head=attention_head, structure=structure)
    last_cp = torch.load(os.path.abspath(f'log/result/{load_name}.pt'))
    model.load_state_dict(last_cp['state_dict'])

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
    print(f'trainable parameters: {train_percent}')
    print(trainable_para)
    exit()


    model = model.to(device)
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        with open(f"{dataset}.tsv","wt") as f:
            tsv_writer = csv.writer(f, delimiter='\t')
            tsv_writer.writerow(['index','prediction'])
            for batch in testloader:
                inputs = preprocess_fn(batch)
                inputs.to(device)
                output = model(inputs['input_ids'], inputs['attention_mask'])
                if dataset != "stsb":
                    _, pos = torch.max(output.logits, 1)
                else:
                    pos = output.logits
                if pos.item() == 0:
                    temp = "entailment"
                elif pos.item() == 1:
                    temp = "neutral"
                else:
                    temp = "contradiction"
                tsv_writer.writerow([batch['idx'].item(), temp])


if __name__=='__main__':
    if len(sys.argv) > 1:
        raw_arguments = sys.argv[1]
        arguments = ast.literal_eval(raw_arguments)
        print(arguments)
        train(**arguments)
    else:
        train()
