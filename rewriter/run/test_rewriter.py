from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
import torch
import os.path

from torch import nn
import sys
import ast

from rewriter.model.RobertaForSC import RobertaForSC

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

def train(dataset: str="rte", batch_size: int=16, nhead: int=4, d_hid: int=512, nlayers: int=1, dropout: float=0.1, is_rewriter: bool=True, output_nlayers: int=1, cache_dir: str='data', load_name:str = "weight-best-dataset=rte-nlayers=1-d_hid=512-nhead=4-lr=5e-05-is_rewriter=True-output_nlayers=1-weight_decay=0.01-seed=42-warmup_steps=1000-epoch_num=50-scheduler_type=linear") -> None:
    #initializes
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("roberta-large", cache_dir=cache_dir)

    raw_datasets = load_dataset("glue", dataset, cache_dir=cache_dir)
    is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
    if is_regression:
        num_labels = 1
    else:
        label_list = raw_datasets["train"].unique("label")
        label_list.sort()  # Let's sort it for determinism
        num_labels = len(label_list)
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
    def metric_fn(preds, labels):
        result = metric.compute(predictions=preds, references=labels)
        score = list(result.values())[0]
        return score, result

    model = RobertaForSC.from_pretrained("roberta-large", output_nlayers=output_nlayers, is_rewriter=is_rewriter, rewriter_nhead=nhead, rewriter_d_hid=d_hid, rewriter_dropout=dropout, rewriter_nlayers=nlayers, n_labels=num_labels)
    last_cp = torch.load(os.path.abspath(f'log/result/{load_name}.pt'))
    model.load_state_dict(last_cp['state_dict'])

    model = model.to(device)
    print(eval(model, preprocess_fn, devloader, metric, dataset))

    
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