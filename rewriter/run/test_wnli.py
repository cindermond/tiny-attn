from datasets import load_dataset
import torch

import sys
import ast
import csv


def train(dataset: str="wnli", cache_dir = "data") -> None:
    #initializes
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    raw_datasets = load_dataset("glue", dataset, cache_dir=cache_dir)
    num_labels = 2
    trainloader =  torch.utils.data.DataLoader(raw_datasets['train'], batch_size=1, shuffle=False)
    testloader = torch.utils.data.DataLoader(raw_datasets['test'], batch_size=1, shuffle=False)

    count0 = 0
    count1 = 0
    with torch.no_grad():
        for batch in trainloader:
            if batch['label'].item() == 0:
                count0+=1
            else:
                count1+=1

    if count0 > count1:
        fill_in = 0
    else:
        fill_in = 1
    with torch.no_grad():
        with open(f"{dataset}.tsv","wt") as f:
            tsv_writer = csv.writer(f, delimiter='\t')
            tsv_writer.writerow(['index','prediction'])
            for batch in testloader:
                tsv_writer.writerow([batch['idx'].item(), fill_in])


if __name__=='__main__':
    if len(sys.argv) > 1:
        raw_arguments = sys.argv[1]
        arguments = ast.literal_eval(raw_arguments)
        print(arguments)
        train(**arguments)
    else:
        train()
