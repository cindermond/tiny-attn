import itertools

args = {
    'dataset': ['mnli'],
    'nhead': [4],
    'd_hid': [512],
    'is_rewriter': [False],
    'lr': [0.00001,0.00005,0.0001,0.0005,0.001,0.003,0.005],
    'dropout': [0.1],
    'nlayers': [1],
    'output_nlayers': [0],
    'weight_decay': [0,0.01,0.05,0.1,0.2,0.3,0.4,0.5],
    'batch_size': [16],
    'seed': [42],
    'warmup_steps': [0],
    'epoch_num': [20],
    'load_name': ['weight-best-model_name=roberta-large-dataset=mnli-nlayers=1-d_hid=512-nhead=4-lr=0.001-is_rewriter=False-output_nlayers=0-weight_decay=0-seed=42-warmup_steps=20000-epoch_num=20-scheduler_type=linear-attention_emd=1-attention_head=1-structure=m0'],
    'scheduler_type':['linear','cosine'],
    'eval_times': [4],
    'model_name': ['roberta-large'],
    'attention_emd': [1],
    'attention_head': [1]
}

def args2jobname(**kwargs):
    jobname = f"lr={kwargs['lr']}-weight_decay={kwargs['weight_decay']}-warmup_steps={kwargs['warmup_steps']}-scheduler_type={kwargs['scheduler_type']}-d_hid={kwargs['d_hid']}-epoch_num={kwargs['epoch_num']}-attention_head={kwargs['attention_head']}-attention_emd={kwargs['attention_emd']}"
    return jobname

with open('grid_search.sh','w') as f:
    for vl in itertools.product(*list(args.values())):
        command = "sbatch"
        cfg = dict(list(zip(args.keys(), vl)))
        jobname = args2jobname(**cfg)
        command += f" -J {jobname} -d singleton ada.sh"
        for p, v in cfg.items():
            command += " --"
            command += p
            command += f" {v}"
        command += "\n"
        f.write(command)