import itertools

args = {
    'dataset': ['cb','rte'],
    'lr': [0.001],
    'weight_decay': [0.02],
    'seed': [1],
    'warmup_steps': [0,64],
    'epoch_num': [20],
    'load_name': ['None'],
    'scheduler_type':['constant'],
    'eval_times': [1],
    'attention_emd': [1],
    'attention_head': [1]
}

with open('grid_search.sh','w') as f:
    for vl in itertools.product(*list(args.values())):
        command = "sbatch"
        cfg = dict(list(zip(args.keys(), vl)))
        command += f" few_shot.sh"
        for p, v in cfg.items():
            command += " --"
            command += p
            command += f" {v}"
        command += "\n"
        f.write(command)