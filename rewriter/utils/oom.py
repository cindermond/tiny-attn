import gc
import warnings

import torch

def search_num_chunks(model, preprocess_fn, dataset, bsz, passes=100, verbose=False):
    device = next(model.parameters()).device
    start_bsz = bsz
    low = 1
    high = None
    num_chunks = 1
    low_chunks = 1
    high_chunks = None
    fits_in_memory = False

    while not fits_in_memory:
        gc_cuda()
        try:
            # get a sorted sampler for sampling longest examples first (worst case in memory)
            sampler = LongestFirstSampler(dataset, preprocess_fn)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=start_bsz, sampler=sampler)

            model.train()
            for j, batch in enumerate(dataloader):
                inputs = preprocess_fn(batch)
                if batch['label'].dtype == torch.double: batch['label'] = batch['label'].float() # for stsb
                inputs.update({'labels': batch['label']})
                inputs.to(device)

                for mbatch in chunk_batch(dict(inputs), num_chunks):
                    output = model(mbatch['input_ids'], mbatch['attention_mask'], labels=mbatch['labels'])
                    loss = output.loss / num_chunks
                    loss.backward()

                if j == passes - 1 or j == len(dataloader) - 1:
                    break

            model.zero_grad()
            low = bsz
            low_chunks = num_chunks
            fits_in_memory = True

            # break when full trainset
            if j == len(dataloader) - 1: break
            if low == start_bsz: break # break if low works

            if high is not None:
                if high / low <= 2: break
                bsz = high // 2
                num_chunks = high_chunks * 2
            else:
                bsz = low * 2
                num_chunks = low_chunks // 2
        except RuntimeError as e:
            if is_oom_error(e):
                gc_cuda()
                high = bsz
                high_chunks = num_chunks
                bsz = high // 2
                num_chunks = high_chunks * 2
                if high / low <= 2:
                    break
            else:
                raise

    # underestimate when over computation budget
    gc_cuda()
    bsz = low
    num_chunks = low_chunks
    return num_chunks

def chunk_batch(batch, num_chunks):
    assert num_chunks >= 1
    if num_chunks == 1:
        return [batch]

    if isinstance(batch, dict):
        microbatches = {k: v.chunk(num_chunks) for k, v in batch.items()}
        num_chunks = len(list(microbatches.values())[0]) # edge case where batch has less than chunks
        microbatches = [{k: v[i] for k, v in microbatches.items()} for i in range(num_chunks)]
    else: # tuple or list iterator
        microbatches = [v.chunk(num_chunks) for v in batch]
        microbatches = list(zip(*microbatches))
    return microbatches


# memory management, from 
# https://github.com/BlackHC/toma/blob/master/toma/torch_cuda_memory.py
# and cpu_memory.py

def is_oom_error(exception):
    return (is_cuda_oom(exception) or is_cudnn_snafu(exception) or is_cpu_oom(exception) or is_cudnn_rnn_snafu(exception) or is_cudnn_conv_snafu(exception))

def is_cuda_oom(exception):
    return (isinstance(exception, RuntimeError)
            and len(exception.args) == 1
            and 'CUDA out of memory.' in exception.args[0])

# lstm with too high sequence length throws this
def is_cudnn_rnn_snafu(exception):
    return (isinstance(exception, RuntimeError)
            and len(exception.args) == 1
            and 'cuDNN error: CUDNN_STATUS_EXECUTION_FAILED' in exception.args[0])

def is_cudnn_conv_snafu(exception):
    return (isinstance(exception, RuntimeError)
            and len(exception.args) == 1
            and 'Unable to find a valid cuDNN algorithm to run convolution' in exception.args[0])

def is_cudnn_snafu(exception):
    # For/because of https://github.com/pytorch/pytorch/issues/4107
    return (isinstance(exception, RuntimeError)
            and len(exception.args) == 1
            and 'cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.' in exception.args[0])

def is_cpu_oom(exception):
    return (isinstance(exception, RuntimeError)
            and len(exception.args) == 1
            and "DefaultCPUAllocator: can't allocate memory" in exception.args[0])

def gc_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# modifying from https://github.com/Oldpan/Pytorch-Memory-Utils/blob/master/gpu_mem_track.py
# and
# https://github.com/pytorch/pytorch/issues/32101
def cuda_tensors():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                tensor = obj
            else:
                continue
            if tensor.is_cuda:
                    yield tensor
        except:
            pass

def cuda_tensor_usage():
    size = sum(tensor.element_size() * tensor.nelement() for tensor in cuda_tensors())
    return size

# samples longest elements first assuming huggingface tokenized dataset
class LongestFirstSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, preprocess_fn):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        lengths = []
        for i, batch in enumerate(dataloader):
            inputs = preprocess_fn(batch)
            lengths.append((i, inputs.input_ids.shape[1]))
        lengths = sorted(lengths, key=lambda tup: tup[1], reverse=True)
        self.indices = [i for i, l in lengths]

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        return iter(self.indices) 
