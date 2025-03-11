import os
import argparse
import tiktoken
import time
import json
import pickle
import mlx.core as mx
from mlx.utils import tree_unflatten, tree_flatten

from mlxGPT.model_2 import GPT, GPTConfig

init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = './out-shakespeare-char' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1 # number of samples to draw
max_new_tokens = 40 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
mx.random.seed(seed)

model_weights_path = os.path.join(out_dir,'weights.npz')
model_args_path = os.path.join(out_dir,'model_args.json')
model_config_path = os.path.join(out_dir,'model_config.json')

with open(model_args_path, "r") as f:
    model_args = json.load(f)

with open(model_config_path, "r") as f:
    model_config = json.load(f)

model = GPT(GPTConfig(**model_args))

weights = mx.load(model_weights_path)

model.update(tree_unflatten(list(weights.items())))
mx.eval(model.parameters())

# look for the meta pickle in case it is available in the dataset folder
meta_path = os.path.join('data', model_config['dataset'], 'meta.pkl')
print(f"Loading meta from {meta_path}...")
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
# TODO want to make this more general to arbitrary encoder/decoder schemes
stoi, itos = meta['stoi'], meta['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (mx.array([start_ids], dtype=mx.uint32))

# run generation
start = time.time()
for k in range(num_samples):
    y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
    print(decode(y[0].tolist()))
    print('---------------')
end = time.time()
