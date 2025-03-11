import torch
import json
import mlx.core as mx
import numpy as np
import os
out_dir = 'out-shakespeare-char'

model_in = os.path.join(out_dir, 'ckpt.pt')
model_weights_npz = os.path.join(out_dir,'weights.npz')
model_args_json = os.path.join(out_dir, 'model_args.json')
model_config_json = os.path.join(out_dir, 'model_config.json')
model_pt = torch.load(model_in, map_location=torch.device("cpu"))
# Save model args (JSON)
model_args = model_pt['model_args']
with open(model_args_json, 'w') as fp:
    json.dump(model_args, fp)
# Save model config (JSON)
config = model_pt['config']
with open(model_config_json, 'w') as fp:
    json.dump(config, fp)
# Save model weight (NPZ)
state_dict = model_pt['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
np_state_dict = {key: mx.array(value.cpu().numpy(), dtype=mx.float32) for key, value in state_dict.items()}
np.savez(model_weights_npz, **np_state_dict)

