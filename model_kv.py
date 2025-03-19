import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map_with_path

from dataclasses import dataclass

class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool = False):
        super().__init__()
        self.weight = mx.ones((ndim,))
        self.bias = mx.zeros((ndim,)) if bias else None

    def __call__(self, input):
        return mx.fast.layer_norm(input, self.weight, self.bias, 1e-5, stream=mx.cpu)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.mask = mx.tril(mx.ones([config.block_size, config.block_size])).reshape(1, 1, config.block_size, config.block_size)
        # FlashAtention is not ignored here

    def __call__(self, x, kvcache=None):
        B, T, C = x.shape # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        query, key, value = mx.split(self.c_attn(x), 3, axis=2)
        key = key.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3) # (B, nh, T, hs)
        query = query.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3) # (B, nh, T, hs)
        value = value.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3) # (B, nh, T, hs)
        if kvcache:
            prev_k, prev_v = kvcache
            key = mx.concatenate([prev_k, key], axis=2)
            value = mx.concatenate([prev_v, value], axis=2)

        new_kvcache = [key, value]
        curr_T = key.shape[2]
        if kvcache:
            
            att = (query @ key.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(key.shape[3]))
            att = mx.where(mx.ones_like(self.mask[:,:,:T,:curr_T]) == 0, float('-1e9'), att)
            att = mx.softmax(att.astype(mx.float32), axis=-1).astype(x.dtype)
            att = self.attn_dropout(att)
            y = mx.matmul(att, value)
        else:
            att = (query @ key.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(key.shape[3]))
            att = mx.where(self.mask[:,:,:T,:T] == 0, float('-1e9'), att)
            att = mx.softmax(att.astype(mx.float32), axis=-1).astype(x.dtype)
            att = self.attn_dropout(att)
            y = mx.matmul(att, value)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C) # re-assemble all head outputs side by side
        y = self.resid_dropout(self.c_proj(y))
        return y, new_kvcache


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def __call__(self, x):
        
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def __call__(self, x, kvcache=None):
        attn_out, cache_ele = self.attn(self.ln_1(x), kvcache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, cache_ele


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class transformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.h = [Block(config) for _ in range(config.n_layer)]
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = transformer(config)
        
        self.update(tree_map_with_path(self._init_weights, self.parameters()))

        ########## The following code is included in _init_weights() ##########
        #
        # for pn, p in self.named_parameters():
        #     if pn.endswith('c_proj.weight'):
        #         torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        #
        #######################################################################
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        params = tree_flatten(self.parameters())
        n_params = 0
        for pn, p in params:
            if not pn.endswith("mask"):
                if non_embedding and pn == 'transformer.wpe.weight':
                    pass
                else:
                    n = 1
                    for d in p.shape:
                        n *= d
                    n_params += n
        return n_params
    
    def _init_weights(self, module_path, module_weight):
        if module_path.endswith('c_proj.weight'):
            return nn.init.normal(mean=0.0, std=0.02/math.sqrt(2 * self.config.n_layer))(module_weight)
        module_name = module_path.split('.')
        module = self
        for i in range(len(module_name) - 1):
            if isinstance(module, list):
                module = module[int(module_name[i])]
            else:
                module = module.__getattr__(module_name[i])
        if isinstance(module, nn.Linear):
            if module_name[-1] == 'weight':
                module_weight = nn.init.normal(mean=0.0, std=0.02)(module['weight'])
            elif module_name[-1] == 'bias':
                module_weight = nn.init.constant(0)(module['bias'])
        elif isinstance(module, nn.Embedding):
            module_weight = nn.init.normal(mean=0.0, std=0.02)(module['weight'])
        return module_weight

    def __call__(self, idx, targets=None, kvcache=None):
        b, t = idx.shape
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = mx.arange(0, t, 1, dtype=idx.dtype)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        if not kvcache:
            kvcache = [None] * self.config.n_layer
        else:
            x = x[:, [-1], :]
        new_kvcache = []
        for block, kvcache_block in zip(self.transformer.h, kvcache):
            x, cache_ele = block(x, kvcache=kvcache_block)
            new_kvcache.append(cache_ele)
        
        x = self.transformer.ln_f(x)
        if targets is not None:
            # Weight Tying
            logits = self.lm_head(x)
            ########## This line may have bug ##########
            loss = nn.losses.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), axis=-1)
            ########## This line may have bug ##########
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss, new_kvcache

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe['weight'] = self.transformer.wpe['weight'][:block_size]

        ######### Unused in original NanoGPT repo, ignored here. #########
        # for block in self.transformer.h:
        #     if hasattr(block.attn, 'bias'):
        #         block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]
        ######### Unused in original NanoGPT repo, ignored here. #########

    def generate(self, idx: mx.array, max_new_tokens=256, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        kvcache = None
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.shape[1] <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _, kvcache  = self(idx_cond, kvcache=kvcache)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v = mx.sort(mx.topk(logits, min(top_k, logits.shape[-1])))[...,::-1]
                logits = mx.where(logits < v[:, [-1]], -float('Inf'), logits)

            # apply softmax to convert logits to (normalized) probabilities
            probs = mx.softmax(logits)
            # Sample from the distribution
            idx_next = mx.random.categorical(mx.log(probs), 1)[...,mx.newaxis]
            # Append sampled index to the running sequence
            idx = mx.concatenate([idx, idx_next], axis=1)
        return idx