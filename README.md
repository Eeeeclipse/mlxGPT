# `mlxGPT`

This repo is developed based on [nanoGPT_mlx](https://github.com/vithursant/nanoGPT_mlx) but will be more like the original version of [nanoGPT](https://github.com/karpathy/nanoGPT).

It is mainly designed for inferencing models trained by [nanoGPT](https://github.com/karpathy/nanoGPT). I will provide code to convert the .pt model to mlx model.

It is also the start of my CS498MLS's project UMA. 
<!-- 
## install

Create a conda environment using the provided
   environment configuration file.

```bash
conda env create -f environment.yaml
```

Activate conda environment.
```bash
conda activate apple_mlx
```

Dependencies:
- [mlx](https://ml-explore.github.io/mlx/build/html/index.html)
- [numpy](https://numpy.org/install/)
-  `datasets` for huggingface datasets (if you want to download + preprocess OpenWebText)
-  `tiktoken` for OpenAI's fast BPE code
-  `tensorboardX` for optional logging
-  `tqdm` for progress bars

## quick start
To train a character-level GPT, prepare shakespeare dataset similar to nanoGPT. This will create a `train.bin` and `val.bin` in that data directory.
```bash
python data/shakespeare/prepare.py
```

Now, let's train a "baby GPT" model on your MAC GPU:
```bash
python train.py configs/train_gpt2_shakespeare.py
```

On my Macbook M3 Pro, I am observing `~0.37 iterations/second` when training a `~45M parameter` GPT-2 model, at `batch_size=64` (i.e., `local_batch_size=4` and `gradient_accumulation=16`).

![repro124m](assets/baby_gpt2_shakespeare_pretrain_loss.png)

So once the training finishes we can sample from the best model by pointing the sampling script at this directory:
```bash
python sample.py --out_dir=gpt2_small_shakespeare
```

## openwebtext
To train a GPT-2 model on OpenWebText similar to nanoGPT, first prepare the dataset:
```bash
python data/openwebtext/prepare.py
```

Then, train a 124M GPT-2 model on your MAC GPU:
```bash
python train.py configs/train_gpt2_owt.py
```

## todos
- [ ] disable weight decay on non-decay params in optimizer
- [ ] add bfloat16 training support
- [ ] integrate Eleuther Eval
- [ ] add checkpoint conversion for loading pre-trained HF models 
- [x] add saveing and loading pre-trained MLX models 
- [ ] enable finetuning models from pre-trained checkpoints
- [x] enable inference with pre-trained models

## acknowledgements
Thank you [Andrej Karpthy](https://github.com/karpathy) for creating the nanoGPT codebase. It's been awesome for quick prototyping! -->
