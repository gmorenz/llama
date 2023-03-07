# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json

from pathlib import Path

import llama
from llama import ModelArgs, Transformer, Tokenizer, LLaMA


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    ckpt_dir = Path(ckpt_dir)
    with open(ckpt_dir / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.HalfTensor)
    model = Transformer(ckpt_dir, model_args, device="meta")
    torch.set_default_tensor_type(torch.FloatTensor)
    state_dict = {
        name: llama.model.load_weight_file(ckpt_dir.joinpath(name), param)
        for name, param in model.named_parameters()
        if not name.startswith("layers.")
    }
    model.load_state_dict(state_dict, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 1024,
    max_batch_size: int = 1,
):
    torch.manual_seed(1)

    generator = load(
        ckpt_dir, tokenizer_path, max_seq_len, max_batch_size
    )

    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "Journal of Philosophy, 2020, 23, 102-350\nWhat is the meaning of life?\nJonathan Archer Ba(hons) M Phil PhD\nUniversity of Toronto, Department of Philosophy\nAccepted for publication",
    ]
    results = generator.generate(
        prompts, max_gen_len=max_seq_len, temperature=temperature, top_p=top_p
    )

    for result in results:
        print(result)
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
