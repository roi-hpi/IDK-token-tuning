import argparse
import json
from pathlib import Path
import datasets
import numpy as np
import torch
import wandb
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, GenerationConfig

from idk_trainer import IdkTrainer
import pandas as pd
import os
from tqdm import tqdm


class IdkDecoder:
    def __init__(self, model_path, config, device):
        self.device = device
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer_path = model_path
        self.model = model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, do_lower_case=True)
        # self.tokenizer.add_special_tokens({'additional_special_tokens': ['[IDK_our]']})
        # self.model.resize_token_embeddings(len(self.tokenizer))

    def _tokenize_text(self, examples):
        return self.tokenizer(examples['text'])

    def _group_text(self, examples):
        chunk_size = self.model.config.max_position_embeddings
        # Concatenate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        # Compute length of concatenated texts
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the last chunk if it's smaller than chunk_size
        total_length = (total_length // chunk_size) * chunk_size
        # Split by chunks of max_len
        result = {
            k: [t[i: i + chunk_size] for i in range(0, total_length, chunk_size)]
            for k, t in concatenated_examples.items()
        }
        return result
