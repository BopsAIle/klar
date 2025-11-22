import torch
import pytorch_lightning
from torch.utils.data import DataLoader,Dataset
import json
import numpy as np
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, set_seed
import sys
import os
from torch import nn
import argparse

class QwenPredictEntity(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.qwen = AutoModelForCausalLM.from_config(config)
        self.output_projection = nn.Linear(config.hidden_size, config.entity_size, bias=False)

args = argparse.ArgumentParser()
args.add_argument("-p","--save_path", default=None, help="Path to save the model")
args.add_argument("-n","--entity_num",type=int, default=None, help="The number of entities under this task")
args.add_argument("-mn","--model_name", default="Qwen/Qwen3-0.6B", help="Qwen model name. e.g. 'Qwen/Qwen3-0.6B'")
args = args.parse_args()

# load the qwen3 as base model and get the state dict
qwen = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    dtype="auto",
    device_map="auto"
)
qwen_state_dict = qwen.state_dict()

# add a linear projection layer relative to entity number to the base model
config = AutoConfig.from_pretrained(args.model_name)
config.tie_word_embeddings = False
config.entity_size = args.entity_num
config.save_pretrained(args.save_path)
model = QwenPredictEntity(config)
model.load_state_dict(qwen_state_dict, strict=False)

torch.save(model.state_dict(), os.path.join(args.save_path,'model_checkpoint.ckpt'))