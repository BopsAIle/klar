import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import pytorch_lightning
from torch.utils.data import DataLoader,Dataset
import json
from tqdm import tqdm
from pytorch_lightning import loggers
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from .standard_data import ToDataset,get_dataset
import pdb

def pad_batch(batch:list):
    
    new_batch = dict()
    max_input_ids = max([len( _['input_ids'] )for _ in batch ])

    padded_input_ids = []
    padded_attention_mask = []
    same_label = []

    seq_id = []
    seq_id_mask = []

    for i in batch:
        padded_input_ids.append(torch.nn.functional.pad(torch.tensor(i['input_ids']),(max_input_ids-len(i['input_ids']),0),'constant',1))
        padded_attention_mask.append(torch.nn.functional.pad(torch.tensor(i['attention_mask']),(max_input_ids-len(i['attention_mask']),0),'constant',0))
        same_label.append(i['same_label'])
        seq_id.append(i['seq_id'])
        seq_id_mask.append(i['seq_id_mask'])

    input_ids = torch.stack(padded_input_ids,0)
    attention_mask = torch.stack(padded_attention_mask,0)
    same_label = torch.stack(same_label)

    seq_id = torch.stack(seq_id,0)
    seq_id_mask = torch.stack(seq_id_mask,0)

    new_batch['input_ids'] = input_ids
    new_batch['attention_mask'] = attention_mask
    new_batch['seq_id'] = seq_id
    new_batch['seq_id_mask'] = seq_id_mask
    new_batch['same_label'] = same_label
   
    return new_batch

class DInterface(pytorch_lightning.LightningDataModule): 
    def __init__(self, args): 
        super().__init__()
        self.args = args
    def prepare_data(self):
        pass
    def setup(self, stage=None):
        self.data_train = get_dataset("train",self.args)
        self.data_val   = get_dataset("val"  ,self.args)
        self.data_test  = get_dataset("test" ,self.args)
    def train_dataloader(self) -> DataLoader: 
        return DataLoader(self.data_train, drop_last=True, collate_fn=pad_batch, num_workers=self.args.data_loader_workers, batch_size= self.args.batch_size, shuffle=True)
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.data_val, drop_last=False, collate_fn=pad_batch, num_workers=self.args.data_loader_workers, batch_size= self.args.batch_size, shuffle=False)
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.data_test, drop_last=False, collate_fn=pad_batch, num_workers=self.args.data_loader_workers, batch_size= self.args.batch_size, shuffle=False)
