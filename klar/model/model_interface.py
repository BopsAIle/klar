import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, AutoConfig, AutoModel
import pytorch_lightning
from torch.utils.data import DataLoader,Dataset
import json
from tqdm import tqdm
from pytorch_lightning import loggers
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import pdb
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import os
from .moe import MoE


class QwenPredictEntity(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.qwen = AutoModelForCausalLM.from_config(config)
        self.output_projection = nn.Linear(config.hidden_size, config.entity_size, bias=False)


class FusionModel(pytorch_lightning.LightningModule):
    def __init__(self, args): 
        super().__init__() 
        config = AutoConfig.from_pretrained(args.model_load_path, trust_remote_code=True)
        args.entity_size = config.entity_size
        args.embedding_size = config.hidden_size
        model = QwenPredictEntity(config)
        model.load_state_dict(torch.load(args.model_load_path+'/model_checkpoint.ckpt'), strict=True)
        model.output_projection = None
        self.model = model 

        self.batch_size = args.batch_size
        self.max_path_len = args.max_path_len

        if os.path.isfile(args.KG_embedding_path) and args.WO_KG_embedding == False:
            self.KGE = True
            KG_emb = torch.load(args.KG_embedding_path,map_location='cpu')
            self.embedding_size = args.embedding_size
            self.ent_emb = nn.Parameter(KG_emb['final_entity_embeddings'],requires_grad=False)
            self.rel_emb = nn.Parameter(KG_emb['final_relation_embeddings'],requires_grad=False)
            self.W_q = torch.nn.Linear(args.embedding_size,args.embedding_size)
            self.W_k = torch.nn.Linear(args.embedding_size,args.embedding_size)
            self.W_v = torch.nn.Linear(args.embedding_size,args.embedding_size)

            self.RNN = torch.nn.RNN(input_size=config.hidden_size, hidden_size=args.KG_RNN_hidden_size, num_layers=args.KG_RNN_num_layers, batch_first=True)
            self.KG_W_q = torch.nn.Linear(args.embedding_size,args.embedding_size)
            self.KG_W_k = torch.nn.Linear(args.embedding_size,args.embedding_size)
            self.KG_W_v = torch.nn.Linear(args.embedding_size,args.embedding_size)

            self.change_input_emd = False
            if(self.ent_emb.size()[1] != self.embedding_size and self.rel_emb.size()[1] != self.embedding_size):
                self.change_input_emd = True
                self.input_ent = torch.nn.Linear(self.ent_emb.size()[1],self.embedding_size)
                self.input_rel = torch.nn.Linear(self.rel_emb.size()[1],self.embedding_size)

        else:
            print('Running without KG embedding')
            self.KGE = False
            self.ent_emb = None
            self.rel_emb = None
        
        if(args.moe_use):
            self.output_projection = MoE(args.embedding_size, args.entity_size, args.moe_expert_num, args.moe_hidden_size, k=args.moe_k, noisy_gating=True, args=args)
        else:
            self.output_projection = nn.Linear(args.embedding_size, args.entity_size, bias=False)

        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.model_device = args.device
        self.loss_fct = CrossEntropyLoss(reduction='none')
        self.args = args

    def forward(self, inputs):
        inputs.to(self.model_device)
        outputs = self.model.generate(**inputs)
        return  outputs
    
    def configure_optimizers(self): # -> optimizer 
         return Adam(self.parameters(), lr=self.args.learning_rate, capturable=True)
        
    def training_step(self, batch, batch_idx): # -> train loss 
        input_ids,attention_mask,seq,seq_mask,same_label = \
            batch['input_ids'],batch['attention_mask'],batch['seq_id'],batch['seq_id_mask'],batch['same_label']
        outputs = self.model.qwen(
            input_ids = input_ids.to(self.model_device),
            attention_mask = attention_mask.to(self.model_device),
            head_mask=None,
            inputs_embeds=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        )
        sequence_output = outputs[0][:,-1:,:] #shape: [batch_size,1,emb_size(h)(1024)]

        if self.KGE:
            q = sequence_output
            k_ent = self.ent_emb[ seq[:,0::2] ].contiguous()
            k_rel = self.rel_emb[ seq[:,1::2] ].contiguous()
            if(self.change_input_emd):
                k_ent = self.input_ent(k_ent)
                k_rel = self.input_rel(k_rel)
            k = torch.zeros([self.batch_size,self.max_path_len*2+2,self.embedding_size]).to(self.model_device)#[b,id_seq_len(L),emb_size]
            k[:,0::2,:] = k_ent
            k[:,1::2,:] = k_rel
            v = k
            Q = self.W_q(q.float()) #[b,1,h]
            K = self.W_k(k.float()) #[b,L,h]
            V = self.W_v(v.float()) #[b,L,h]
            
            temp = torch.matmul(Q, K.transpose(1,2)) #[b,1,L]
            seq_mask = seq_mask.view(self.batch_size, 1, -1) #[b,1,L]
            temp[torch.where(seq_mask == 0)] = float("-inf")
            att = torch.matmul(torch.nn.functional.softmax(temp,dim = 2),V)
            sequence_output = sequence_output +att

            KG_output = self.calculate_KG_attention(k,outputs[0],seq_mask.view(self.batch_size,-1))
            sequence_output += KG_output

        sequence_output = sequence_output.squeeze(1)
        prediction_scores = self.output_projection(sequence_output)#
        if(self.args.moe_use):
            prediction_scores,moe_loss = prediction_scores
            self.log('moe_loss',moe_loss,prog_bar=True,on_epoch=True,logger=True,sync_dist=True)
        else:
            moe_loss = 0

        shifted_prediction_scores = prediction_scores
        assert shifted_prediction_scores.size() == same_label.size()
        loss = self.loss_fct(shifted_prediction_scores,same_label.float())
        same_label_count = torch.sum(same_label,dim=1)
        loss = loss.div(same_label_count)
        loss = torch.mean(loss)
        loss += moe_loss
            
        self.log('train_loss',loss,prog_bar=True,on_epoch=True,logger=True,sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        result = self._eval(batch, batch_idx)
        self.validation_step_outputs.append(np.array(result))
        return result
    
    def on_validation_epoch_end(self) -> None:

        ranks = np.concatenate([_ for _ in self.validation_step_outputs])
        hits20 = (ranks<=20).mean()
        hits10 = (ranks<=10).mean()
        hits3 = (ranks<=3).mean()
        hits1 = (ranks<=1).mean()

        self.log("Eval-hits10", hits10,sync_dist=True,prog_bar=True)
        self.log("Eval-hits20", hits20,sync_dist=True)
        self.log("Eval-hits3", hits3,sync_dist=True)
        self.log("Eval-hits1", hits1,sync_dist=True,prog_bar=True)
        self.log("Eval-mean_rank", ranks.mean(),sync_dist=True)
        self.log("Eval-mrr", (1. / ranks).mean(), prog_bar=True,sync_dist=True)
        # self.log("hits10", hits10, prog_bar=True)
        # self.log("hits1", hits1, prog_bar=True)
        self.validation_step_outputs.clear()
        
    def test_step(self, batch, batch_idx): 
        result = self._eval(batch, batch_idx)
        self.test_step_outputs.append(np.array(result))
        return result
    
    def on_test_epoch_end(self) -> None:
        
        ranks = np.concatenate([_ for _ in self.test_step_outputs])

        hits20 = (ranks<=20).mean()
        hits10 = (ranks<=10).mean()
        hits3 = (ranks<=3).mean()
        hits1 = (ranks<=1).mean()

        self.log("Test-hits10", hits10,sync_dist=True)
        self.log("Test-hits20", hits20,sync_dist=True)
        self.log("Test-hits3", hits3,sync_dist=True)
        self.log("Test-hits1", hits1,sync_dist=True)
        self.log("Test-mean_rank", ranks.mean(),sync_dist=True)
        self.log("Test-mrr", (1. / ranks).mean(),sync_dist=True)
        self.test_step_outputs.clear()

    def _eval(self, batch, batch_idx):
        input_ids,attention_mask,seq,seq_mask,same_label = \
            batch['input_ids'],batch['attention_mask'],batch['seq_id'],batch['seq_id_mask'],batch['same_label']
        batch_size = min(self.batch_size, input_ids.size(0))
        outputs = self.model.qwen(
            input_ids = input_ids.to(self.model_device),
            attention_mask = attention_mask.to(self.model_device),
            head_mask=None,
            inputs_embeds=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        )
        sequence_output = outputs[0][:,-1:,:] #shape: [batch_size,1,emb_size(h)(1024)]

        if self.KGE:
            q = sequence_output

            k_ent = self.ent_emb[ seq[:,0::2] ].contiguous()
            k_rel = self.rel_emb[ seq[:,1::2] ].contiguous()
            if(self.change_input_emd):
                k_ent = self.input_ent(k_ent)
                k_rel = self.input_rel(k_rel)

            k = torch.zeros([batch_size,self.max_path_len*2+2,self.embedding_size]).to(self.model_device)#[b,id_seq_len(L),emb_size]
            k[:,0::2,:] = k_ent
            k[:,1::2,:] = k_rel
            v = k
            Q = self.W_q(q.float()) #[b,1,h]
            K = self.W_k(k.float()) #[b,L,h]
            V = self.W_v(v.float()) #[b,L,h]
            
            temp = torch.matmul(Q, K.transpose(1,2)) #[b,1,L]
            seq_mask = seq_mask.view(batch_size, 1, -1) #[b,1,L]
            temp[torch.where(seq_mask == 0)] = float("-inf")
            att = torch.matmul(torch.nn.functional.softmax(temp,dim = 2),V)
            sequence_output = sequence_output +att
            
            KG_output = self.calculate_KG_attention(k,outputs[0],seq_mask.view(batch_size,-1))
            sequence_output = sequence_output + KG_output
        
        sequence_output = sequence_output.squeeze(1)
        prediction_scores = self.output_projection(sequence_output)#
        if(self.args.moe_use):
            prediction_scores,moe_loss = prediction_scores
            self.log('moe_loss',moe_loss,prog_bar=True,on_epoch=True,logger=True,sync_dist=True)

        shifted_prediction_scores = prediction_scores
        assert shifted_prediction_scores.size() == same_label.size()
        batch_ranks = []
        _, outputs = torch.sort(shifted_prediction_scores, dim=1, descending=True)
        _, outputs = torch.sort(outputs, dim=1)
        
        # ranks = outputs[torch.arange(min(self.batch_size, input_ids.size(0))), label].detach().to('cpu') + 1
        inf_tensor = torch.full_like(outputs,self.args.entity_size+1,dtype=int)
        ranks = torch.where(same_label==1,outputs,inf_tensor)
        ranks,_ = torch.min(ranks,dim=1)
        ranks = ranks.detach().to('cpu') + 1
        batch_ranks.append(np.array(ranks))
        batch_ranks = np.concatenate([_ for _ in batch_ranks])

        return batch_ranks

    def calculate_KG_attention(self, KG_emb_seq, GPT_emb_seq, seq_mask):

        Q = self.KG_W_q(KG_emb_seq.float()) #[b,L1,h]
        K = self.KG_W_k(GPT_emb_seq) #[b,L2,h]
        V = self.KG_W_v(GPT_emb_seq) #[b,L2,h]
        temp = torch.matmul(Q, K.transpose(1,2)) #[b,L1,L2]
        att = torch.matmul(torch.nn.functional.softmax(temp,dim = 2),V) #[b,L1,h]
        att[seq_mask==0]=0
        KG_emb_seq = KG_emb_seq + att
        h0 = torch.zeros(self.args.KG_RNN_num_layers, KG_emb_seq.size()[0], self.args.KG_RNN_hidden_size)
        output, hn = self.RNN(KG_emb_seq,h0.to(self.args.device))
        
        return output[:, -1:, :]  #size [batch_size,1,hidden_size 1024]
