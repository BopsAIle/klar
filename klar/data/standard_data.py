from transformers import AutoTokenizer
from torch.utils.data import Dataset
import json
import os
import random
import torch

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)  
cached_graph = {}

def randget_entity_neighbor(triplet, reverse_graph):
    neigh_triplet = []
    if(triplet == None):
        return None
    try:
        nei = list(reverse_graph[triplet[0]].keys())
        neigh_triplet = [tuple((nei[i], reverse_graph[triplet[0]][nei[i]],triplet[0])) for i in range(len(nei))]
        # delete the triplet itself in neigh_triplet
        triplet = tuple(reversed(triplet))
        if( triplet in  neigh_triplet ):
            neigh_triplet.remove(triplet)
    except KeyError:
        pass
    if ( len(neigh_triplet) == 0 ):
        return None
    return random.sample(neigh_triplet,1)[0]
    
def get_triplet_subpath(triplet , reverse_graph, max_path_len, mask_tail = True):
    # get the path of the entity in the triplet
    all_triplet = []
    # get the neighbor of the initial triplet
    neighbor_triplet = triplet
    for i in range(max_path_len):
        # continue to find the neighbor of the neighbor's head
        neighbor_triplet = randget_entity_neighbor(neighbor_triplet, reverse_graph)
        if(neighbor_triplet is not None):
            all_triplet.insert(0,neighbor_triplet)
        else:
            break
    return all_triplet

class ToDataset(Dataset):
    def __init__(self, _dict,args):
        self.triples = _dict['mode_file_triplet']
        self.reverse_graph = _dict['reverse_graph']
        self.entity2id = _dict['entity2id']
        self.relation2id = _dict['relation2id']
        self.relation2text = _dict['relation2text']
        self.same_label_dict = _dict['same_label_dict']
        self.args = args

    def __getitem__(self, index):
        triple = self.triples[index]
        mask_tail_triplet = get_triplet_subpath(triple, self.reverse_graph,max_path_len=self.args.max_path_len)
        seq = ''
        seq_id = torch.zeros([(self.args.max_path_len+1)*2,],dtype = int)
        seq_len = (len(mask_tail_triplet)+1)*2
        for i in range(len(mask_tail_triplet)):
            k = mask_tail_triplet[i]
            if(self.relation2text != None):
                seq = seq + k[0] + ' ' +self.relation2text[k[1]] +' '
            else:
                seq = seq + k[0] + ' ' +k[1] +' '

            seq_id[i*2] = self.entity2id[k[0]]
            seq_id[i*2+1] = self.relation2id[k[1]]

        if(self.relation2text != None):
            seq = seq + triple[0]+' '+self.relation2text[triple[1]]
        else:
            seq = seq + triple[0]+' '+triple[1]
            
        seq_id[seq_len-2] = self.entity2id[triple[0]]
        seq_id[seq_len-1] = self.relation2id[triple[1]]
        inputs = tokenizer(seq)
        seq_id_mask = torch.zeros([self.args.max_path_len*2+2,],dtype = int)
        seq_id_mask[0:seq_len] = 1

        head_id = str(self.entity2id[triple[0]])
        relation_id = str(self.relation2id[triple[1]])
        same_label = torch.zeros([self.args.entity_size,],dtype = int)
        for i in self.same_label_dict[head_id][relation_id]:
            same_label[i] = 1

        feature = {'input_ids': inputs['input_ids'] ,'attention_mask':inputs['attention_mask']}
        feature['seq_id'] = seq_id
        feature['seq_id_mask'] = seq_id_mask
        feature['same_label'] = same_label

        return feature
    
    def __len__(self):
        return len(self.triples)
    
        
def get_dataset(mode,args):

    graph_path = os.path.join(args.data_dir,'reverse_graph.json')
    if graph_path in cached_graph:
        reverse_graph = cached_graph[graph_path]
    else:
        reverse_graph = json.load(open(graph_path,'r'))
        cached_graph[graph_path] = reverse_graph

    entity2id = dict()
    with open(os.path.join(args.data_dir,'entity2id.txt'),'r') as file:
        for i in file.read().splitlines():
            entity2id[i.split('\t')[0]] = int(i.split('\t')[1])

    relation2id = dict()
    with open(os.path.join(args.data_dir, 'relation2id.txt'),'r') as file:
        for i in file.read().splitlines():
            relation2id[i.split('\t')[0]] = int(i.split('\t')[1])

    relation2text = dict()
    if os.path.isfile(os.path.join(args.data_dir, 'relation2text.txt')):
        with open(os.path.join(args.data_dir, 'relation2text.txt'),'r') as file:
            relation2text = json.load(file)
    else:
        relation2text = None

    mode_file_triplet = []
    with open(os.path.join(args.data_dir, mode+'.txt'),'r') as file:
        file_triplet = file.read().splitlines()
        for i in file_triplet:
            mode_file_triplet.append(tuple(i.split('\t')))
    
    with open(os.path.join(args.data_dir, mode+'_label.json'),'r') as file:
        same_label_dict = json.load(file)
    
    _dict = dict()
    _dict['mode_file_triplet'] = mode_file_triplet
    _dict['reverse_graph'] = reverse_graph
    _dict['entity2id'] = entity2id
    _dict['relation2id'] = relation2id
    _dict['relation2text'] = relation2text
    _dict['same_label_dict'] = same_label_dict
    examples = ToDataset(_dict,args=args)
    
    return examples