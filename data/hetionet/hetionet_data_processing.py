import pandas as pd
import json
import torch
from tqdm import tqdm
import random

#step 1
hetionet_path = './raw_data/hetionet-v1.0.json'
with open(hetionet_path,'r') as file:
    hetionet = json.load(file)
nodes_path = './raw_data/hetionet-v1.0-nodes.tsv'
nodes = pd.read_csv(nodes_path, sep='\t')

allow_node_kind = ['Compound','Disease','Gene']

node2entity = dict()

node_num = len(nodes)
for i in range(node_num):
    if(nodes.iloc[i]['kind'] in allow_node_kind):
        key = nodes.iloc[i]['id']
        name = nodes.iloc[i]['name']
        node2entity[key] = name

all_trplets = []
relation_set = set()

train_triple = []
val_triple = []
test_triple = []

for i in hetionet['edges']:
    if(i['source_id'][0] in allow_node_kind and i['target_id'][0] in allow_node_kind):
        rand  = random.random()
        head = node2entity[i['source_id'][0]+'::'+str(i['source_id'][1])]
        relation = i['kind']
        tail = node2entity[i['target_id'][0]+'::'+str(i['target_id'][1])]
        direction = 'B' if(i['direction']== 'both') else 'F'
        relation_set.add(relation)
        all_trplets.append('\t'.join([head,relation,tail,direction])+'\n')
        if(rand<0.8):
            train_triple.append('\t'.join([head,relation,tail])+'\n')
        elif(rand<0.9):
            val_triple.append('\t'.join([head,relation,tail])+'\n')
        else:
            test_triple.append('\t'.join([head,relation,tail])+'\n')

relation_set = list(relation_set)
relation_set.sort()
real_set_df = pd.DataFrame({'name':relation_set})
real_set_df['idx'] = real_set_df.index
real_set_df.to_csv(f'./pro_data/relation2id.txt', header=None, sep='\t', index=False)

with open('./pro_data/hetionet_data.txt','w') as file:
    for i in tqdm(all_trplets):
        file.write(i)

with open('./pro_data/train.txt','w') as file:
    for i in train_triple:
        file.write(i)
with open('./pro_data/val.txt','w') as file:
    for i in val_triple:
        file.write(i)
with open('./pro_data/test.txt','w') as file:
    for i in test_triple:
        file.write(i)


#step 2
all_triple = pd.read_csv('./pro_data/hetionet_data.txt',sep='\t',header=None, names=['h', 'r', 't','d']) 
entity = set(all_triple['h']).union(all_triple['t'])
entity = list(entity)
entity.sort()

entity = pd.DataFrame({'name':entity})
entity['idx'] = entity.index
entity.to_csv(f'./pro_data/entity2id.txt', header=None, sep='\t', index=False)

#step 3
file_path = './pro_data/hetionet_data.txt'
all_triplet = []
with open(file_path) as file:
    file = file.read().splitlines()
    for i in file:
        all_triplet.append(tuple(i.split('\t')))

reverse_graph = {}
for triple in all_triplet:
    head = triple[0]
    rela = triple[1]
    tail = triple[2]
    direction = triple[3]
    if(direction == 'B'):
        if(head not in reverse_graph.keys()):
            reverse_graph[head] = {}
        if(tail not in reverse_graph.keys()):
            reverse_graph[tail] = {}
        reverse_graph[head][tail] = rela
        reverse_graph[tail][head] = rela
    else:
        if(tail not in reverse_graph.keys()):
            reverse_graph[tail] = {}
        reverse_graph[tail][head] = rela

with open('./pro_data/reverse_graph.json','w') as file:
    file.write(json.dumps(reverse_graph))

#step 4
entity2id = dict()
with open('./pro_data/entity2id.txt') as file:
    file = file.read().splitlines()
    for i in file:
        temp = i.split('\t')
        entity2id[temp[0]] = int(temp[1])

relation2id = dict()
with open('./pro_data/relation2id.txt') as file:
    file = file.read().splitlines()
    for i in file:
        temp = i.split('\t')
        relation2id[temp[0]] = int(temp[1])

for mode in ['train','val','test']:
    file_name = mode + '.txt'
    all_triple = []
    with open('./pro_data/'+file_name,'r') as file:
        file = file.read().splitlines()
        for i in file:
            triple = tuple([ entity2id[i.split('\t')[0]], relation2id[i.split('\t')[1]], entity2id[i.split('\t')[2]] ])
            all_triple.append(triple)
    
    target_dict = dict()
    for i in all_triple:
        if(i[0] not in target_dict.keys()):
            target_dict[i[0]] = {}

        if(i[1] not in target_dict[i[0]].keys()):
            target_dict[i[0]][i[1]] = list()
            
        target_dict[i[0]][i[1]].append(i[2])
        
    with open('./pro_data/'+mode+'_label.json','w') as file:
        json.dump(target_dict,file)     
