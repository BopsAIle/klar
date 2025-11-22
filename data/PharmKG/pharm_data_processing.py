import pandas as pd
import json
import torch
from tqdm import tqdm
#step 1
relation_set = set()

train_data = pd.read_csv('./raw_data/train.tsv',sep='\t',header=None) 
val_data = pd.read_csv('./raw_data/val.tsv',sep='\t',header=None) 
test_data = pd.read_csv('./raw_data/test.tsv',sep='\t',header=None)

for i in range(len(train_data)):
    relation_set.add(train_data.iat[i,1])
for i in range(len(val_data)):
    relation_set.add(val_data.iat[i,1])
for i in range(len(test_data)):
    relation_set.add(test_data.iat[i,1])

relation2text = dict()
with open('./raw_data/paper_relation2text.txt','r') as file:
    file = file.read().splitlines()
    for i in file:
        relation2text[i.split(',')[0]] = i.split(',')[1]

relation2text_set = set(relation2text.keys())
real_set = relation2text_set & relation_set
real_set = list(real_set)
real_set.sort()

real_set_df = pd.DataFrame({'name':real_set})
real_set_df['idx'] = real_set_df.index
real_set_df.to_csv(f'./pro_data/relation2id.txt', header=None, sep='\t', index=False)

real_relation2text = dict()
for i in real_set:
    real_relation2text[i] = relation2text[i]

with open('./pro_data/relation2text.txt','w') as file:
    file.write(json.dumps(real_relation2text))

train_triple = []
val_triple = []
test_triple = []
for i in range(len(train_data)):
    if(train_data.iat[i,1] in real_set):
        train_triple.append('\t'.join([train_data.iat[i,0],train_data.iat[i,1],train_data.iat[i,2]])+'\n')
for i in range(len(val_data)):
    if(val_data.iat[i,1] in real_set):
        val_triple.append('\t'.join([val_data.iat[i,0],val_data.iat[i,1],val_data.iat[i,2]])+'\n')
for i in range(len(test_data)):
    if(test_data.iat[i,1] in real_set):
        test_triple.append('\t'.join([test_data.iat[i,0],test_data.iat[i,1],test_data.iat[i,2]])+'\n')

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
train = pd.read_csv('./pro_data/train.txt',sep='\t',header=None, names=['h', 'r', 't']) 
valid = pd.read_csv('./pro_data/val.txt',sep='\t',header=None, names=['h', 'r', 't']) 
test = pd.read_csv('./pro_data/test.txt',sep='\t',header=None, names=['h', 'r', 't'])

entity = set(train['h']).union(train['t'])
entity = entity.union(valid['h']).union(valid['t'])
entity = entity.union(test['h']).union(test['t'])
entity = list(entity)
entity.sort()

entity = pd.DataFrame({'name':entity})
entity['idx'] = entity.index
entity.to_csv(f'./pro_data/entity2id.txt', header=None, sep='\t', index=False)

all_data = pd.concat([train,valid,test])
all_data.insert(all_data.shape[-1],'d','B')
all_data.to_csv(f'./pro_data/KG_data.txt', header=None, sep='\t', index=False)

#step 3
file_path = './pro_data/KG_data.txt'
all_triplet = []
with open(file_path) as f:
    file = f.read().splitlines()
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
