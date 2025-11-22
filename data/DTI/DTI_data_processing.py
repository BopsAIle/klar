import pandas as pd
import random
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import json
#step 1
path = './raw_data/DTINet_total_relationwithname_0129.csv'
data = pd.read_csv(path)

data.insert(data.shape[-1],'d','B')

all_trplets = []
relation_set = set(data.loc[:,'relation'])
train_triple = []
val_triple = []
test_triple = []
for i in range(len(data)):
    rand  = random.random()
    head = data.loc[i,'entity1']
    relation = data.loc[i,'relation']
    tail = data.loc[i,'entity2']
    direction = data.loc[i,'d']
    all_trplets.append('\t'.join([head,relation,tail,direction])+'\n')
    if(rand<0.8):
        train_triple.append('\t'.join([head,relation,tail])+'\n')
    elif(rand<0.9):
        val_triple.append('\t'.join([head,relation,tail])+'\n')
    else:
        test_triple.append('\t'.join([head,relation,tail])+'\n')

entity = set(data.loc[:,'entity1'])
entity = entity.union(data.loc[:,'entity2'])
entity = list(entity)
entity.sort()
entity = pd.DataFrame({'name':entity})
entity['idx'] = entity.index
entity.to_csv(f'./pro_data/entity2id.txt', header=None, sep='\t', index=False)

relation = set(data.loc[:,'relation'])
relation = list(relation)
relation.sort()
relation = pd.DataFrame({'name':relation})
relation['idx'] = relation.index
relation.to_csv(f'./pro_data/relation2id.txt', header=None, sep='\t', index=False)

with open('./pro_data/DTI_data.txt','w') as file:
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
file_path = './pro_data/DTI_data.txt'
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

#step 3
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
