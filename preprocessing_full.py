import pickle
import torch
from torch_geometric.data import Data
import numpy as np
from rdflib import Graph, URIRef, Literal, BNode

def rdf_to_edge_index(triples_list, node2ind):
    """
    Converts a list of RDF triples into an edge index tensor for training TransE.
    """
    # create edge index tensor
    edge_index = torch.zeros((2, len(triples_list)), dtype=torch.long)
    # iterate over triples
    for i, (s, p, o) in enumerate(triples_list):
        # get index of subject and object
        s_ind = node2ind[s]
        o_ind = node2ind[o]
        # add to edge index tensor
        edge_index[0, i] = s_ind
        edge_index[1, i] = o_ind
    return edge_index
    
# load graph from ttl
g = Graph()
g.parse('data/papers_v4/all.ttl', format='ttl')

# save graph to pickle
pickle.dump(g, open('data/aida_full/full_graph.pkl', 'wb'))

# # load graph from pickle
# g = pickle.load(open('data/aida/full_graph.pkl', 'rb'))

# get all triples
triples = []
for s, p, o in g:
    triples.append((s, p, o))
    
# get all entities
ents = set()
for s, p, o in triples:
    ents.add(s)
    ents.add(o)
    
# get all relations
rels = set()
for s, p, o in triples:
    rels.add(p)
    
# number of entities and relations
g.num_edges = len(triples)
g.num_nodes = len(ents)

# Create a random permutation of indices
rand_graph_index = torch.randperm(len(triples))

# Convert the indices for each split to a list
val_indices = rand_graph_index[:int(len(triples) * 0.001)].tolist()
test_indices = rand_graph_index[int(len(triples) * 0.001):int(len(triples) * 0.002)].tolist()
train_indices = rand_graph_index[int(len(triples) * 0.002):int(len(triples) * 0.01)].tolist()

# Use the list of indices to create data splits
val_data = [triples[i] for i in val_indices]
test_data = [triples[i] for i in test_indices]
train_data = [triples[i] for i in train_indices]

print(len(train_data))

# create index for rel types
rel2ind = {}
ind2rel = {}
for i, rel in enumerate(rels):
    rel2ind[rel] = i
    ind2rel[i] = rel
print(rel2ind)

# create tensor of rel indices for each triple
rel_indices = []
for s, p, o in triples:
    rel_indices.append(rel2ind[p])
rel_indices = torch.tensor(rel_indices)
print(rel_indices)
print(len(rel_indices))

# same for train, val, test
rel_indices_train = []
for s, p, o in train_data:
    rel_indices_train.append(rel2ind[p])
rel_indices_train = torch.tensor(rel_indices_train)
print(rel_indices_train)
print(len(rel_indices_train))

rel_indices_val = []
for s, p, o in val_data:
    rel_indices_val.append(rel2ind[p])
rel_indices_val = torch.tensor(rel_indices_val)
print(rel_indices_val)
print(len(rel_indices_val))

rel_indices_test = []
for s, p, o in test_data:
    rel_indices_test.append(rel2ind[p])
rel_indices_test = torch.tensor(rel_indices_test)
print(rel_indices_test)
print(len(rel_indices_test))

# save rel2ind and ind2rel
pickle.dump(rel2ind, open('data/aida_full/rel2ind.pkl', 'wb'))
pickle.dump(ind2rel, open('data/aida_full/ind2rel.pkl', 'wb'))

# also for train, val, test
pickle.dump(rel_indices_train, open('data/aida_full/rel_indices_train.pkl', 'wb'))
pickle.dump(rel_indices_val, open('data/aida_full/rel_indices_val.pkl', 'wb'))
pickle.dump(rel_indices_test, open('data/aida_full/rel_indices_test.pkl', 'wb'))

# create node2ind and ind2node
node2ind = {}
ind2node = {}
for i, node in enumerate(ents):
    node2ind[node] = i
    ind2node[i] = node
print(node2ind)
print(ind2node)

# also for train, val, test
node2ind_train = {}
ind2node_train = {}
for i, (s, p, o) in enumerate(train_data):
    node2ind_train[s] = i
    ind2node_train[i] = s
    node2ind_train[o] = i
    ind2node_train[i] = o
print(node2ind_train)

node2ind_val = {}
ind2node_val = {}
for i, (s, p, o) in enumerate(val_data):
    node2ind_val[s] = i
    ind2node_val[i] = s
    node2ind_val[o] = i
    ind2node_val[i] = o
print(node2ind_val)

node2ind_test = {}
ind2node_test = {}
for i, (s, p, o) in enumerate(test_data):
    node2ind_test[s] = i
    ind2node_test[i] = s
    node2ind_test[o] = i
    ind2node_test[i] = o
print(node2ind_test)

# save node2ind and ind2node
pickle.dump(node2ind, open('data/aida_full/node2ind.pkl', 'wb'))
pickle.dump(ind2node, open('data/aida_full/ind2node.pkl', 'wb'))

# also for train, val, test
pickle.dump(node2ind_train, open('data/aida_full/node2ind_train.pkl', 'wb'))
pickle.dump(ind2node_train, open('data/aida_full/ind2node_train.pkl', 'wb'))
pickle.dump(node2ind_val, open('data/aida_full/node2ind_val.pkl', 'wb'))
pickle.dump(ind2node_val, open('data/aida_full/ind2node_val.pkl', 'wb'))
pickle.dump(node2ind_test, open('data/aida_full/node2ind_test.pkl', 'wb'))
pickle.dump(ind2node_test, open('data/aida_full/ind2node_test.pkl', 'wb'))

# calculate num_nodes for train, val and test
num_nodes_train = 0
for s, p, o in train_data:
    num_nodes_train = max(num_nodes_train, node2ind_train[s], node2ind_train[o])
num_nodes_train += 1
print(num_nodes_train)

num_nodes_val = 0
for s, p, o in val_data:
    num_nodes_val = max(num_nodes_val, node2ind_val[s], node2ind_val[o])
num_nodes_val += 1
print(num_nodes_val)

num_nodes_test = 0
for s, p, o in test_data:
    num_nodes_test = max(num_nodes_test, node2ind_test[s], node2ind_test[o])
num_nodes_test += 1
print(num_nodes_test)

# create torch geometric data object for train, val, test
# convert triples to edge index
train_edge_index = rdf_to_edge_index(train_data, node2ind_train)
val_edge_index = rdf_to_edge_index(val_data, node2ind_val)
test_edge_index = rdf_to_edge_index(test_data, node2ind_test)

# graph contains all triples and the split index
train_graph = Data()
train_graph.num_nodes = num_nodes_train
train_graph.num_edges = len(train_data)
train_graph.edge_index = train_edge_index
train_graph.num_edge_types = len(rels)
train_graph.edge_type = rel_indices_train
train_graph.edge_attr = torch.ones(len(train_data), 1)
train_graph.train_mask = torch.zeros(len(train_data), dtype=torch.bool)
train_graph.train_mask[:] = True

val_graph = Data()
val_graph.num_nodes = num_nodes_val
val_graph.num_edges = len(val_data)
val_graph.edge_index = val_edge_index
val_graph.num_edge_types = len(rels)
val_graph.edge_type = rel_indices_val
val_graph.edge_attr = torch.ones(len(val_data), 1)
val_graph.val_mask = torch.zeros(len(val_data), dtype=torch.bool)
val_graph.val_mask[:] = True

test_graph = Data()
test_graph.num_nodes = num_nodes_test
test_graph.num_edges = len(test_data)
test_graph.edge_index = test_edge_index
test_graph.num_edge_types = len(rels)
test_graph.edge_type = rel_indices_test
test_graph.edge_attr = torch.ones(len(test_data), 1)
test_graph.test_mask = torch.zeros(len(test_data), dtype=torch.bool)
test_graph.test_mask[:] = True

# save graphs to pickle
pickle.dump(train_graph, open('data/aida_full/test0_1_train_graph.pkl', 'wb'))
pickle.dump(val_graph, open('data/aida_full/test0_1_val_graph.pkl', 'wb'))
pickle.dump(test_graph, open('data/aida_full/test0_1_test_graph.pkl', 'wb'))

# dump final graph to pickle
pickle.dump(g, open('data/aida_full/test0_1_graph.pkl', 'wb'))