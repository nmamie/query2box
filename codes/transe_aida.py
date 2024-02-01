import argparse
import os.path as osp

import torch
import torch.optim as optim

import pickle

from rdflib import Graph

from torch_geometric.nn import ComplEx, DistMult, TransE

# model mapping
model_map = {
    'transe': TransE,
    'complex': ComplEx,
    'distmult': DistMult
}

# parse the runtime arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=model_map.keys(), type=str.lower,
                    required=True)
parser.add_argument('--epochs', type=int, default=100)
args = parser.parse_args()

# detect GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using: {device}')

# load AIDA graph from pickle
train_data = pickle.load(open('data/aida/test0_1_train_graph.pkl', 'rb'))
val_data = pickle.load(open('data/aida/test0_1_val_graph.pkl', 'rb'))
test_data = pickle.load(open('data/aida/test0_1_test_graph.pkl', 'rb'))
# print(f'Loaded graph with {graph.num_nodes} nodes and {graph.num_edges} edges.')
print(f'Loaded graph with {train_data.num_nodes} nodes and {train_data.num_edges} edges.')

# load to device
train_data = train_data.to(device)
val_data = val_data.to(device)
test_data = test_data.to(device)

# define the model
model_arg_map = {'rotate': {'margin': 9.0}}
model = model_map[args.model](
    num_nodes=train_data.num_nodes,
    num_relations=train_data.num_edge_types,
    hidden_channels=50,
    **model_arg_map.get(args.model, {}),
).to(device)

print(train_data.num_nodes)
print(train_data.num_edge_types)
print(train_data.edge_type)
print(len(train_data.edge_type))
print(len(train_data.edge_index[0]))

# load the data into the model
loader = model.loader(
    head_index=train_data.edge_index[0],
    rel_type=train_data.edge_type,
    tail_index=train_data.edge_index[1],
    batch_size=5000,
    shuffle=True,
)
print(f'Loaded {len(loader)} batches.')

# training optimizer
optimizer_map = {
    'transe': optim.Adam(model.parameters(), lr=0.01),
    'complex': optim.Adagrad(model.parameters(), lr=0.001, weight_decay=1e-6),
    'distmult': optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-6)
    }
optimizer = optimizer_map[args.model]

print("Starting training...")
# define the training loop
def train():
    model.train()
    total_loss = total_examples = 0
    for head_index, rel_type, tail_index in loader:
        optimizer.zero_grad()
        loss = model.loss(head_index, rel_type, tail_index)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * head_index.numel()
        total_examples += head_index.numel()
    return total_loss / total_examples

# define the testing loop
@torch.no_grad()
def test(data):
    model.eval()
    return model.test(
        head_index=data.edge_index[0],
        rel_type=data.edge_type,
        tail_index=data.edge_index[1],
        batch_size=50000,
        k=10,
    )

# run the training
for epoch in range(1, 1 + args.epochs):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    if epoch % 25 == 0:
        rank, hits = test(val_data)
        print(f'Epoch: {epoch:03d}, Val Mean Rank: {rank:.2f}, '
              f'Val Hits@10: {hits:.4f}')

# evaluate the performance
rank, hits_at_10 = test(test_data)
print(f'Test Mean Rank: {rank:.2f}, '
      f'Test Hits@10: {hits_at_10:.4f}')

# save the model
torch.save(model.state_dict(), f'models/{args.model}.pt')

# save the embeddings
torch.save(model.node_emb.weight.data, f'models/{args.model}_emb.pt')
torch.save(model.rel_emb.weight.data, f'models/{args.model}_rel.pt')