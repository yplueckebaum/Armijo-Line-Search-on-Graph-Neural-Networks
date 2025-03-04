import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sls.adam_sls import AdamSLS

#set seed
random_seed = 42
torch.manual_seed(random_seed)
# Load ZINC dataset
dataset = ZINC(root='/tmp/ZINC', subset=True)
print(len(dataset))

# training, validation, and test set
train_dataset = dataset[:7500]
val_dataset = dataset[7500:9000]
test_dataset = dataset[9000:]

# data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# GraphConvLayer
class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolutionLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adjacency_matrix):
        x = x.float()
        adjacency_matrix = adjacency_matrix.float()
        return torch.matmul(adjacency_matrix, self.linear(x))

class GraphNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphNeuralNetwork, self).__init__()
        self.gc1 = GCNConv(input_dim, hidden_dim)
        self.gc2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        x = torch.relu(self.gc1(x, edge_index))
        x = torch.relu(self.gc2(x, edge_index))
        x = global_mean_pool(x, batch)  # pooling
        x = self.fc(x)
        return x
# Init model
input_dim = dataset.num_node_features
hidden_dim = 64
output_dim = 1

model = GraphNeuralNetwork(input_dim, hidden_dim, output_dim)

# optimizer
optimizer = AdamSLS([list(model.parameters())], smooth = True, c = 0.5)

def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()

    data.x = data.x.float()
    data.edge_index = data.edge_index.long()
    #output = model(data.x, data.edge_index, data.batch)
    #add sls optimizer
    closure = lambda: torch.mean((model(data.x, data.edge_index, data.batch).squeeze() - data.y)**2)  # mae would be better than mse for the dataset? todo
    loss = optimizer.step(closure=closure)
    # loss.backward() Not needed because of the above line i think
    return loss.item()

def evaluate(model, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data.x = data.x.float()
            data.edge_index = data.edge_index.long()
            output = model(data.x, data.edge_index,data.batch)
            loss = F.mse_loss(output.squeeze(), data.y)
            total_loss += loss.item()
    return total_loss / max(len(loader), 1)  #Avoid div by zero

# Training
num_epochs = 10 # todo 100
for epoch in range(num_epochs):
    train_losses = []
    for data in train_loader:
        data.x = data.x.float()  # Ensure data.x is float
        data.edge_index = data.edge_index.float()  # Ensure edge_index is float
        train_loss = train(model, optimizer, data)
        train_losses.append(train_loss)
    train_loss = sum(train_losses) / len(train_losses)

    val_loss = evaluate(model, val_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}")

# Save model
print(os.getcwd())
torch.save(model,'../savedModels/gnn_line_search.pth')
# Evaluate
model = torch.load('../savedModels/gnn_line_search.pth')
test_loss = evaluate(model, test_loader)
print(f"Test Loss: {test_loss:.4f}")


