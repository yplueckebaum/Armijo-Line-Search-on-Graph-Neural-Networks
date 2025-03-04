import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sls.adam_sls import AdamSLS
from sls.SaLSA import SaLSA

# Result dictionary
result_dict = {
    "c_value": -1,
    "train_loss": [],
    "val_loss": [],
    "learning_rate": [],
    "test_loss": [],
    "optimizer": "",
    "title": ""
}

# seed
random_seed = 42
torch.manual_seed(random_seed)

# Load ZINC dataset
dataset = ZINC(root='/tmp/ZINC', subset=True).shuffle()
dataset_length = len(dataset)
print(dataset_length)

# Training, validation, and test datasets

train_dataset = dataset[:dataset_length//2]
val_dataset = dataset[dataset_length//2 +1 :dataset_length*3//4]
test_dataset = dataset[dataset_length*3//4 + 1:]

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Def GNN class
# removed custom GCN layer and used pytorch one
class GraphNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphNeuralNetwork, self).__init__()
        self.gc1 = GCNConv(input_dim, hidden_dim)
        self.gc2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        x = torch.relu(self.gc1(x, edge_index))
        x = torch.relu(self.gc2(x, edge_index))
        x = global_mean_pool(x, batch)  # Pooling
        x = self.fc(x)
        return x


# Init model
input_dim = dataset.num_node_features
hidden_dim = 64
output_dim = 1
model = GraphNeuralNetwork(input_dim, hidden_dim, output_dim)


# set optimizer based on choice
def get_optimizer(optimizer_name, model, c=0.5):
    if optimizer_name == 'AdamSLS':
        result_dict["optimizer"] = "AdamSLS"
        result_dict["title"] = f"AdamSLS (c_value={c})"
        return AdamSLS([list(model.parameters())], smooth=True, c=c)
    elif optimizer_name == 'SaLSA':
        result_dict["optimizer"] = "SaLSA"
        result_dict["title"] = f"SaLSA (c_value={c})"
        return SaLSA(model.parameters(),c=c)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


# train step that changed based on optimizer choice
def train_step(model, optimizer, data, optimizer_choice):
    model.train()
    optimizer.zero_grad()

    data.x = data.x.float()
    data.edge_index = data.edge_index.long()

    def closure(backwards=False):
        output = model(data.x, data.edge_index, data.batch).squeeze()
        loss = F.mse_loss(output, data.y)
        if backwards:
            loss.backward()
        return loss

    if optimizer_choice == 'AdamSLS':
        loss = optimizer.step(closure=closure)
        result_dict["learning_rate"].append(optimizer.state["step_sizes"][0])
    elif optimizer_choice == 'SaLSA':
        loss = optimizer.step(closure=closure)
        result_dict["learning_rate"].append(optimizer.step_size)

    else:
        raise ValueError("Where optimizer?")

    return loss.item()


# calls train steps
def train_model(model, optimizer, train_loader, val_loader, optimizer_choice, num_epochs=10):
    for epoch in range(num_epochs):
        train_losses = []
        for data in train_loader:
            data.x = data.x.float()
            data.edge_index = data.edge_index.float()
            train_loss = train_step(model, optimizer, data, optimizer_choice=optimizer_choice)
            train_losses.append(train_loss)
        train_loss = sum(train_losses) / len(train_losses)

        val_loss = evaluate(model, val_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        result_dict["train_loss"].append(train_loss)
        result_dict["val_loss"].append(val_loss)


# eval functino
def evaluate(model, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data.x = data.x.float()
            data.edge_index = data.edge_index.long()
            output = model(data.x, data.edge_index, data.batch)
            loss = F.mse_loss(output.squeeze(), data.y)
            total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def save_results(model, c, test_loader, result_dict, optimizer_name):
    torch.save(model, f'../savedModels/gnn_{optimizer_name}.pth')
    model = torch.load(f'../savedModels/gnn_{optimizer_name}.pth')
    test_loss = evaluate(model, test_loader)
    result_dict["test_loss"].append(test_loss)

    filename = f"../results/{optimizer_name}_c={c}.pickle"
    with open(filename, 'wb') as handle:
        pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(filename, 'rb') as handle:
        b = pickle.load(handle)

    print(result_dict == b)  # should return True if saving and loading works correctly


optimizer_name = "AdamSLS"  #"SaLSA" / "AdamSLS"
c_value = 0.7
result_dict["c_value"] = c_value
optimizer = get_optimizer(optimizer_name, model, c=c_value)

# Trains model
train_model(model, optimizer, train_loader, val_loader, optimizer_name, num_epochs=200)

print(os.getcwd())  # pycharm likes to prank me
torch.save(model, '../savedModels/gnn_line_search.pth')
model = torch.load('../savedModels/gnn_line_search.pth')
test_loss = evaluate(model, test_loader)
print(f"Test Loss: {test_loss:.4f}")
result_dict["test_loss"].append(test_loss)

# save results

filename = f"../results/{optimizer_name} c=" + str(c_value) + ".pickle"
with open(filename, 'wb') as handle:
    pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(filename, 'rb') as handle:
    b = pickle.load(handle)
