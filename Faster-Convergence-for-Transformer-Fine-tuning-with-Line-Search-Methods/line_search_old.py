import torch
import torch.nn as nn
import torch.optim as optim
from sls.adam_sls import AdamSLS

class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolutionLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adjacency_matrix):
        return torch.matmul(adjacency_matrix, self.linear(x))

class GraphNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphNeuralNetwork, self).__init__()
        self.gc1 = GraphConvolutionLayer(input_dim, hidden_dim)
        self.gc2 = GraphConvolutionLayer(hidden_dim, output_dim)

    def forward(self, x, adjacency_matrix):
        x = torch.relu(self.gc1(x, adjacency_matrix))
        x = self.gc2(x, adjacency_matrix)
        return x

input_dim = 10
hidden_dim = 20
output_dim = 1
adjacency_matrix = torch.rand((input_dim, input_dim))
features = torch.rand((input_dim, input_dim))
target = torch.rand((input_dim, output_dim))

model = GraphNeuralNetwork(input_dim, hidden_dim, output_dim)

# optimizer
optimizer = AdamSLS([list(model.parameters())], smooth = True, c = 0.5) #optim.Adam(model.parameters(), lr=0.01)

def train(model, optimizer, features, adjacency_matrix, target):
    model.train()
    optimizer.zero_grad()

    # Encapsulate forward pass within a closure function
    closure = lambda: torch.mean((model(features, adjacency_matrix) - target)**2)

    loss = optimizer.step(closure=closure)
    return loss.item()

def evaluate(model, features, adjacency_matrix, target):
    model.eval()
    with torch.no_grad():
        output = model(features, adjacency_matrix)
        loss = torch.mean((output - target) ** 2)
    return output, loss.item()

#train
num_epochs = 100
for epoch in range(num_epochs):
    train_loss = train(model, optimizer, features, adjacency_matrix, target)
    output, eval_loss = evaluate(model, features, adjacency_matrix, target)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}")

print("Example Output:", output)

#Epoch 100/100, Train Loss: 0.0908, Eval Loss: 0.0909
