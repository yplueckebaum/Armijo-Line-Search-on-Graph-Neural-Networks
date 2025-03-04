import torch
import torch.nn as nn

# random chatgpt model havent tested
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate RNN
        out, _ = self.rnn(x, h0)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


# Example usage:
input_size = 10
hidden_size = 20
output_size = 2
seq_length = 5
batch_size = 3

# Create dummy input
input_data = torch.randn(batch_size, seq_length, input_size)

# Create the RNN model
model = SimpleRNN(input_size, hidden_size, output_size)

# Forward pass
output = model(input_data)
print("Output shape:", output.shape)
