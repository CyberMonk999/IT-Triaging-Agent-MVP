src/qnetwork.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims=(128, 64)):
        """
        Q-network for ticket routing using embeddings as input.
        
        Args:
            input_dim (int): Dimension of input embeddings
            output_dim (int): Number of teams / actions
            hidden_dims (tuple): Sizes of hidden layers
        """
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
