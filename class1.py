import torch
import torch.nn as nn


class UserScorer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)  # Wᵤ and bᵤ
    

    def forward(self, y_i):
        return torch.tanh(self.fc(y_i))  # ỹᵢ