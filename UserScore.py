import torch
import torch.nn as nn


class UserScorer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        """
        input_dim  : dimension de y_i (features concaténées)
        hidden_dim : dimension de la représentation ỹ_i (input de la seconde couche)
        output_dim : 1 pour un score par utilisateur
        """
        super().__init__()
        # 1) couche pour calculer la représentation compacte ỹ_i = tanh(W_u*y_i + b_u)
        self.fc_user = nn.Linear(input_dim, hidden_dim)
        # 2) couche pour produire le score s_i = σ(w_sT*ỹ_i + b_s)
        self.fc_score = nn.Linear(hidden_dim, output_dim)

    def forward(self, y_i):
        # y_i : tensor [batch_size, input_dim]
        # 1) représentation utilisateur
        y_tilde = torch.tanh(self.fc_user(y_i))            # [batch_size, hidden_dim]
        # 2) score utilisateur
        s_i = torch.sigmoid(self.fc_score(y_tilde))        # [batch_size, 1]
        return y_tilde, s_i