import rumorProject as RP
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
torch.manual_seed(RP.config.SEED_TORCH)


#format input : x=(nbre_engagement, temps entre chaque engagement,source = x_u ,x_t=caractéristiques d'un texte)
class CaptureModule(nn.Module):
    # 3 layers : a linear NN, then a RNN (LSTM) and the last one also a linear NN
    def __init__(self, dim_x_t, dim_embedding_wa, dim_hidden, dim_v_j):
        super(CaptureModule, self).__init__()
        self.embedding_wa = nn.Linear(dim_x_t, dim_embedding_wa)
        self.ln1 = nn.LayerNorm(dim_embedding_wa)  # Replace BatchNorm with LayerNorm
        self.dropout_wa = nn.Dropout(p=0.2)
        self.rnn = nn.LSTM(dim_embedding_wa, dim_hidden, batch_first=True)
        self.ln2 = nn.LayerNorm(dim_hidden)  # Replace BatchNorm with LayerNorm
        self.fc_wr = nn.Linear(dim_hidden, dim_v_j)
        self.dropout_wr = nn.Dropout(p=0.2)

    def forward(self, x_t, lengths):  # xt = (η, ∆t, xu , xτ) are the features of the article
        batch_size, seq_len, feat_dim = x_t.size()

        #process everything in the batch
        flat = x_t.view(-1, feat_dim) # the batch aren't sperated in different dimension, they just follow each other
        emb_flat = self.embedding_wa(flat) 
        emb_flat = self.ln1(emb_flat)
        emb_flat = torch.tanh(emb_flat)
        x_tt = emb_flat.view(batch_size, seq_len, -1)        # (batch, seq_len, dim_emb)
        x_tt = self.dropout_wa(x_tt)
        #tell the LSTM that the sequence might be padded even with non aanymore zero values
        packed = pack_padded_sequence(
            x_tt,
            lengths.cpu().long(),
            batch_first=True, #to keep (batch_size, seq_len, features) format
            enforce_sorted=False #
        ) 
        #the packed will give only the non-padded part of the sequence
        _, (h_T, _) = self.rnn(packed)  # second layer : LSTM #est  ce qu'on est sur que h_n bien le dernier état caché? oui
        h_T = h_T.squeeze(0)  # remove the first dimension (1,batch_size, hidden_size) ->(batch_size, hidden_size)
        
        # Apply LayerNorm before final linear layer
        h_T = self.ln2(h_T)
        h_T = self.dropout_wr(h_T)
        v_j = torch.tanh(self.fc_wr(h_T))  # removed tanh for identity
        return v_j


class ScoreModule(nn.Module):
    def __init__(self, dim_y_i, dim_embedding_wu):
        super(ScoreModule, self).__init__()
        #register buffer move with the model and not trained and saved with the model
        self.user_fc = nn.Linear(dim_y_i, dim_embedding_wu)
        self.ln = nn.LayerNorm(dim_embedding_wu)  # Replace BatchNorm with LayerNorm
        self.score_fc = nn.Linear(dim_embedding_wu, 1)
        
        # we put y_is instead of y_i bc element in the batch is composed
        #  of multiple y_i  corresponding to the engaged users
    def forward(self, y_is): 
        h = self.user_fc(y_is)
        h = self.ln(h)  # Apply LayerNorm
        y_i_ts = torch.tanh(h)
        s_is = torch.sigmoid(self.score_fc(y_i_ts))
        return s_is, y_i_ts
    
    def get_wu(self):
        return self.user_fc.weight


class IntegrateModule(nn.Module):
    def __init__(self, dim_v_j, user_scores_dim=1, alpha=1):
        super(IntegrateModule, self).__init__()
        self.alpha = alpha # learnable scale for user score
        self.fc = nn.Linear(dim_v_j + user_scores_dim, 1)

    def forward(self, v_j, p_j):
        """
        v_j : Tensor of shape (B, dim_v_j)
        p_j : Tensor of shape (B, 1) or None
        """
        if p_j is None:            # simple‑CSI branch (no user scores)
            c_j = v_j
        else:                      # scale the average user score to avoid tiny gradients
            p_j = self.alpha * p_j
            c_j = torch.cat((v_j, p_j), dim=1)
        return torch.sigmoid(self.fc(c_j))


class CSI_model(nn.Module):
    def __init__(
            self, 
            dim_x_t, 
            dim_embedding_wa, 
            dim_hidden, 
            dim_v_j,
            dim_y_i, 
            dim_embedding_wu,
            alpha=1
            ):
        super(CSI_model, self).__init__()
        self.capture_module = CaptureModule(dim_x_t, dim_embedding_wa, dim_hidden, dim_v_j)
        self.score_module = ScoreModule(dim_y_i, dim_embedding_wu)
        self.integrate_module = IntegrateModule(dim_v_j, alpha=alpha)
    """
       #y_is and s_is instead of y_i and s_i because for all enguaged users (no mask)
    def forward(self, x_t, lengths, y_is):
        # article score
        v_j = self.capture_module(x_t, lengths)

        s_is, y_i_ts = self.score_module(y_is) 
        p_j = s_is.mean()  # Average over the engaged users

        prediction = self.integrate_module(v_j, p_j)
        prediction = prediction.squeeze(-1)

        return prediction, y_i_ts, v_j"""
    
    def forward(self, x_t, lengths, y_flat, src_idx):
        B   = x_t.size(0)
        v_j = self.capture_module(x_t, lengths)          # (B, dim_v)

        # Compute the scores for every engaged user
        s_is, _ = self.score_module(y_flat)          # (N, 1)

        # --- robust aggregation ---  
        # Avoid a trailing singleton dimension in the destination tensor, which
        # triggers a rank‑mismatch assertion on Apple MPS.  
        s_is_flat = s_is.squeeze(1)                  # (N,)

        # Sum of user scores per article
        sums   = torch.zeros(B, device=s_is.device).scatter_add_(
                    0,              # aggregate along article dimension
                    src_idx,        # (N,)
                    s_is_flat       # (N,)
                 )

        # Count of engaged users per article
        counts = torch.zeros(B, device=s_is.device).scatter_add_(
                    0,
                    src_idx,
                    torch.ones_like(s_is_flat)
                 )

        # Average user score for each article
        p_j = (sums / counts.clamp_min(1)).unsqueeze(1)   # (B, 1)

        out = self.integrate_module(v_j, p_j)            # (B,1)
        return out.squeeze(1), s_is, v_j
        
    def get_wu(self):
        return self.score_module.get_wu()
    

class Simple_CSI_model(nn.Module):
    def __init__(
            self, 
            dim_x_t, 
            dim_embedding_wa, 
            dim_hidden, 
            dim_v_j,
            dim_y_i, 
            dim_embedding_wu
            ):
        super(Simple_CSI_model, self).__init__()
        self.capture_module = CaptureModule(dim_x_t, dim_embedding_wa, dim_hidden, dim_v_j)
        self.integrate_module = IntegrateModule(dim_v_j, user_scores_dim=0)  # No user scores in this model

    def forward(self, x_t, lengths, y_flat, src_idx):
        v_j = self.capture_module(x_t, lengths)
        out = self.integrate_module(v_j, None)  # Use a constant p_j of 0.5
        return out.squeeze(1), None, v_j
    

    