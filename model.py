import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


#format input : x=(nbre_engagement, temps entre chaque engagement,source = x_u ,x_t=caractéristiques d'un texte)
class CaptureModule(nn.Module):
    # 3 layers : a linear NN, then a RNN (LSTM) and the last one also a linear NN
    def __init__(self, dim_x_t, dim_embedding_wa, dim_hidden, dim_v_j):
        super(CaptureModule, self).__init__()
        self.embedding_wa = nn.Linear(dim_x_t, dim_embedding_wa)
        self.dropout_wa = nn.Dropout(p=0.5) # Increased dropout
        self.rnn = nn.LSTM(dim_embedding_wa, dim_hidden, batch_first=True)
        self.fc_wr = nn.Linear(dim_hidden, dim_v_j) 
        self.dropout_wr = nn.Dropout(p=0.5) # Increased dropout

    def forward(self, x_t, lengths):  # xt = (η, ∆t, xu , xτ) are the features of the article
        batch_size, seq_len = x_t.shape[0], x_t.shape[1]
        x_tt = torch.zeros(batch_size, seq_len, self.embedding_wa.out_features, 
                    device=x_t.device)
    
    # Process only valid timesteps for each sequence in the batch
        for i in range(batch_size):
            x_tt[i, :lengths[i]] = torch.tanh(self.embedding_wa(x_t[i, :lengths[i]]))
        
        x_tt = self.dropout_wa(x_tt)
        packed = pack_padded_sequence(
            x_tt,
            lengths.cpu().long(),
            batch_first=True, #to keep (batch_size, seq_len, features) format
            enforce_sorted=False #
        ) 
        _, (h_T, _) = self.rnn(packed)  # second layer : LSTM #est  ce qu'on est sur que h_n bien le dernier état caché?
        #print('shape',h_n.shape)
        h_T = h_T.squeeze(0)  # remove the first dimension (1,batch_size, hidden_size) ->(batch_size, hidden_size)
        h_T = self.dropout_wr(h_T)
        #print('after',h_n.shape)v_j = torch.tanh(self.fc_wr(h_T))
        v_j = torch.tanh(self.fc_wr(h_T))  # removed tanh for identity
        return v_j


class ScoreModule(nn.Module):
    def __init__(self, dim_y_i, dim_embedding_wu):
        super(ScoreModule, self).__init__()
        self.user_fc = nn.Linear(dim_y_i, dim_embedding_wu) #to get dim_y_it = dim_embedding_wu
        self.score_fc = nn.Linear(dim_embedding_wu, 1) # to get dim_s_i = 1
        
    def forward(self, y_i):

        y_i_t = torch.tanh(self.user_fc(y_i)) # ỹ_i = tanh(W_u*y_i + b_u)
        
        s_i = torch.sigmoid(self.score_fc(y_i_t)) # s_i = σ(w_sT*ỹ_i + b_s)
        
        return s_i, y_i_t
    
    def get_wu(self):
        return self.user_fc.weight


class IntegrateModule(nn.Module):
    def __init__(self, dim_v_j, user_scores_dim=1):
        super(IntegrateModule, self).__init__()
        self.fc = nn.Linear(dim_v_j + user_scores_dim, 1)
    
    def forward(self, v_j, s_i, m_j):
        # user_article_mask is already the correct mask for this batch's articles
        # No need to index it further
        
        # Rest of the code remains the same
        #masked_scores = s_i * batch_mask
        #sum_scores = torch.sum(masked_scores, dim=1)
        #because user_article mask is boolean, we can use it to mask the scores
        #put 0 where the mask is False (~user_article_mask = where the mask is False)
        sum_scores   = torch.sum(s_i.masked_fill(~m_j, 0.0),dim=1)
        count_users = torch.sum(m_j, dim=1).float().clamp_(min=1.0)
        p_j = (sum_scores / count_users).unsqueeze(1)
        c_j = torch.cat((v_j, p_j), dim=1)
        prediction = torch.sigmoid(self.fc(c_j))
        return prediction

class CSI_model(nn.Module):
    def __init__(
            self, 
            dim_x_t, 
            dim_embedding_wa, 
            dim_hidden, 
            dim_v_j,
            dim_y_i, 
            dim_embedding_wu
            ):
        super(CSI_model, self).__init__()
        self.capture_module = CaptureModule(dim_x_t, dim_embedding_wa, dim_hidden, dim_v_j)
        self.score_module = ScoreModule(dim_y_i, dim_embedding_wu)
        self.integrate_module = IntegrateModule(dim_v_j)
        #self.integrate_module = IntegrateModule(hidden_size, user_feature_size)

    def forward(self, x_t, lengths, y_i, m_j):
        # article score
        v_j = self.capture_module(x_t, lengths)
        
       

        s_i, y_i_t = self.score_module(y_i)
        s_i = s_i.squeeze(-1)
        prediction = self.integrate_module(v_j, s_i, m_j)
        prediction = prediction.squeeze(-1)

        return prediction, y_i_t, v_j
    
    def get_wu(self):
        return self.score_module.get_wu()
