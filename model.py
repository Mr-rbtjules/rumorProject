import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


#format input : x=(nbre_engagement, temps entre chaque engagement,source = x_u ,x_t=caractéristiques d'un texte)
class CaptureModule(nn.Module):
    # 3 layers : a linear NN, then a RNN (LSTM) and the last one also a linear NN
    def __init__(self, input_size, embedding_size, hidden_size):
        super(CaptureModule, self).__init__()
        self.embedding = nn.Linear(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, x_t, lengths):  # xt = (η, ∆t, xu , xτ) are the features of the article
        x_tt = torch.tanh(self.embedding(x_t))  # first layer : LNN
        packed = pack_padded_sequence(x_tt,
                                      lengths.cpu(),
                                      batch_first=True,
                                      enforce_sorted=False)
        _, (h_n, _) = self.rnn(packed)  # second layer : LSTM
        #print('shape',h_n.shape)
        h_n = h_n.squeeze(0)  # remove the first dimension (batch_size, hidden_size)
        #print('after',h_n.shape)
        v_j = torch.tanh(self.fc(h_n))  # last layer : LNN
        return v_j


class ScoreModule(nn.Module):
    def __init__(self, user_feature_size, hidden_size):
        super(ScoreModule, self).__init__()
        self.user_fc = nn.Linear(user_feature_size, hidden_size)
        self.score_fc = nn.Linear(hidden_size, 1)
        
    def forward(self, user_features):

        user_repr = torch.tanh(self.user_fc(user_features)) # ỹ_i = tanh(W_u*y_i + b_u)
        
        scores = torch.sigmoid(self.score_fc(user_repr)) # s_i = σ(w_sT*ỹ_i + b_s)
        
        return scores, user_repr


class IntegrateModule(nn.Module):
    def __init__(self, article_representations_size, user_scores_dim=1):
        super(IntegrateModule, self).__init__()
        self.fc = nn.Linear(article_representations_size + user_scores_dim, 1)
    
    def forward(self, v_j, s_i, user_article_mask):
        # user_article_mask is already the correct mask for this batch's articles
        # No need to index it further
        batch_mask = user_article_mask
        
        # Rest of the code remains the same
        masked_scores = s_i * batch_mask
        sum_scores = torch.sum(masked_scores, dim=1)
        count_users = torch.sum(batch_mask, dim=1)
        count_users = torch.clamp(count_users, min=1.0)
        p_j = (sum_scores / count_users).unsqueeze(1)
        c_j = torch.cat((v_j, p_j), dim=1)
        prediction = torch.sigmoid(self.fc(c_j))
        return prediction

class CSI_model(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, user_feature_size):
        super(CSI_model, self).__init__()
        self.capture_module = CaptureModule(input_size, embedding_size, hidden_size)
        self.score_module = ScoreModule(user_feature_size, hidden_size)
        self.integrate_module = IntegrateModule(hidden_size)
        #self.integrate_module = IntegrateModule(hidden_size, user_feature_size)

    def forward(self, article_features, lengths, user_features, user_article_mask):
        # article score
        v_j = self.capture_module(article_features, lengths)
        article_repr = v_j
        # user score

        s_i, user_repr = self.score_module(user_features)
        s_i = s_i.squeeze(-1)
        prediction = self.integrate_module(v_j, s_i, user_article_mask)
        prediction = prediction.squeeze(-1)

        return prediction, user_repr, article_repr
   