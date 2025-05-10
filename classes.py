import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
from torch.utils.data import random_split, DataLoader, Dataset
import config
from dataBase import DataBase


#format input : x=(nbre_engagement, temps entre chaque engagement,source = x_u ,x_t=caractéristiques d'un texte)
class CaptureModule(nn.Module):
    # 3 layers : a linear NN, then a RNN (LSTM) and the last one also a linear NN
    def __init__(self, input_size, embedding_size, hidden_size):
        super(CaptureModule, self).__init__()
        self.embedding = nn.Linear(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, x_t):  # xt = (η, ∆t, xu , xτ) are the features of the article
        x_tt = torch.tanh(self.embedding(x_t))  # first layer : LNN
        _, (h_n, _) = self.rnn(x_tt)  # second layer : LSTM
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

    def forward(self, article_features, user_features, user_article_mask):
        # article score
        v_j = self.capture_module(article_features)
        article_repr=v_j
        # user score

        s_i, user_repr = self.score_module(user_features)
        s_i = s_i.squeeze(-1)
        prediction = self.integrate_module(v_j, s_i, user_article_mask)
        prediction = prediction.squeeze(-1)

        return prediction, user_repr, article_repr
    

def loss_function(predictions, labels, Wu, lambda_reg=0.01):
    #eps avoid log0
    epsilon = 1e-7
    predictions = torch.clamp(predictions, epsilon, 1-epsilon)
    
    L_accuracy = -torch.mean(labels * torch.log(predictions) + 
                      (1 - labels) * torch.log(1 - predictions))
    
    L_regularization = lambda_reg * torch.norm(Wu, p=2)**2 / 2
    
    return L_accuracy + L_regularization


def create_dataset():
    #pour convertir les données en tensor
    article_ids = data.article_ids()
    X_sequences = []
    y_labels = []
    
    for art_id in article_ids:
        seq, label = data.article_sequence(art_id)
        
        feature_list = []
        for item in seq:
            # Format: [eta, delta_t, x_u, x_tau]
            features = [item['eta'], item['delta_t']]
            features.extend(item['x_u'])
            features.extend(item['x_tau'])
            feature_list.append(features)
        
        max_seq_len = 100 #On doit avoir la même taille pour tous les articles -> si trop grand, on coupe, sinon on remplit avec 0
        if len(feature_list) > max_seq_len:
            feature_list = feature_list[:max_seq_len]
        else:
            padding = [[0] * len(feature_list[0]) for _ in range(max_seq_len - len(feature_list))]
            feature_list.extend(padding)
            
        X_sequences.append(torch.tensor(feature_list, dtype=torch.float))
        y_labels.append(label)
    for i in range(len(X_sequences)):
        mean = torch.mean(X_sequences[i], dim=0)
        std = torch.std(X_sequences[i], dim=0)
        std[std < 1e-5] = 1.0  # Prevent division by zero
        X_sequences[i] = (X_sequences[i] - mean) / std
    
    # print('x',X_sequences[0])
    # print('y',y_labels[0])
    return X_sequences, torch.tensor(y_labels, dtype=torch.float)

class RumorDataset(Dataset):
    def __init__(self, article_features, user_features, labels, user_article_mask):
        self.article_features = article_features
        self.user_features = user_features
        self.labels = labels
        self.user_article_mask = user_article_mask
        
    def __len__(self):
        return len(self.article_features)
    
    def __getitem__(self, idx):  
        return {
            'article_features': self.article_features[idx],
            'user_features': self.user_features,
            'labels': self.labels[idx],
            'user_article_mask': self.user_article_mask[idx]
        }
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = DataBase(bin_size=1, user_svd_dim=50)


user_ids, user_features = data.user_feature_matrix()
user_features_tensor = torch.tensor(user_features, dtype=torch.float)
article_ids = data.article_ids()

X_sequences, y_labels = create_dataset() #label dit si rumeur ou pas et X_sequ=[eta, delta_t, x_u, x_tau]

#user-article mask: 1 si user a interagi avec l'article et 0 sinon
user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
user_article_mask = torch.zeros(len(article_ids), len(user_ids))
for i, art_id in enumerate(article_ids):
    thread_users = data.tweets_df[data.tweets_df.thread == art_id].user_id.unique()
    for user_id in thread_users:
        if user_id in user_id_to_idx:
            user_article_mask[i, user_id_to_idx[user_id]] = 1.0


dataset = RumorDataset(
    article_features=X_sequences,
    user_features=user_features_tensor,
    labels=y_labels,
    user_article_mask=user_article_mask
)


input_size = X_sequences[0].shape[1]  # eta, delta_t, x_u, x_tau
embedding_size = 100
hidden_size = 100
user_feature_size = user_features_tensor.shape[1]

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)  # For reproducibility
)


batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

model = CSI_model(
    input_size=input_size,
    user_feature_size=user_feature_size,
    embedding_size=embedding_size, #choisir pour que x_t tilde ait dim=100?
    hidden_size=hidden_size #au pif?
).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)




num_epochs = 20
best_accuracy = 0.0
print("\n=== DATASET STATS ===")
print(f"Total dataset size: {len(dataset)}")
print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")
print(f"User features shape: {user_features_tensor.shape}")
print(f"User-article mask shape: {user_article_mask.shape}")
print(f"Non-zero entries in mask: {user_article_mask.sum().item()}")

for epoch in range(num_epochs):
    total_loss_training=0
    total_loss_validation=0
    model.train()
    #trainign loop
    for batch in train_loader:
        #print(batch)
        article_features = batch['article_features'].to(device)
        user_features = batch['user_features'].to(device)
        labels = batch['labels'].to(device)
        mask = batch['user_article_mask'].to(device)
        
        optimizer.zero_grad()
        
        predictions, user_repr, article_repr = model(article_features, user_features, mask)

        Wu = model.score_module.user_fc.weight

        loss = loss_function(predictions, labels, Wu)
        loss.backward()
        optimizer.step()
        total_loss_training+=loss.item()
    #validation loop
    model.eval()
    all_labels = []
    all_preds = []
    for batch in val_loader:
        article_features = batch['article_features'].to(device)
        user_features = batch['user_features'].to(device)
        labels = batch['labels'].to(device)
        mask = batch['user_article_mask'].to(device)
        
        with torch.no_grad():
            predictions, user_repr, article_repr = model(article_features, user_features, mask)
            Wu = model.score_module.user_fc.weight
            loss = loss_function(predictions, labels, Wu)
            total_loss_validation += loss.item()
        
        # Convert to numpy arrays for consistent processing
        all_labels.append(labels.cpu().numpy())
        all_preds.append(predictions.cpu().numpy())

    total_loss_validation /= len(val_loader)

    # Convert to numpy arrays for metrics calculation
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    print(all_labels[:3],all_preds[:3])

    print(all_preds.shape)
    predictions_binary = np.round(all_preds)
    accuracy = np.mean((predictions_binary == all_labels).astype(float))
    print(f'Accuracy: {accuracy:.4f}')
    # At the end of validation
    unique_preds, counts = np.unique(predictions_binary, return_counts=True)
    print(f"Prediction distribution: {dict(zip(unique_preds.astype(int), counts))}")

    # Add this to see if predictions are changing
    print(f"Raw prediction stats - Min: {np.min(all_preds):.4f}, Max: {np.max(all_preds):.4f}, Mean: {np.mean(all_preds):.4f}")
    if accuracy>best_accuracy:
        best_accuracy=accuracy
        torch.save(model.state_dict(), 'best_model.pth')
    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {total_loss_training:.4f}, Validation Loss: {total_loss_validation:.4f}')
        




#datas qu'on veut; 
"""
1) article/event:
seq = [
    {
        'eta': engagement_value,
        'delta_t': time_difference,
        'x_u': [user_feature_1, user_feature_2, ...],
        'x_tau': [text_feature_1, text_feature_2, ...] 
    },
    # More posts...
]

2)User feature matrix

user_features = [
    [feature1, feature2, ...],  # User 1
    [feature1, feature2, ...],  # User 2
    # More users...
]


3) user-article mask:
user_article_mask[article_index, user_index] = 1.0  # If user engaged with article


4) labels:
tensor containing truth for each event (1 or 0)
labels = torch.tensor([1, 0, 1, ...], dtype=torch.float)
"""





# ex_model=CaptureModule(3, 5, 10)
# x = torch.randn(2, 3, 3)  # batch_size=2, seq_len=3, input_size=3
# print(x)
# output = ex_model(x)
# print(output) 