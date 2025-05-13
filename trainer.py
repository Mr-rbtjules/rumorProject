import torch
import torch.optim as optim
import rumorProject as RP
from torch.utils.data import DataLoader, random_split
import numpy as np
import gc


class Trainer:

    """  main.py :   if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    

    database = RP.DataBase(
        bin_size=1,
        save_file_name="rumor_data_jules"
    )


    
    trainer = RP.Trainer( 
        database=database, 
        device=device
    )
    
    trainer.train(num_epochs=20)
    
    """

    def __init__(self, database, device, learning_rate=0.001, batch_size=16):
        self.model = None
        self.database = database
        self.device = device

        self.user_ids, self.user_features = self.database.user_feature_matrix()
        self.user_features_tensor = torch.tensor(self.user_features, dtype=torch.float)
        self.article_ids = self.database.article_ids()
        #anciennemnet X_sequences, y_labels = create_dataset() #label dit si rumeur ou pas et X_sequ=[eta, delta_t, x_u, x_tau]
        self.X_sequences, self.y_labels = self.create_X_y()
        
        self.user_article_mask = self.create_user_article_mask()

    
        

        self.dataset = RP.RumorDataset(
            article_features=self.X_sequences,
            user_features=self.user_features_tensor,
            labels=self.y_labels,
            user_article_mask=self.user_article_mask
        )
        self.set_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)



        self.train_dataset = None
        self.val_dataset = None
        self.set_train_val_dataset()
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size)
        self.means = {}
        self.stds = {}

    def set_model(self):
        input_size = self.X_sequences[0].shape[1]  # eta, delta_t, x_u, x_tau
        embedding_size = 100
        hidden_size = 100
        user_feature_size = self.user_features_tensor.shape[1]


        self.model = RP.CSI_model(
            input_size=input_size,
            user_feature_size=user_feature_size,
            embedding_size=embedding_size, #choisir pour que x_t tilde ait dim=100?
            hidden_size=hidden_size #au pif?
        ).to(self.device)

    def train(self, num_epochs=20):

        best_accuracy = 0.0
        print("\n=== DATASET STATS ===")
        print(f"Total dataset size: {len(self.dataset)}")
        print(f"Training set size: {len(self.train_dataset)}")
        print(f"Validation set size: {len(self.val_dataset)}")
        print(f"User features shape: {self.user_features_tensor.shape}")
        print(f"User-article mask shape: {self.user_article_mask.shape}")
        print(f"Non-zero entries in mask: {self.user_article_mask.sum().item()}")

        for epoch in range(num_epochs):
            total_loss_training=0
            total_loss_validation=0
            self.model.train()
            #trainign loop
            for batch in self.train_loader:
                #print(batch)
                article_features = batch['article_features'].to(self.device)
                user_features = batch['user_features'].to(self.device)
                labels = batch['labels'].to(self.device)
                mask = batch['user_article_mask'].to(self.device)
                
                self.optimizer.zero_grad()
                
                predictions, user_repr, article_repr = self.model(article_features, user_features, mask)

                Wu = self.model.score_module.user_fc.weight

                loss = self.loss_function(predictions, labels, Wu)
                loss.backward()
                self.optimizer.step()
                total_loss_training+=loss.item()
            #validation loop
            self.model.eval()
            all_labels = []
            all_preds = []
            for batch in self.val_loader:
                article_features = batch['article_features'].to(self.device)
                user_features = batch['user_features'].to(self.device)
                labels = batch['labels'].to(self.device)
                mask = batch['user_article_mask'].to(self.device)
                
                with torch.no_grad():
                    predictions, user_repr, article_repr = self.model(article_features, user_features, mask)
                    Wu = self.model.score_module.user_fc.weight
                    loss = self.loss_function(predictions, labels, Wu)
                    total_loss_validation += loss.item()
                
                # Convert to numpy arrays for consistent processing
                all_labels.append(labels.cpu().numpy())
                all_preds.append(predictions.cpu().numpy())

            total_loss_validation /= len(self.val_loader)

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
                torch.save(self.model.state_dict(), 'best_model.pth')
            print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {total_loss_training:.4f}, Validation Loss: {total_loss_validation:.4f}')
                


    def set_train_val_dataset(self):

        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            self.dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )

        return None

    def create_user_article_mask(self):
        user_id_to_idx = {uid: i for i, uid in enumerate(self.user_ids)}
        user_article_mask = torch.zeros(len(self.article_ids), len(self.user_ids))
        for i, art_id in enumerate(self.article_ids):
            thread_users = self.database.tweets_df[self.database.tweets_df.thread_id == art_id].user_id.unique()
            for user_id in thread_users:
                if user_id in user_id_to_idx:
                    user_article_mask[i, user_id_to_idx[user_id]] = 1.0
        return user_article_mask


    def create_X_y(self):
        #pour convertir les données en tensor
        article_ids = self.database.article_ids()
        X_sequences = []
        y_labels = []
        
        for art_id in article_ids:
            seq, label = self.database.article_sequence(art_id)
            
            feature_list = []
            for item in seq:
                # Format: [eta, delta_t, x_u, x_tau]
                features = [item['eta'], item['delta_t']]
                features.extend(item['x_u'])
                features.extend(item['x_tau'])
                feature_list.append(features)

            max_seq_len = 100
            raw = torch.tensor(feature_list, dtype=torch.float)
            mean = raw.mean(dim=0)
            std = raw.std(dim=0)
            std[std < 1e-5] = 1.0
            norm_raw = (raw - mean) / std
            if norm_raw.size(0) > max_seq_len:
                norm_raw = norm_raw[:max_seq_len]
            else:
                pad_len = max_seq_len - norm_raw.size(0)
                pad = torch.zeros(pad_len, norm_raw.size(1))
                norm_raw = torch.cat([norm_raw, pad], dim=0)
            X_sequences.append(norm_raw)
            y_labels.append(label)

            self.means[art_id] = mean
            self.stds[art_id] = std
        return X_sequences, torch.tensor(y_labels, dtype=torch.float)



    def loss_function(self, predictions, labels, Wu, lambda_reg=0.01):
        L_accuracy = -torch.mean(labels * torch.log(predictions) + 
                        (1 - labels) * torch.log(1 - predictions))
        
        L_regularization = lambda_reg * torch.norm(Wu, p=2)**2 / 2
        
        return L_accuracy + L_regularization