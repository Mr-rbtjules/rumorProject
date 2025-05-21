import torch
import torch.optim as optim
import torch.utils.tensorboard
import rumorProject as RP
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import gc
from pathlib import Path


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

    def __init__(
            self, 
            database,
            device, 
            learning_rate=0.001, 
            batch_size=64, 
            threshold=0.5,
            dim_hidden=50,
            dim_v_j=100,
            lambda_reg=0.01,
            reg_all=False
    ):
        self.model = None
        self.database = database
        self.max_seq_len = 20 #self.database.T
        self.device = device
        self.threshold = threshold
        self.dim_hidden = dim_hidden
        self.dim_v_j = dim_v_j
        self.lambda_reg = lambda_reg
        self.reg_all = reg_all

        self.user_ids, self.user_features = self.database.user_feature_matrix()
        # trainer.py  __init__
        self.user_features_tensor = torch.tensor(
            self.user_features, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.article_ids = self.database.article_ids()
        #anciennemnet X_sequences, y_labels = create_dataset() #label dit si rumeur ou pas et X_sequ=[eta, delta_t, x_u, x_tau]
        self.X_sequences, self.lengths, self.y_labels = self.create_X_y()

        


        self.user_article_mask = self.create_user_article_mask()

    
        

        self.dataset = RP.RumorDataset(
            lengths=self.lengths,
            article_features=self.X_sequences,
            user_features=self.user_features_tensor,
            labels=self.y_labels,
            user_article_mask=self.user_article_mask,
            device=self.device
        )
        # Balance the dataset by undersampling the majority class
        labels = self.dataset.labels
        neg_idx = (labels == 0).nonzero(as_tuple=True)[0]
        pos_idx = (labels == 1).nonzero(as_tuple=True)[0]
        min_count = min(len(neg_idx), len(pos_idx))
        # reproducible random sampling
        g = torch.Generator().manual_seed(RP.config.SEED_RAND)
        neg_sample = neg_idx[torch.randperm(len(neg_idx), generator=g)[:min_count]]
        pos_sample = pos_idx[torch.randperm(len(pos_idx), generator=g)[:min_count]]
        balanced_idx = torch.cat([neg_sample, pos_sample])
        self.dataset = Subset(self.dataset, balanced_idx.tolist())
        self.balanced_labels = self.dataset.dataset.labels[self.dataset.indices]


        self.set_model()
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate
        )
        # Add learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.1, 
            patience=8 # Reduce LR after 8 epochs without validation loss improvement (matching early stopping patience)
        )

        log_dir = Path(RP.config.LOGS_DIR) / "tensorboard" / "modeltest1"
        log_dir.mkdir(parents=True, exist_ok=True)  # create directory if it doesn't exist
        self.writer = torch.utils.tensorboard.SummaryWriter(log_dir=str(log_dir))


        self.train_dataset = None
        self.val_dataset = None
        self.set_train_val_dataset()

        self.standardize_data()

        
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=batch_size,
                                       sampler=None)
        # keep the validation loader unchanged        
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size)

        # Print label distributions for checking class balance
        total_counts = torch.bincount(self.balanced_labels)
        print(f"Overall label distribution: {{'neg': {total_counts[0].item()}, 'pos': {total_counts[1].item()}}}")
        train_counts = torch.bincount(self.balanced_labels[self.train_dataset.indices])
        print(f"Training label distribution: {{'neg': {train_counts[0].item()}, 'pos': {train_counts[1].item()}}}")
        val_counts = torch.bincount(self.balanced_labels[self.val_dataset.indices])
        print(f"Validation label distribution: {{'neg': {val_counts[0].item()}, 'pos': {val_counts[1].item()}}}")
        
    def standardize_data(self):
        print("Standardizing data...")
        # Get only training indices from the original features
        train_indices = [self.dataset.indices[i] for i in self.train_dataset.indices]
        
        # Collect valid timesteps from training data only
        valid_timesteps = []
        for idx in train_indices:
            seq = self.X_sequences[idx]
            length = self.lengths[idx]
            valid_timesteps.append(seq[:length])
        
        flat = torch.cat(valid_timesteps, dim=0)
        
        # Log-transform the first two features on training data
        flat[:, 0] = torch.log1p(flat[:, 0])
        flat[:, 1] = torch.log1p(flat[:, 1])
        
        # Compute means and stds ONLY for scalar features (first two columns)
        # These features are already log-transformed in DataBase.py
        self.scalar_means = flat[:, :2].mean(dim=0)
        self.scalar_stds = flat[:, :2].std(dim=0).clamp_min(1e-6)
        
        # Now standardize ALL data using train-derived stats
        # The log transformation is now done in DataBase.py
        self.X_sequences = torch.stack([
            self._standardise_seq(seq, length) 
            for seq, length in zip(self.X_sequences, self.lengths)
        ])
        
        # Update the datasets with standardized features
        self.dataset.dataset.article_features = self.X_sequences
        print("Standardization complete.")
    def _standardise_seq(self, seq, length=None):
        seq = seq.clone()
        # If length not provided, standardize the whole sequence
        if length is None:
            seq[:, :2] = (seq[:, :2] - self.scalar_means) / self.scalar_stds
        else:
            # Only standardize up to the actual length
            seq[:length, :2] = (seq[:length, :2] - self.scalar_means) / self.scalar_stds
        return seq
    def set_model(self):

        #capture
        dim_x_t = self.X_sequences[0].shape[1] # eta, delta_t, x_u, x_tau
        dim_embedding_wa = 100 #dim x_tt
        dim_hidden = self.dim_hidden #dimension of h_t and c_t

        dim_v_j = self.dim_v_j
        dim_embedding_wr = dim_v_j

        #score
        dim_y_i = self.user_features_tensor.shape[1]
        dim_embedding_wu = 100
        dim_y_it = dim_embedding_wu
        dim_w_s = dim_y_i
        dim_s_i = 1


        #integrate
        dim_c_j = dim_v_j + dim_s_i
        dim_w_c = dim_c_j


        self.model = RP.CSI_model(
            dim_x_t=dim_x_t,
            dim_embedding_wa=dim_embedding_wa, #choisir pour que x_t tilde ait dim=100?
            dim_hidden=dim_hidden, #au pif? oui
            dim_v_j=dim_v_j,
            dim_y_i=dim_y_i,
            dim_embedding_wu=dim_embedding_wu
        ).to(self.device)

        # after you create self.model
        with torch.no_grad():
            # final sigmoid layer is self.model.integrate_module.fc
            # bias = log(pos/neg)
            pos = (self.balanced_labels == 1).sum().item()
            neg = (self.balanced_labels == 0).sum().item()
            init_bias = np.log(pos / neg)
            self.model.integrate_module.fc.bias.fill_(init_bias)
        
    def train(self, num_epochs=20):

        patience = 8 # Increased patience to allow scheduler to trigger
        best_val_loss = float('inf')
        epochs_no_improve = 0

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
            self.model.train() #add dropout 0.2 on wa and wr

            #trainign loop
            for batch in self.train_loader:
                article_features = batch['article_features'].to(self.device)
                #user_features = batch['user_features'].to(self.device)
                user_features = self.user_features_tensor #why not in model directly ?
                labels = batch['labels'].to(self.device)
                mask = batch['user_article_mask'].to(self.device)
                lengths = batch['lengths']            # keep on CPU for pack_padded_sequence

                self.optimizer.zero_grad()


                predictions, user_repr, article_repr = self.model(
                    x_t=article_features, 
                    lengths=lengths, 
                    y_i=user_features, 
                    m_j=mask
                )

                Wu = self.model.get_wu()

                loss = self.loss_function(predictions, labels, Wu)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()
                total_loss_training += loss.item()

            total_loss_training /= len(self.train_loader)
            #validation loop
            self.model.eval()
            all_labels = []
            all_preds = []
            for batch in self.val_loader:
                article_features = batch['article_features'].to(self.device)
                #user_features = batch['user_features'].to(self.device)
                user_features = self.user_features_tensor #why not in model directly ?

                labels = batch['labels'].to(self.device).float()
                mask = batch['user_article_mask'].to(self.device)
                lengths = batch['lengths']            # keep on CPU for pack_padded_sequence

                with torch.no_grad():
                    predictions, user_repr, article_repr = self.model(
                        article_features, lengths, user_features, mask
                    )
                    Wu = self.model.get_wu()
                    loss = self.loss_function(predictions, labels, Wu)
                    total_loss_validation += loss.item()

                # Convert to numpy arrays for consistent processing
                all_labels.append(labels.cpu().numpy())
                all_preds.append(predictions.cpu().numpy())

            total_loss_validation /= len(self.val_loader)

            # Step the learning rate scheduler
            self.scheduler.step(total_loss_validation)

            if total_loss_validation < best_val_loss - 1e-4:
                best_val_loss = total_loss_validation
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs).")
                    break

            # Convert to numpy arrays for metrics calculation
            all_labels = np.concatenate(all_labels)
            all_preds = np.concatenate(all_preds)
            print(all_labels[:3],all_preds[:3])

            print(all_preds.shape)
            #predictions_binary = np.round(all_preds)
            custom_threshold = self.threshold  # Try different values
            predictions_binary = (all_preds >= custom_threshold).astype(int)
            accuracy = np.mean((predictions_binary == all_labels).astype(float))
            print(f'Accuracy: {accuracy:.4f}')
            # At the end of validation
            unique_preds, counts = np.unique(predictions_binary, return_counts=True)
            print(f"Prediction distribution: {dict(zip(unique_preds.astype(int), counts))}")

            # Add this to see if predictions are changing
            print(f"Raw prediction stats - Min: {np.min(all_preds):.4f}, Max: {np.max(all_preds):.4f}, Mean: {np.mean(all_preds):.4f}")
            print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {total_loss_training:.4f}, Validation Loss: {total_loss_validation:.4f}')
            self.writer.add_scalar('Loss/Train', total_loss_training, epoch)
            self.writer.add_scalar('Loss/validation', total_loss_validation, epoch)
                


    def set_train_val_dataset(self):
        # Retrieve labels for the current dataset, handling Subset wrappers
        base_labels = self.dataset.dataset.labels
        indices = self.dataset.indices
        labels = base_labels[indices].cpu().numpy()
    
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=0.2,
            random_state=RP.config.SEED_RAND
        )
        train_idx, val_idx = next(
            splitter.split(
                np.zeros(len(labels)),
                labels
            )
        )
        self.train_dataset = Subset(self.dataset, train_idx)
        self.val_dataset = Subset(self.dataset, val_idx)
        return None

    def create_user_article_mask(self):
        user_id_to_idx = {uid: i for i, uid in enumerate(self.user_ids)}
        user_article_mask = torch.zeros(
            len(self.article_ids), 
            len(self.user_ids),
            dtype=torch.bool
        )
        for i, art_id in enumerate(self.article_ids):
            thread_users = self.database.tweets_df[self.database.tweets_df.thread_id == art_id].user_id.unique()
            for user_id in thread_users:
                if user_id in user_id_to_idx:
                    user_article_mask[i, user_id_to_idx[user_id]] = True
        return user_article_mask


    def create_X_y(self):
        #pour convertir les données en tensor
        X_sequences = []
        lengths = []
        y_labels = []
        
        for art_id in self.article_ids:
            seq, label = self.database.article_sequence(art_id)
            size_raw = len(seq)
            lengths.append(size_raw)
            feature_list = []
            for item in seq:
                # Format: [eta, delta_t, x_u, x_tau]
                features = [item['eta'], item['delta_t']]
                features.extend(item['x_u'])
                features.extend(item['x_tau'])
                feature_list.append(features)

            #normalization ? to device ?
            raw = torch.tensor(feature_list, dtype=torch.float)
                    # pad (or truncate) to exactly 27 timesteps
            max_len = self.max_seq_len  # New fixed maximum length
            if raw.size(0) > max_len:
                # Truncate to first 10 timesteps
                raw = raw[:max_len]
                # Update the stored length if we truncated
                lengths[-1] = max_len
            elif raw.size(0) < max_len:
                # Pad with zeros at the end
                pad = torch.zeros((max_len - raw.size(0), raw.size(1)), device=raw.device)
                raw = torch.cat([raw, pad], dim=0)
            X_sequences.append(raw)
            y_labels.append(label)


            
        X = torch.stack(X_sequences)
        L = torch.tensor(lengths, dtype=torch.long)
        Y = torch.tensor(y_labels, dtype=torch.long)
        return X, L, Y


    """
    def loss_function(self, predictions, labels, Wu, eps=1e-7):
        p = predictions.clamp(eps, 1. - eps)          # avoid 0 or 1
        l_acc = -(labels * p.log() + (1 - labels) * (1 - p).log()).mean()
        if self.reg_all:
                l_reg = 0
                for param in self.model.parameters():
                    l_reg += self.lambda_reg * param.norm(p=2).pow(2) / 2
                
        else:
            l_reg = self.lambda_reg * Wu.norm(p=2).pow(2) / 2  # same regulariser
        
        return l_acc + l_reg"""
    
    def loss_function(self, predictions, labels, Wu, eps=1e-7):
        # Binary cross-entropy loss

        p = predictions.clamp(eps, 1. - eps)
        labels = labels.float()

        bce_loss = torch.nn.functional.binary_cross_entropy(
            p, labels, reduction='mean'
        )
        
        # L2 regularization
        l2_reg = 0
        if self.reg_all:
            for param in self.model.parameters():
                l2_reg += 0.5 * self.lambda_reg * torch.norm(param, p=2) ** 2
        else:
            # Original L2 regularization on Wu
            l2_reg = 0.5 * self.lambda_reg * torch.norm(Wu, p=2) ** 2
        
        # Total loss
        return bce_loss + l2_reg
