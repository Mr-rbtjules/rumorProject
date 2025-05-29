import torch
import torch.optim as optim
import torch.utils.tensorboard
import rumorProject as RP
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import gc
from pathlib import Path

# for reproducibility
torch.manual_seed(RP.config.SEED_TORCH)  
np.random.seed(RP.config.SEED_NP)



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
            max_seq_len=50,  # Maximum sequence length for articles
            batch_size=256, 
            threshold=0.5,
            dim_hidden=50,
            dim_v_j=100,
            lambda_reg=0.01,
            reg_all=False,
            simple_model=False, # If True, use a simpler model
            balance_classes=False
    ):
        self.model = None
        self.simple_model = simple_model
        self.optimizer = None
        self.scheduler = None
        self.database = database
        self.frac_data = database.frac
        self.balance_classes = balance_classes
        self.max_seq_len = max_seq_len #self.database.T 
        self.device = device
        self.threshold = threshold
        self.dim_hidden = dim_hidden
        self.dim_v_j = dim_v_j
        self.lambda_reg = lambda_reg
        self.reg_all = reg_all
        self.learning_rate = learning_rate

        self.user_features_tensor = self.get_user_features_tensor()

        self.article_ids = self.database.article_ids()



        self.X_sequences, self.lengths, self.y_labels = self.create_X_y()
        self._free_memory()
    
        self.dataset = RP.RumorDataset(
            lengths=self.lengths,
            article_features=self.X_sequences,
            user_features=self.user_features_tensor,
            labels=self.y_labels,
            article_user_idxs=self.database.article_user_idxs,
            device=self.device
        )

        orig_counts = torch.bincount(self.dataset.labels)
        print(f"[Dataset] Original label distribution: "
        f"{{'neg': {orig_counts[0].item()}, 'pos': {orig_counts[1].item()}}}")
        # Balance the dataset by undersampling the majority class
        if balance_classes:
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
        else:
            self.balanced_labels = self.dataset.labels
        self.set_model()

        
        self.train_dataset = None
        self.val_dataset = None
        self.set_train_val_dataset()

        self.standardize_data()
        # ── One‑time device transfer of all article tensors ────────────
        if self.device.type in {"cuda", "mps"}:
            # self.X_sequences is shared by every Dataset/Subset slice,
            # so moving it once avoids per‑batch .to(device) copies.
            self.X_sequences = self.X_sequences.to(self.device, non_blocking=True)
            # Keep the RumorDataset reference in sync
            self.dataset.dataset.article_features = self.X_sequences
        
        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            sampler=None,
            collate_fn=RP.dataBase.rumor_collate
        )  # Use custom collate function to handle variable-length sequences)
        # keep the validation loader unchanged        
        self.val_loader = DataLoader(
            dataset=self.val_dataset, 
            batch_size=batch_size,
            collate_fn=RP.dataBase.rumor_collate
        )

        self.set_optimizer_and_scheduler()
        self.writer = self.set_tensorboard_logs()

        # Print label distributions for checking class balance
        total_counts = torch.bincount(self.balanced_labels)
        print(f"Overall label distribution: {{'neg': {total_counts[0].item()}, 'pos': {total_counts[1].item()}}}")
        train_counts = torch.bincount(self.balanced_labels[self.train_dataset.indices])
        print(f"Training label distribution: {{'neg': {train_counts[0].item()}, 'pos': {train_counts[1].item()}}}")
        val_counts = torch.bincount(self.balanced_labels[self.val_dataset.indices])
        print(f"Validation label distribution: {{'neg': {val_counts[0].item()}, 'pos': {val_counts[1].item()}}}")


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
        
        # Free additional memory now that datasets are created
        self._free_train_memory()
    
        return None
    
    def standardize_data(self):
        print("Standardizing data...")
        # Get only training indices from the original features
        train_indices = []
        for i in self.train_dataset.indices:
            train_indices.append(self.dataset.indices[i])

        sum_values = torch.zeros(2)
        sum_squares = torch.zeros(2)
        count = 0

        for idx in train_indices:
            seq = self.X_sequences[idx]
            length = self.lengths[idx].item()
            valid_data = seq[:length, :2]
            
            # Log-transform in-place
            valid_data[:, 0].log1p_()
            valid_data[:, 1].log1p_()
            
            sum_values += valid_data.sum(dim=0)
            sum_squares += (valid_data ** 2).sum(dim=0)
            count += length

        self.scalar_means = sum_values / count
        self.scalar_stds = (sum_squares / count - self.scalar_means ** 2).sqrt().clamp_min(1e-6)

    # Standardize ALL sequences in-place
        for seq, length in zip(self.X_sequences, self.lengths):
            # Log-transform non-training sequences
            
            seq[:length.item(), :2].log1p_() #.item because length is a tensor of size 1
            # Standardize
            seq[:length.item(), :2].sub_(self.scalar_means).div_(self.scalar_stds)
        
        print("Standardization complete.")    


    
    def create_X_y(self):
        X_sequences = []
        lengths = []
        y_labels = []
        
        for art_id in self.article_ids:
            seq, label = self.database.article_sequence(art_id)
            # Immediately limit sequence to max_seq_len to save memory
            if len(seq) > self.max_seq_len:
                seq = seq[:self.max_seq_len]
            
            size_raw = len(seq)
            lengths.append(size_raw)
            
            # Create tensor directly at the right size to avoid list conversion
            if seq:
                feature_dim = 2 + len(seq[0]['x_u']) + len(seq[0]['x_tau'])
            else:
                feature_dim = 0
            raw = torch.zeros(
                size =(self.max_seq_len, feature_dim), 
                dtype=torch.float
            )
            
            # Only fill valid timesteps
            for j, item in enumerate(seq):
                features = [item['eta'], item['delta_t']]
                features.extend(item['x_u'])
                features.extend(item['x_tau'])
                raw[j] = torch.tensor(features, dtype=torch.float)
            
            X_sequences.append(raw)
            y_labels.append(label)
            
            # Clear seq after processing to free memory during loop
            seq = None
            
        X = torch.stack(X_sequences)
        L = torch.tensor(lengths, dtype=torch.long)
        Y = torch.tensor(y_labels, dtype=torch.long)
        return X, L, Y


       
    def get_user_features_tensor(self):
        self.user_ids, self.user_features = self.database.user_feature_matrix()
    
        return torch.tensor(
            data=self.user_features, 
            dtype=torch.float, 
            device=self.device, 
            requires_grad=False
        )

    def set_tensorboard_logs(self):
        if self.simple_model:
                log_dir = Path(RP.config.LOGS_DIR) / "tensorboard" / \
                f"Simple_seqlen{self.max_seq_len}_dimvj{self.dim_v_j}_dimh{self.dim_hidden}_frac{self.frac_data}_lr{self.learning_rate}_reg{self.lambda_reg}_regall{self.reg_all}"
        else:
            log_dir = Path(RP.config.LOGS_DIR) / "tensorboard" / \
                f"seqlen{self.max_seq_len}_dimvj{self.dim_v_j}_dimh{self.dim_hidden}_frac{self.frac_data}_lr{self.learning_rate}_reg{self.lambda_reg}_regall{self.reg_all}_balance{self.balance_classes}"
        log_dir.mkdir(parents=True, exist_ok=True)  # create directory if it doesn't exist
        return torch.utils.tensorboard.SummaryWriter(log_dir=str(log_dir))

    def _free_memory(self):
        """Free unnecessary data structures from memory after tensor creation"""
        print("Freeing memory...")
        
        # Free large database structures
        if hasattr(self.database, 'tweets_df') \
        and self.database.tweets_df is not None:
            del self.database.tweets_df
            
        # Clear threads_seq and other large structures after we've extracted what we need
        large_db_attrs = [
            'threads_seq', 'events', 
            'threads_source', 'user_vecs_global', 
            'user_vecs_source'
        ]
        for attr in large_db_attrs:
            if hasattr(self.database, attr) \
            and getattr(self.database, attr) is not None:
                setattr(self.database, attr, None)
        
        # Clear original numpy user features as we now have the tensor version
        if hasattr(self, 'user_features') and self.user_features is not None:
            del self.user_features
        
        # After creating datasets, many original structures aren't needed
        # We'll do this in a subsequent call from train() or after dataset creation
        # as we still need these for dataset creation
        
        # Force garbage collection
        gc.collect()
        
        # Clear GPU cache if using GPU
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        elif self.device.type == 'mps':
            torch.mps.empty_cache()
        
        print("Memory freed")
    
    
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

        if self.simple_model:
            self.model = RP.Simple_CSI_model(
                dim_x_t=dim_x_t,
                dim_embedding_wa=dim_embedding_wa, #choisir pour que x_t tilde ait dim=100?
                dim_hidden=dim_hidden, #au pif? oui
                dim_v_j=dim_v_j,
                dim_y_i=dim_y_i,
                dim_embedding_wu=dim_embedding_wu
            ).to(self.device)
        else:
            self.model = RP.CSI_model(
                dim_x_t=dim_x_t,
                dim_embedding_wa=dim_embedding_wa, #choisir pour que x_t tilde ait dim=100?
                dim_hidden=dim_hidden, #au pif? oui
                dim_v_j=dim_v_j,
                dim_y_i=dim_y_i,
                dim_embedding_wu=dim_embedding_wu
            ).to(self.device)

        """#maybe remove
        with torch.no_grad():
            # final sigmoid layer is self.model.integrate_module.fc
            # bias = log(pos/neg)
            pos = (self.balanced_labels == 1).sum().item()
            neg = (self.balanced_labels == 0).sum().item()
            init_bias = np.log(pos / neg)
            self.model.integrate_module.fc.bias.fill_(init_bias)"""

    def set_optimizer_and_scheduler(self):
        self.optimizer = optim.Adam(
            params=self.model.parameters(), 
            lr    =self.learning_rate
        )
        # Add learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode    ='min', 
            factor  =0.1, 
            patience=8 # Reduce LR after 8 epochs without validation loss improvement (matching early stopping patience)
        )

    
    def train(self, num_epochs=20):

        patience = 15 # Increased patience to allow scheduler to trigger
        best_val_loss = float('inf')
        epochs_no_improve = 0

        print("\n=== DATASET STATS ===")
        print(f"Total dataset size: {len(self.dataset)}")
        print(f"Training set size: {len(self.train_dataset)}")
        print(f"Validation set size: {len(self.val_dataset)}")
        print(f"User features shape: {self.user_features_tensor.shape}")

        for epoch in range(num_epochs):
            total_loss_training=0
            total_loss_validation=0
            self.model.train() #add dropout 0.2 on wa and wr
            train_all_labels = []
            train_all_preds = []
            #trainign loop
            for batch in self.train_loader:
                self.optimizer.zero_grad()

                # batch['y_is'] is (n_engaged, dim_y_i) per example
                predictions, user_repr, art_repr = self.model(
                    x_t     = batch['article_features'],  # already on device
                    lengths = batch['lengths'],
                    y_flat  = batch['y_flat'].to(self.device), #all engaged users in the batch
                    src_idx = batch['src_idx'].to(self.device) #to keep track of which article each user belongs to
                )


                if not self.simple_model: 
                    Wu = self.model.get_wu()
                else:
                    Wu = None

                loss = self.loss_function(
                    predictions, 
                    batch['labels'].to(self.device), 
                    Wu
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    parameters=self.model.parameters(), 
                    max_norm  =5.0
                )
                self.optimizer.step()
                total_loss_training += loss.item()

                train_all_labels.append(batch['labels'].cpu().numpy())
                train_all_preds.append(predictions.detach().cpu().numpy())


            total_loss_training /= len(self.train_loader)
            train_all_labels = np.concatenate(train_all_labels)
            train_all_preds = np.concatenate(train_all_preds)
            train_predictions_binary = (train_all_preds >= self.threshold).astype(int)
            train_accuracy = np.mean((train_predictions_binary == train_all_labels).astype(float))
            
            #validation loop
            self.model.eval()
            all_labels = []
            all_preds = []
            for batch in self.val_loader:

                labels = batch['labels'].to(self.device).float()
                with torch.no_grad():
                    predictions, user_repr, art_repr = self.model(
                        x_t     = batch['article_features'],
                        lengths = batch['lengths'],
                        y_flat  = batch['y_flat'].to(self.device),
                        src_idx = batch['src_idx'].to(self.device)
                    )

                    if not self.simple_model: 
                        Wu = self.model.get_wu()
                    else:
                        Wu = None

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
                    print(f"Early stopping at epoch {epoch+1} \
                          (no improvement for {patience} epochs).")
                    break

            # Convert to numpy arrays for metrics calculation
            all_labels = np.concatenate(all_labels)
            all_preds = np.concatenate(all_preds)
            print(all_labels[:3],all_preds[:3])

            print(all_preds.shape)
            #predictions_binary = np.round(all_preds)
            custom_threshold = self.threshold  # Try different values
            predictions_binary = (all_preds >= custom_threshold).astype(int)
            val_accuracy = np.mean((predictions_binary == all_labels).astype(float))
            accuracy = val_accuracy
            print(f'Accuracy: {accuracy:.4f}')
            # At the end of validation
            unique_preds, counts = np.unique(predictions_binary, return_counts=True)
            print(f"Prediction distribution: {dict(zip(unique_preds.astype(int), counts))}")

            # Add this to see if predictions are changing
            print(f"Raw prediction stats - Min: {np.min(all_preds):.4f}, Max: {np.max(all_preds):.4f}, Mean: {np.mean(all_preds):.4f}")
            print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {total_loss_training:.4f}, Validation Loss: {total_loss_validation:.4f}')
            self.writer.add_scalar('Loss/Train', total_loss_training, epoch)
            self.writer.add_scalar('Loss/validation', total_loss_validation, epoch)
            self.writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
            self.writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)
        
        # Additional logging if needed
        self.writer.add_scalar('LearningRate', self.optimizer.param_groups[0]['lr'], epoch)
        
                

    def _free_train_memory(self):
        """Free memory after datasets are created but before training"""
        print("Freeing additional memory before training...")
        
        # Reference to database is no longer needed after dataset creation
        self.database = None
        
        # We only need the article IDs for the user_article_mask creation
        self.article_ids = None
        
        # Force garbage collection
        
        
        # Clear GPU cache again
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()
        
        print("Additional memory freed")




    
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
            if Wu is not None:
                # Wu is the user weights matrix
                l2_reg = 0.5 * self.lambda_reg * torch.norm(Wu, p=2) ** 2
        
        # Total loss
        return bce_loss + l2_reg
