# pheme_csi_data.py
#import rumorProject as RP
import config
import tools

import json, pickle, os
from pathlib import Path
from collections import defaultdict, Counter
import networkx as nx
import numpy as np
import pandas as pd
from tqdm.auto import tqdm # for progress bar
from sentence_transformers import SentenceTransformer
from scipy.sparse import csr_matrix, save_npz, load_npz
from scipy.sparse.linalg import svds
import gc  # for garbage collection

class DataBase:
    """
    End-to-end loader + pre-processor for the PHEME rumour dataset,
    producing CSI-ready tensors.

         data = PhemeCSIData("data/pheme-rnr-dataset", bin_hours=1)
         X, y = data.article_sequence("ferguson_170.json")
         users, user_vecs = data.user_feature_matrix()

         note : pheme contain threads , each thread contain tweets = source or reactions
    """

    def __init__(
            self, bin_size: int = 1, #in hours
            user_svd_dim: int = 50, 
            batch_size: int = 16,
            use_checkpoints: bool = True
    ):
        self.root = Path(config.RAW_DATA_DIR)
        self.bin_h = bin_size
        self.user_dim = user_svd_dim
        self.batch_size = batch_size
        self.use_checkpoints = use_checkpoints
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(config.CACHE_DATA_DIR) / "checkpoints"
            
        self.cache_path = Path(config.CACHE_DATA_DIR) / \
            f"uDim_{user_svd_dim}_binSize_{bin_size}.pkl"
        self.tweets_df = None
        self.tweets = []          #(check _add_row_to_tweets to understand)
        self.threads = {}  # not containing answers, only time reactions
        self.article_seq = {}
        self.labels = {}
        self.user_vecs = {}  # Initialiser comme un dictionnaire vide au lieu de None

        self._model = None # SentenceTransformer model

        if self.cache_path.exists():
            self._load_cache()
        else:
            self._process_data()

    def _process_data(self):
        """either parse all data or load from pickle files"""
    
        #create checkpoint directory/files
        threads_checkpoint = self.checkpoint_dir / "threads.pkl"
        tweets_df_checkpoint = self.checkpoint_dir / "tweets_df.pkl"
        
        if self.use_checkpoints and threads_checkpoint.exists() and tweets_df_checkpoint.exists():
            print("Loading threads from cache")
            with open(threads_checkpoint, 'rb') as f:
                self.threads = pickle.load(f)
            with open(tweets_df_checkpoint, 'rb') as f:
                self.tweets_df = pickle.load(f)
        else:
            # Load llm for embedding
            self._load_model()
            self._parse_all_threads()
            
            if self.use_checkpoints:
                print("Saving threads checkpoint...")
                with open(threads_checkpoint, 'wb') as f:
                    pickle.dump(self.threads, f)
                with open(tweets_df_checkpoint, 'wb') as f:
                    pickle.dump(self.tweets_df, f)
        
        # Build user vectors
        user_vecs_checkpoint = self.checkpoint_dir / "user_vecs.pkl"
        if self.use_checkpoints and user_vecs_checkpoint.exists():
            print("Loading user vectors from checkpoint...")
            with open(user_vecs_checkpoint, 'rb') as f:
                self.user_vecs = pickle.load(f)
        else:
            self._build_user_vectors()
            
            if self.use_checkpoints:
                print("Saving user vectors checkpoint...")
                with open(user_vecs_checkpoint, 'wb') as f:
                    pickle.dump(self.user_vecs, f)
        
        # Build article sequences
        article_seq_checkpoint = self.checkpoint_dir / "article_seq.pkl"
        labels_checkpoint = self.checkpoint_dir / "labels.pkl"
        
        if self.use_checkpoints and article_seq_checkpoint.exists() and labels_checkpoint.exists():
            print("Loading article sequences from checkpoint...")
            with open(article_seq_checkpoint, 'rb') as f:
                self.article_seq = pickle.load(f)
            with open(labels_checkpoint, 'rb') as f:
                self.labels = pickle.load(f)
        else:
            # Load model if not already loaded
            if self._model is None:
                self._load_model()
                
            self._build_article_sequences()
            
            if self.use_checkpoints:
                print("Saving article sequences checkpoint...")
                with open(article_seq_checkpoint, 'wb') as f:
                    pickle.dump(self.article_seq, f)
                with open(labels_checkpoint, 'wb') as f:
                    pickle.dump(self.labels, f)
        
        # Free up memory
        if self._model is not None:
            del self._model
            self._model = None
            gc.collect()
        
        # Save final cache
        self._save_cache()
    
    def _load_model(self):
        """Load the sentence transformer model only when needed"""
        if self._model is None:
            print("Loading embedding model")
            self._model = SentenceTransformer("all-MiniLM-L6-v2") #self._model = SentenceTransformer("paraphrase-MiniLM-L3-v2")  # smaller model
    
    def article_ids(self):
        return list(self.article_seq.keys())

    def article_sequence(self, art_id):
        """Return (X_seq:list[dict], label:int)"""
        return self.article_seq[art_id], self.labels[art_id]

    def user_feature_matrix(self):
        """Return (list[user_id], np.ndarray[num_users, user_k])"""
        ids = list(self.user_vecs.keys())
        mat = np.vstack([self.user_vecs[u] for u in ids])
        return ids, mat

    def _parse_all_threads(self):
        """Parse all threads in the dataset and organize in a DataFrame"""
        self.tweets = []
        dfs = []
        
        # Parcourir tous les événements
        for event_dir in tqdm(list(self.root.glob("*")), desc="Événements"):
            if not event_dir.is_dir():
                continue
                
            event_name = event_dir.name
            
            # Parcourir rumeurs/non-rumeurs
            for category in ["rumours", "non-rumours"]:
                cat_dir = event_dir / category
                if not cat_dir.exists():
                    continue
                    
                # Parcourir les threads
                i=0
                for thread_dir in cat_dir.glob("*"):
                    if not thread_dir.is_dir():
                        continue
                    if i>5000: #just for testing
                        break
                    i+=1
                    thread_id = thread_dir.name
                    
                    # Chercher le dossier source-tweet
                    source_dir = None
                    for variant in ["source-tweet", "source-tweets", "source_tweet", "source_tweets"]:
                        if (thread_dir / variant).exists():
                            source_dir = thread_dir / variant
                            break
                    
                    if not source_dir:
                        continue
                    
                    # Trouver les fichiers JSON valides (pas de fichiers cachés)
                    source_files = [f for f in source_dir.glob("*.json") if not f.name.startswith('.')]
                    if not source_files:
                        continue
                    
                    source_file = source_files[0]
                    if source_file.stat().st_size == 0:
                        continue
                    
                    # Charger le tweet source
                    with open(source_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if not content:
                            continue
                        
                        src = json.loads(content)
                    
                    # Créer l'ID du thread et déterminer le label
                    unified_thread_id = f"{category}-{thread_id}"
                    label = 1 if category == "rumours" else 0
                    
                    # Extraire les infos utilisateur et timestamp
                    src_uid = src["user"]["id_str"]
                    src_time = tools._to_ts(src["created_at"])
                    
                    # Enregistrer les infos du thread
                    self.threads[unified_thread_id] = {
                        "source_id": src["id_str"],
                        "user_id": src_uid,
                        "time": src_time,
                        "label": label,
                        "veracity": src.get("veracity", None)
                    }
                    
                    # Ajouter le tweet source aux tweets
                    self._add_row_to_tweets(src, unified_thread_id, label, src["id_str"], None)
                    
                    # Traiter les réactions
                    reactions_dir = thread_dir / "reactions"
                    if reactions_dir.exists():
                        for reaction_file in reactions_dir.glob("*.json"):
                            if reaction_file.name.startswith('.') or reaction_file.stat().st_size == 0:
                                continue
                                
                            with open(reaction_file, 'r', encoding='utf-8') as f:
                                content = f.read().strip()
                                if not content:
                                    continue
                                    
                                reply = json.loads(content)
                            
                            self._add_row_to_tweets(
                                reply,
                                unified_thread_id,
                                label,
                                reply["id_str"],
                                reply.get("in_reply_to_status_id_str")
                            )
        
        # Créer un DataFrame à partir des tweets collectés
        if self.tweets:
            chunk_size = 10000
            for i in range(0, len(self.tweets), chunk_size):
                dfs.append(pd.DataFrame(self.tweets[i:i+chunk_size]))
            
            self.tweets_df = pd.concat(dfs) if dfs else pd.DataFrame(columns=[
                "thread", "tweet_id", "parent_id", "user_id", "text", "ts", "label"
            ])
        else:
            self.tweets_df = pd.DataFrame(columns=[
                "thread", "tweet_id", "parent_id", "user_id", "text", "ts", "label"
            ])
        
        # Libérer la mémoire
        self.tweets = None
        gc.collect()
    def _add_row_to_tweets(self, tweet, thread_id, label, tid, parent_id): 
        self.tweets.append({
            "thread": thread_id,
            "tweet_id": tid,
            "parent_id": parent_id,
            "user_id": tweet["user"]["id_str"],
            "text": tweet["text"],
            "ts": tools._to_ts(tweet["created_at"]),
            "label": label
        })


    def _build_user_vectors(self):
        """Build user vector (don't care about nb of reactions only
         if react and in which thread)"""
        print("Building users vectors")
        #create dict with user id as key and index as value
        user_idx = {}
        thd_idx  = {}
        for i,u_id in enumerate(self.tweets_df.user_id.unique()):
            user_idx[u_id] = i
        for i,t_id in enumerate(self.tweets_df.thread.unique()):
            thd_idx[t_id] = i

        print(f"Found {len(user_idx)} users and {len(thd_idx)} threads")
    
        # collect unique user-thread pairs first
        # user/thread pairs in a set to avoid duplicates
        #bc don't care about nb of occurence only about which user in which thread
        user_thread_pairs = set()
        
        # prepare for memory efficiency
        chunk_size = 50000  
        total_rows = len(self.tweets_df)
        
        # collect unique user-thread pairs
        for start in tqdm(
            range(0, total_rows, chunk_size), 
            desc="Collecting unique user-thread pairs"
        ):
            end = min(start + chunk_size, total_rows)
            chunk_df = self.tweets_df.iloc[start:end] #select data for a range of rows
            
            for _ , row in chunk_df.iterrows():
                user_thread_pairs.add((user_idx[row.user_id], thd_idx[row.thread]))
        
        # set to lists for matrix constrct 
        rows = [pair[0] for pair in user_thread_pairs]
        cols = [pair[1] for pair in user_thread_pairs]
        data = np.ones(len(rows))
        
        # build the matrix directly with unique pairs (values only 1 and 0)
        M = csr_matrix(
            (data, (rows, cols)), 
            shape=(len(user_idx), len(thd_idx))
        )
        
        # use scipy fct to save sparse matrix
        if self.use_checkpoints:
            save_npz(self.checkpoint_dir / "user_thread_matrix.npz", M)
        
        print(f"SVD with k={self.user_dim}")

        u, s, vt = svds(M, k=self.user_dim)
        print("SVD completed")
        #now we have a more compressed version of the "rows" that is a vector u[i] representing ith user associeted 
        #to user id user_idx[i]
        for u_id, i in user_idx.items():
            self.user_vecs[u_id] = u[i]


    def _encode_texts_in_batches(self, texts):
        """texts in batches to avoid memory pblm"""
        #to avoid crash of _model.encode
        if not texts:
            return np.zeros(
                (0, self._model.get_sentence_embedding_dimension())
            )
        
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embedding = self._model.encode(batch, show_progress_bar=False)
            embeddings.append(batch_embedding)
            
        return np.vstack(embeddings)

    def _build_article_sequences(self): #here care about nb of reactions of a user in a thread
        """prepare data for capture module using  temporal sequence of user interactions per thread"""

        #container, tweets are either source or reactions and now sort by threads
        thread_groups = self.tweets_df.groupby("thread")
        
        #  threads in batches to avoid memory pblm
        threads = list(thread_groups.groups.keys()) #thread_id: list of row index
        batch_size = 100  # Process 100 threads at a time
        
        for batch_start in tqdm(range(0, len(threads), batch_size), desc="Processing thread batches"):
            batch_end = min(batch_start + batch_size, len(threads))
            batch_threads = threads[batch_start:batch_end]
            
            for thread_id in tqdm(batch_threads, desc="Processing threads", leave=False):
                grp = thread_groups.get_group(thread_id)
                self._process_single_thread(thread_id, grp)
                
            # Free memory after each batch
            gc.collect()

    def _process_single_thread(self, thread_id, grp):
        """single thread to build its sequence"""
        source_time_st = self.threads[thread_id]["time"]
        grp = grp.sort_values("ts")
        #grp.ts is the colomns of time of the tweets
        bins = ( (grp.ts - source_time_st) // (self.bin_h * 3600 )).astype(int)
        #add new colomn to grp for the bin 
        grp = grp.assign(bin=bins)

        last_non_empty = None
        seq = []
        
        for b_idx, bin_df in grp.groupby("bin"):
            eta = len(bin_df)
            if last_non_empty is None:
                delta_t = 0
            else:
                delta_t = b_idx - last_non_empty
            last_non_empty = b_idx

            # process text embeddings in batches, ! bin_df.text.tolist() might be empty if weird text encoding
            x_tau = self._encode_texts_in_batches(bin_df.text.tolist()).mean(axis=0)

            # Process user vectors more efficiently
            user_ids = bin_df.user_id.tolist()
            user_vecs_list = []
            for u_id in user_ids:
                user_vecs_list.append(self.user_vecs[u_id])
            x_u = np.mean(user_vecs_list, axis=0) if user_vecs_list else np.zeros(self.user_dim)

            seq.append({
                "eta": eta,
                "delta_t": delta_t,
                "x_u": x_u,
                "x_tau": x_tau
            })
            
        self.article_seq[thread_id] = seq
        self.labels[thread_id] = self.threads[thread_id]["label"]

    def _save_cache(self):
        with open(self.cache_path, "wb") as f:
            pickle.dump(
                obj={
                    "article_seq": self.article_seq,
                    "labels": self.labels,
                    "user_vecs": self.user_vecs
                },
                file=f
            )

    def _load_cache(self):
        with open(self.cache_path, "rb") as f:
            d = pickle.load(f)
        self.article_seq = d["article_seq"]
        self.labels = d["labels"]
        self.user_vecs = d["user_vecs"]