# pheme_csi_data.py
import rumorProject as RP


import json, pickle, os
from pathlib import Path
from collections import defaultdict, Counter
import networkx as nx
import numpy as np
import pandas as pd
from tqdm.auto import tqdm # for progress bar
from sentence_transformers import SentenceTransformer
from scipy.sparse import csr_matrix, save_npz
from scipy.sparse.linalg import svds
import gc  # for garbage collection
import matplotlib.pyplot as plt

class DataBase:
    
    """Notes:
    tweet = a source tweet OR a reaction
    thread = a source tweet AND all its reactions = article in CSI
    """


    def __init__(
            self, 
            bin_size: int = 1, #in hours
            dim_x_u : int = 20, #20 in CSI paper, None now to check with scree plot
            dim_x_tau : int = None,
            dim_y_i : int = 50, #50 in CSI paper
            save_file_name: str = None
    ):
        self.bin_size = bin_size
        self.dim_x_u = dim_x_u
        self.dim_x_tau = dim_x_tau
        self.dim_y_i = dim_y_i
        self.pheme_dataset_path = Path(RP.config.RAW_DATA_DIR)
        self.save_path = Path(RP.config.CACHE_DATA_DIR) / f"{save_file_name}.pkl"
        self._model = None # SentenceTransformer model
        self.tweets = []
        self.processed_tweet_ids = set()
        self.tweets_df = None
        self.threads_source = {} #dict with key = thread_id and value = dict with source_id, user_id, time, label
        self.threads_seq = {} #dict with key = thread_id and value = list of dict with x_u, x_tau, delta_t, eta as keys
        self.labels = {} #dict with key = thread_id and value = label (0 or 1 for rumor or non-rumor)
        #user vector for capture using threads incidence matrix
        self.user_vecs_global = {} #dict with key = user_id and value = vector of the user,
        #user vector for score using weighted user graph
        self.user_vecs_source = {} #dict with key = user_id and value = vector of the user

        if self.save_path.exists():

            print("Loading precomputed data")
            self._load_precomputed_data()
            print("Data loaded")

        else:


            print("Parse all threads")
            self._parse_all_threads()
            print("Threads parsed")

            print("Build user vectors global")
            self._build_user_vectors_global()
            print("User vectors global built")

            print("Build user vectors source")
            self._build_user_vectors_source()
            print("User vectors source built")

            if input("build article sequences? (y/n)") == "y":

                print("Build article sequences")
                self._build_threads_sequences()
                print("Article sequences built")

    def _parse_all_threads(self):
        """ need to parse all the json (each a tweet), each already grouped by
          rumour->thread->source/or_reaction -> json. as described in build user
          vector and build thread seq we need a dataframe with all the tweets and 
          a dict with the source tweet of each thread.
        """
        for source_path in list(self.pheme_dataset_path.glob("*/*/*/source-tweet/*.json")):
            # ex path in pheme: event/rumours/THREAD_ID/source-tweet/THREAD_ID.json
            thread_dir = source_path.parent.parent
            parent = thread_dir.parent.name # "rumours" or "non-rumours"
            root = thread_dir.name # "552783238415265792"
            thread_id = f"{parent}-{root}"# create thing like ex = "rumours-552783238415265792"
            if parent == "rumours":
                label = 1
            else: label = 0

            #create thread dict with source_id, user_id, time, label + add source tweet to self.tweets
            self.deal_with_source_tweet(thread_id, source_path, label)
            
            # parse all reactions in the thread and add them to self.tweets
            self._deal_with_reactions(thread_dir, thread_id, label)


        print(f"Parsed {len(self.tweets)} tweets")
        self.tweets_df = pd.DataFrame(self.tweets)
        print("DataFrame head: ",self.tweets_df.head()) #should be thread tweet_id parent_id user_id text ts label
        print("Types: ",self.tweets_df.dtypes)

        self.tweets = None # free memory
        gc.collect()

        return None
    

    def _build_user_vectors_global(self):
        """ to build them we need a data frame with all the tweets, 1 by line
        each line has at least a thread id, a user id
        To build user vector (don't care about nb of reactions only if react and in which thread) """
        
        #create dict with user id as key and index as value
        user_index = {}
        thread_index  = {}
        #we do that to build incidence matrix each line represent a user and each column a thread
        #that's why we need to associate integer to each
        for i, u_id in enumerate(self.tweets_df.user_id.unique()): #use unique bc user might be involved in several threads
            user_index[u_id] = i
        for i,t_id in enumerate(self.tweets_df.thread_id.unique()):
            thread_index[t_id] = i

        print(f" {len(user_index)} users and {len(thread_index)} threads")

        # collect unique user-thread pairs first
        # user/thread pairs in a set to avoid duplicates
        #bc don't care about nb of occurence only about which user in which thread
        user_thread_pairs = set() #pair of index
        for _, row in self.tweets_df.iterrows():
            #each row is associated to a user and a thread
            user_thread_pairs.add(
                (user_index[row.user_id], thread_index[row.thread_id]) #look for index associeted to id
            )

        #build coordinate matrix 
        rows = []
        cols = []
        #we get row/cols indices for non zero entries
        for pair in user_thread_pairs:
            rows.append(pair[0])
            cols.append(pair[1])

        data = np.ones(len(rows))
        
        # rows cols data form our matrix, but we need to transform in csr (comrpess the rows list)
        #  for svds function so csr_matrix does rows, cols, data -> ia,ja,data
        Incidence_matrix = csr_matrix(
            (data, (rows, cols)), 
            shape=(len(user_index), len(thread_index))
        )
        print(f"CSR matrix shape: {Incidence_matrix.shape}")
        
        print(f"SVD with k={self.dim_x_u}")

        U, s, vt = svds(Incidence_matrix, k=self.dim_x_u)
        print("SVD completed")

        #now we have a more compressed version of the "rows" that is a vector u[i] representing ith user associeted 
        #to user id user_idx[i]
        for user_id, i in user_index.items():
            self.user_vecs_global[user_id] = U[i]

        return None
    

    

    def _deal_with_reactions(self, thread_dir, thread_id, label):
        # add all reaction to self.tweets
        for rp in (thread_dir / "reactions").glob("*.json"):
            with open(rp) as f:
                reply = json.load(f)
            self._add_row_to_tweets(
                reply, 
                thread_id, 
                label,
                reply["id_str"], # answer id 
                reply.get("in_reply_to_status_id_str") #source
            )

            #verif
            if reply["id_str"] == "552787794503143424":
                print(self.tweets[-1]) #should be associated to 55...92

    def _add_row_to_tweets(self, tweet, thread_id, label, tid, parent_id): 
        if tid in self.processed_tweet_ids:
            return None
        
        self.processed_tweet_ids.add(tid)
        self.tweets.append({
            "thread_id": thread_id,
            "tweet_id": tid,
            "parent_id": parent_id,
            "user_id": tweet["user"]["id_str"],
            "text": tweet["text"],
            "ts": RP.tools._to_ts(tweet["created_at"]),
            "label": label
        })    

    def deal_with_source_tweet(self, thread_id, source_path, label):

        with open(source_path) as f:
            src = json.load(f)
        src_user_id = src["user"]["id_str"]
        src_time = RP.tools._to_ts(src["created_at"])

        self.threads_source[thread_id] = {
            "source_id": src["id_str"], 
            "user_id": src_user_id,
            "time": src_time,
            "label": label
        }

        self._add_row_to_tweets(
                src, thread_id, label, src["id_str"], None
        )
        
        if thread_id == "rumours-552783238415265792":
                print("Thread source example: ",self.threads_source[thread_id])
                print("Tweet source example: ",self.tweets[-1])
        return None
    
    
    def _build_user_vectors_source(self):
        """Build source-level user embeddings yᵢ for the Score module in a manual, transparent style."""
        # 1. map users to indices
        users_list = list(self.tweets_df.user_id.unique())
        user_index = {}
        for i, u in enumerate(users_list):
            user_index[u] = i
        n_users = len(users_list)
        print(f"{n_users} users")

        thread_users = {}
        for _, row in self.tweets_df.iterrows():
            thread_id = row["thread_id"]
            user_id = row["user_id"]
            idx = user_index[user_id]
            if thread_id not in thread_users: #if key not in dict create it
                thread_users[thread_id] = set()
            thread_users[thread_id].add(idx)

        # build weighted graph edges=number of engagment in the same threads
        #create a dict with key = (user1, user2) and value = number of tisame thread
        edge_weights = {}
        for users in thread_users.values(): #don't contain dupplicate 
            users = sorted(users) #to avoid duplicates style (1,2) and (2,1) and stay consistent
            for i in range(len(users)):
                for j in range(i+1, len(users)):# to avoid duplicates also have comparaison between each users
                    pair = (users[i], users[j])#create key
                    if pair in edge_weights:
                        edge_weights[pair] += 1 #involved in the same thread increase connection
                    else:
                        edge_weights[pair] = 1

        # 4. build sparse adjacency matrix
        rows = []
        cols = []
        data = []
        for (ui, uj), weight in edge_weights.items():
            rows.append(ui)
            cols.append(uj)
            data.append(weight)
            #for symmetry
            rows.append(uj)
            cols.append(ui)
            data.append(weight)

        Adjacency_matrix = csr_matrix(
            (data, (rows, cols)), 
            shape=(n_users, n_users)
        )
        print(f"Adj matrix now in csr and shape: {Adjacency_matrix.shape}")

        print(f"SVD with k={self.dim_y_i}")
        U, s, vt = svds(Adjacency_matrix, k=self.dim_y_i)
       

        for user_id, idx in user_index.items():
            self.user_vecs_source[user_id] = U[idx]
    
    def _build_threads_sequences(self):
        """ to build them we need both a df with all the tweets grouped by thread
        and a dict with representing the source associeted to each thread (need more
        specifically the time of the source tweet)"""
        
        print("Load model")
        self._load_model()
        print("Model loaded")

        #here care about nb of reactions of a user in a thread

        """prepare data for capture module using  temporal sequence of user interactions per thread"""

        #container, tweets are either source or reactions and now sort by threads
        thread_groups = self.tweets_df.groupby("thread_id")
        
        #  threads in batches to avoid memory pblm
        threads = list(thread_groups.groups.keys()) #thread_id: list of thread id
        print("first thread to be processed: ", threads[0])
        for thread_id in tqdm(threads, desc="Processing threads"):
                grp = thread_groups.get_group(thread_id)
                self._process_single_thread(thread_id, grp)


        return None
    

    def _process_single_thread(self, thread_id, grp):

        """single thread to build its sequence"""
        source_time_st = self.threads_source[thread_id]["time"]
        grp = grp.sort_values("ts")
        
        #grp.ts is the colomns of time of the tweets
        bins = ( (grp.ts - source_time_st) // (self.bin_size * 3600 )).astype(int)
        #add new colomn to grp for the bin 
        grp = grp.assign(bin=bins)

        if thread_id == "non-rumours-498235547685756928": #suppose to be the first
            print("head: ", grp.head())
            print("Group bins of non-rumours-498235547685756928:\n", bins)
            print("Source time: ", source_time_st)
            print("Time:\n",grp.ts)

        last_non_empty = None
        seq = []
        if thread_id == "non-rumours-498235547685756928":
            print(grp.groupby("bin").head())

        for b_idx, bin_df in grp.groupby("bin"): #bin_df is just a table of the tweet of a given thread in order of bin, subdivised in bin
            eta = len(bin_df)
            if last_non_empty is None:
                delta_t = 0
            else:
                delta_t = b_idx - last_non_empty
            last_non_empty = b_idx #ensure delta t is the time between the last non empty bin and the current one

            # process text embeddings in batches, ! bin_df.text.tolist() might be empty if weird text encoding
            
            #x_tau = self._encode_texts_in_batches(bin_df.text.tolist()).mean(axis=0)
            x_tau = self._model.encode(bin_df.text.tolist(), show_progress_bar=False).mean(axis=0)
            
            
            # harvesrt user id of user present in the bin to make a mean of the user 
            #vector and have a global representation of the users in this bin
            user_ids = bin_df.user_id.tolist()
            user_vecs_global_list = []
            for u_id in user_ids:
                user_vecs_global_list.append(self.user_vecs_global[u_id])
                
            if user_vecs_global_list:
                x_u = np.mean(user_vecs_global_list, axis=0) 
            else:
                x_u = np.zeros(self.dim_x_u)
            seq.append({
                "eta": eta,
                "delta_t": delta_t,
                "x_u": x_u, 
                "x_tau": x_tau
            })

        if thread_id == "non-rumours-498235547685756928" and b_idx == 0:
            print(bin_df.text.tolist()[0:2])
            print("x_tau shape: ", x_tau.shape)
            print("x_tau: ", x_tau)
        self.threads_seq[thread_id] = seq
        self.labels[thread_id] = self.threads_source[thread_id]["label"]


    def _load_model(self):
        """load model in ram only when needed, will be used in _build_threads_sequences
        transform text to vector of dimension 384"""
        if self._model is None:
            self._model = SentenceTransformer("all-MiniLM-L6-v2")


    def _load_precomputed_data(self):

        with open(self.save_path, "rb") as f:
            d = pickle.load(f)  #load a dictionary wi
        self.threads_seq = d["threads_seq"] 
        self.labels = d["labels"]
        self.user_vecs = d["user_vecs_global"]
        self.user_vecs_source = d["user_vecs_source"]


    def save_precomputed_data(self):
        """save the data """
        with open(self.save_path, "wb") as f:
            pickle.dump(
                obj={
                    "threads_seq": self.threads_seq,
                    "labels": self.labels,
                    "user_vecs_global": self.user_vecs_global,
                    "user_vecs_source": self.user_vecs_source
                },
                file=f
            )
        print("Data saved in ",self.save_path)
    


        

class FirstDataBase:
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
            use_checkpoints: bool = True,
            build_seq: bool = False
    ):
        self.build_seq = build_seq
        self.root = Path(RP.config.RAW_DATA_DIR)
        self.bin_h = bin_size
        self.user_dim = user_svd_dim
        self.batch_size = batch_size
        self.use_checkpoints = use_checkpoints
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(RP.config.CACHE_DATA_DIR) / "checkpoints"
        self.cache_path = Path(RP.config.CACHE_DATA_DIR) / \
            f"uDim_{user_svd_dim}_binSize_{bin_size}.pkl"
        self.tweets_df = None
        self.tweets = []          #(check _add_row_to_tweets to understand)
        self.threads = {}  # not containing answers, only time reactions
        self.article_seq = {}
        self.labels = {}
        self.user_vecs = None #dict u_id : vector created in _build_user_vectors
        self._model = None # SentenceTransformer model

        if self.cache_path.exists():
            self._load_cache()
        else:
            if input("process data? (y/n)") == "y":
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
                
            if self.build_seq: self._build_article_sequences()
            
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

    def _parse_all_threads(self): #remember each thread as an id and is composed of reactions each having their id
        #for progress bar , recolt all json files
        #tqdm is for progress bar
        print("Parsing all threads")
        pb = tqdm(list(self.root.glob("*/*/*/source-tweet/*.json")), desc="Threads") 
        for src_path in pb:
            # ex path in pheme: event/rumours/THREAD_ID/source-tweet/THREAD_ID.json
            thread_dir = src_path.parent.parent
            thread_id = "-".join(thread_dir.parts[-2:]) # create thing like ex = "rumours-552783238415265792"
            if "rumours" in thread_dir.parts:
                label = 1
            else: label = 0 
            # bc parts = tuple ex: ('pheme-rnr-dataset', 'charliehebdo', 'rumours', '552783238415265792')

            with open(src_path) as f:
                src = json.load(f)
            src_uid = src["user"]["id_str"]
            src_time = RP.tools._to_ts(src["created_at"])

            # source_id is about the id of tweet source but user_id is about the id of the user that posted the tweet source
            self.threads[thread_id] = {
                "source_id": src["id_str"], 
                "user_id": src_uid,
                "time": src_time,
                "label": label,
                "veracity": src.get("veracity", None) #not all have veracity (maybe remove bc unsusefull?)
            }
            #each line in tweets is a either a tweet source or a reaction
            self._add_row_to_tweets(src, thread_id, label, src["id_str"], None)
            #print(self.threads[thread_id])
            #exit(0)
            # add all reaction to self.tweets
            for rp in (thread_dir / "reactions").glob("*.json"):
                with open(rp) as f:
                    reply = json.load(f)
                self._add_row_to_tweets(
                    reply, 
                    thread_id, 
                    label,
                    reply["id_str"], # answer id 
                    reply.get("in_reply_to_status_id_str") #source
                )
                
        # Df in chunks to reduce memory use
        size = 10000
        dfs = []
        for i in range(0, len(self.tweets), size):
            dfs.append(pd.DataFrame(self.tweets[i:i+size]))
        
        self.tweets_df = pd.concat(dfs)
        # free
        self.tweets = None
        gc.collect()

    def _add_row_to_tweets(self, tweet, thread_id, label, tid, parent_id): 
        if tid in self.processed_tweet_ids:
            return None
            
        self.processed_tweet_ids.add(tid)
            
        self.tweets.append({
            "thread": thread_id,
            "tweet_id": tid,
            "parent_id": parent_id,
            "user_id": tweet["user"]["id_str"],
            "text": tweet["text"],
            "ts": RP.tools._to_ts(tweet["created_at"]),
            "label": label
        })
        if thread_id == "rumours-552783238415265792":
            print(self.tweets[-1])
        
        return None


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
        print(M)
        exit(0)
        
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