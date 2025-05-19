# pheme_csi_data.py
import rumorProject as RP

import torch
from torch.utils.data import Dataset
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
from sklearn.random_projection import GaussianRandomProjection
import gc  # for garbage collection
import matplotlib.pyplot as plt

class DataBase:
    
    """

    End-to-end loader + pre-processor for the PHEME rumour dataset,
    producing CSI-ready tensors.

    ex of use:
        data = DataBase(1, 50, save_file_name="rumor_data_jules")

        user_vecs_for_capture = data.user_vecs_global
        user_vecs_for_score = data.user_vecs_source
        threads_seq = data.threads_seq


        Notes:
        tweet = a source tweet OR a reaction
        thread = a source tweet AND all its reactions = article in CSI
    """


    def __init__(
            self, 
            bin_size: int = 1, #in hours
            dim_x_u : int = 20, #20 in CSI paper, None now to check with scree plot
            dim_y_i : int = 50, #50 in CSI paper
            dim_x_tau: int = 100,
            save_file_name: str = None
    ):
        self.bin_size = bin_size
        self.dim_x_u = dim_x_u
        self.dim_y_i = dim_y_i
        self.dim_x_tau = dim_x_tau
        self.pheme_dataset_path = Path(RP.config.RAW_DATA_DIR)
        self.save_path = Path(RP.config.CACHE_DATA_DIR) / f"{save_file_name}.pkl"
        self._model = None # SentenceTransformer model
        # random‑projection matrix for text‑embedding compression (created lazily)
        self._rp_matrix = None
        self.tweets = []
        self.processed_tweet_ids = set()
        self.tweets_df = None
        self.threads_source = {} #dict with key = thread_id and value = dict with source_id, user_id, time, label
        self.threads_seq = {} #dict with key = thread_id and value = list of dict with x_u, x_tau, delta_t, eta as keys
        self.T = None #max sequence length for training
        self.labels = {} #dict with key = thread_id and value = label (0 or 1 for rumor or non-rumor)
        #user vector for capture using threads incidence matrix
        self.user_vecs_global = {} #dict with key = user_id and value = vector of the user,
        #user vector for score using weighted user graph
        self.user_vecs_source = {} #dict with key = user_id and value = vector of the user

        self.initilize_data()
        


    def initilize_data(self):
        if self.save_path.exists():

            print("Loading precomputed data")
            self._load_precomputed_data()
            print("Data loaded")


            print("Parse all threads")
            self._parse_all_threads()
            print("Threads parsed")

        else:

            print("Parse all threads")
            self._parse_all_threads()
            print("Threads parsed")

            print("Precompute embeddings")
            self._precompute_embeddings()
            print("Embeddings precomputed")

            print("Build user vectors global")
            self._build_user_vectors_global()
            print("User vectors global built")

            print("Build user vectors source")
            self._build_user_vectors_source()
            print("User vectors source built")

            
            print("Build article sequences")
            self._build_threads_sequences()
            print("Article sequences built")

            #del self.tweets_df
            gc.collect()
            print("Free memory")


            print("Save precomputed data")
            self.save_precomputed_data()
            print("Data saved")

    def _parse_all_threads(self):
        """ need to parse all the json (each a tweet), each already grouped 
        (in directories) by the structure rumour->thread->source/or_reaction -> .json
        As described in build_user_vector_source/gloval and build_thread_seq,
        we need a dataframe with all the tweets and a dict with the info source
        tweet of each thread"""
        
        #loop on all json files
        for source_path in list(self.pheme_dataset_path.glob("*/*/*/source-tweet/*.json")):
            # ex path in pheme: event/rumours/THREAD_ID/source-tweet/THREAD_ID.json
            thread_dir = source_path.parent.parent
            parent = thread_dir.parent.name # "rumours" or "non-rumours"
            root = thread_dir.name # "552783238415265792"
            thread_id = f"{parent}-{root}"# create thing like ex = "rumours-552783238415265792"
            if parent == "rumours":
                label = 1
            else: 
                label = 0

            #create threads_source dict with source_id, user_id, time, label + add source tweet to self.tweets
            self.deal_with_source_tweet(thread_id, source_path, label)
            
            # parse all reactions in the thread and add them to self.tweets
            self._deal_with_reactions(thread_dir, thread_id, label)

        print(f"Parsed {len(self.tweets)} tweets")

        #transform self.tweets to a dataframe for easier grouping
        self.tweets_df = pd.DataFrame(self.tweets)

        print("DataFrame head: ",self.tweets_df.head()) #should be thread tweet_id parent_id user_id text ts label
        print("Types: ",self.tweets_df.dtypes)

        self.tweets = None # free memory
        gc.collect()

        return None
    

    def _build_user_vectors_global(self):
        """ to build them we need a data frame with all the tweets, 1 by line
        each line has at least a thread id, a user id
        To build user vector (don't care about nb of reactions only if
        react and in which thread """
        
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

        #now we have a more compressed version of the "rows" that is a vector
        # u[i] representing ith user associeted to user id user_idx[i]
        for user_id, i in user_index.items():
            self.user_vecs_global[user_id] = U[i]

        return None
    

    def _deal_with_reactions(self, thread_dir, thread_id, label):
        """ add all reactions to self.tweets"""
        #parse all reactions in the thread and add them to self.tweets
        #loop on all json files
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
        """ add a row to self.tweets, if the tweet isn't already in self.tweets""" 
        if tid in self.processed_tweet_ids:
            return None
        
        #add the id to the set to avoid duplicates
        self.processed_tweet_ids.add(tid)

        self.tweets.append({
            "thread_id": thread_id,
            "tweet_id": tid,
            "parent_id": parent_id,
            "user_id": tweet["user"]["id_str"],
            "text": tweet["text"],
            "ts": RP.tools._to_ts(tweet["created_at"]),#from string to timestamp int
            "label": label
        })   

        return None 

    def deal_with_source_tweet(self, thread_id, source_path, label):
        """ add info of the tweet to thread_source dict and add 
        the tweet to self.tweets"""
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
        """we build source-level user vector yi for the score module
        based on weighted graph of common thread engament between user"""
        # 1. map users to indices
        user_index = {}
        for i, u in enumerate(self.tweets_df.user_id.unique()):
            user_index[u] = i
        n_users = len(user_index)
        print(f"{n_users} users")

        #create a dict for grouping the user involved in a thread, for each thread
        #we will use this to build the weighted graph
        thread_users = {}
        for _, row in self.tweets_df.iterrows():
            thread_id = row["thread_id"]
            user_id = row["user_id"]
            if thread_id not in thread_users: #if key not in dict create it
                thread_users[thread_id] = set() #to avoid dupplicates
            thread_users[thread_id].add(user_index[user_id])

        # build weighted graph edges=number of engagment in the same threads
        #create a dict with key = (user1, user2) and value = number of common threads involved in
        edge_weights = {}
        for users_of_the_thread in thread_users.values(): #don't contain dupplicate 
            users_of_the_thread = sorted(users_of_the_thread) #to avoid duplicates style (1,2) and (2,1) and stay consistent
            for i in range(len(users_of_the_thread)):
                for j in range(i+1, len(users_of_the_thread)):# to avoid duplicates also have comparaison between each users
                    pair = (users_of_the_thread[i], users_of_the_thread[j]) #create key
                    #if key exist add one or create it
                    if pair in edge_weights:
                        edge_weights[pair] += 1 #involved in the same thread increase connection
                    else:
                        edge_weights[pair] = 1

        # build sparse adjacency matrix
        #start by building coordinate matrix
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

        #from coord matrix to csr
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
        specifically the time of the source tweet)
        here care about nb of reactions of a user in a thread
        prepare data for capture module using  temporal sequence
        of user interactions per thread"""
        

        #container, tweets are either source or reactions and now sort by threads
        thread_groups = self.tweets_df.groupby("thread_id")
        thread_sizes = thread_groups.size()

        print(f"Average tweets per thread: {thread_sizes.mean():.2f}")
        print(f"Max tweets in a thread: {thread_sizes.max()}")


        threads_id_list = list(thread_groups.groups.keys()) #thread_id: list of thread id
        print("first thread to be processed: ", threads_id_list[0])
        for thread_id in tqdm(threads_id_list, desc="Processing threads"):
                
                grp = thread_groups.get_group(thread_id)
                self._process_single_thread(thread_id, grp)

        lengths = [len(seq) for seq in self.threads_seq.values()]
        self.T = max(lengths)
        print("Max sequence length: ", self.T)
        print("Min sequence length: ", min(lengths))
        print("Mean sequence length: ", np.mean(lengths))
        print("Std sequence length: ", np.std(lengths))


        return None
    
    def _precompute_embeddings(self, batch_size: int = 128):
        """Embed every tweet text once and add the 384 dim vector
         to the new colomn embed with tweets_df['embed']."""
        
        #load llm for embeding
        self._load_model()
        #list of all the tweet text
        texts = self.tweets_df["text"].tolist()
        all_vecs = [] #each elem will be an array of size batch_size with each element a vector of dim 384
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding tweets"):
            batch = texts[i : i + batch_size] #if i + batch_size > len(texts) doesn't go out of range
            batch_embeded = self._model.encode(
                    batch, convert_to_numpy=True, show_progress_bar=False
                )
            all_vecs.append(batch_embeded)

        #flat everything to have 
        all_vecs = np.vstack(all_vecs)

        print("Shape of all_vecs: ", all_vecs.shape)#suppose to be (Nb of tweets,384)

        
        #from dim 384 to dim 100 using random projection (preserve pair wise distances)
        grp = GaussianRandomProjection(
            n_components=self.dim_x_tau,   # 100
            random_state=RP.config.SEED_NP
        )
        reduced_vecs = grp.fit_transform(all_vecs)

        print("Shape of reduced_vecs: ", reduced_vecs.shape)
        self.tweets_df["embed"] = list(reduced_vecs)    # one ndarray per row

        #verif
        print("head of tweets_df: ", self.tweets_df.head())
        print("tail of tweets_df: ", self.tweets_df.tail())

        # the model is no longer needed can be removed from ram
        del self._model
        torch.mps.empty_cache()
        gc.collect()

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
            
            #x_tau = self._model.encode(bin_df.text.tolist(), show_progress_bar=False).mean(axis=0)
            x_tau = np.mean(bin_df.embed.tolist(), axis=0) #use the precomputed embedding
            
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
            device = torch.device("mps")
            #verifier pretained
            self._model = SentenceTransformer("all-MiniLM-L6-v2").to(device)


    def _load_precomputed_data(self):

        with open(self.save_path, "rb") as f:
            d = pickle.load(f)  #load a dictionary wi
        self.threads_seq = d["threads_seq"] 
        self.labels = d["labels"]
        self.user_vecs_global = d["user_vecs_global"]
        self.user_vecs_source = d["user_vecs_source"]


        lengths = [len(seq) for seq in self.threads_seq.values()]
        self.T = max(lengths)
        print("Max sequence length: ", self.T)
        print("Min sequence length: ", min(lengths))
        print("Mean sequence length: ", np.mean(lengths))
        print("Std sequence length: ", np.std(lengths))
        #RP.tools.plot_sequence_length_distribution(lengths)
        
        


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


####ajout de antoine adapté au nouveaux noms d'attributs
    def article_ids(self):
        return list(self.threads_seq.keys())

    def article_sequence(self, art_id):
        """Return (X_seq:list[dict], label:int)"""
        return self.threads_seq[art_id], self.labels[art_id]

    def user_feature_matrix(self):
        """Return (list[user_id], np.ndarray[num_users, user_k])"""
        ids = list(self.user_vecs_source.keys())
        mat = np.vstack([self.user_vecs_source[u] for u in ids])
        return ids, mat





class RumorDataset(Dataset):
    def __init__(self, lengths, article_features, user_features, labels, user_article_mask, device):
        self.article_features = article_features
        self.user_features = user_features
        self.labels = labels
        self.user_article_mask = user_article_mask.to(device)
        self.lengths = lengths
    def __len__(self):
        return len(self.article_features)
    
    def __getitem__(self, idx):  
        return {
            'article_features': self.article_features[idx],
            #'user_features': self.user_features, because it's the same for all
            'labels': self.labels[idx],
            'user_article_mask': self.user_article_mask[idx],
            'lengths': self.lengths[idx]    
        }
    
