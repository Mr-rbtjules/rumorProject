# pheme_csi_data.py
import rumorProject as RP
import datetime as dt

import torch
from torch.utils.data import Dataset
import json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm # for progress bar
from sentence_transformers import SentenceTransformer
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.random_projection import GaussianRandomProjection
import gc  # for garbage collection


torch.manual_seed(RP.config.SEED_TORCH)  
np.random.seed(RP.config.SEED_NP)  

class DataBase:
    
    """

    loader + pre-processor for the PHEME rumour dataset,
    producing CSI-ready tensors (normalizaiton non included)

    ex of use:
        data = DataBase(1, 50, save_file_name="rumor_data_jules")

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
            save_file_name: str = None,
            device: torch.device = None
    ):
        self.bin_size = bin_size
        self.dim_x_u = dim_x_u
        self.dim_y_i = dim_y_i
        self.dim_x_tau = dim_x_tau
        self.pheme_dataset_path = Path(RP.config.RAW_DATA_DIR)
        if save_file_name:
            self.save_path = Path(RP.config.CACHE_DATA_DIR)\
                            / f"{save_file_name}.pkl"
        else:
            self.save_path = Path(RP.config.CACHE_DATA_DIR)\
                            / "precomputed_pheme_data.pkl"
        self._model = None # SentenceTransformer model
        
        self.tweets = []
        self.processed_tweet_ids = set() #to collect unique tweets
        self.tweets_df = None
        self.threads_source = {} #dict with key = thread_id and value = dict with source_id, user_id, time, label
        self.threads_seq = {} #dict with key = thread_id and value = list of dict with x_u, x_tau, delta_t, eta as keys
        self.T = None #max sequence length for training
        self.labels = {} #dict with key = thread_id and value = label (0 or 1 for rumor or non-rumor)
        #user vector for capture using threads incidence matrix
        self.user_vecs_global = {} #dict with key = user_id and value = vector of the user,
        #user vector for score using weighted user graph
        self.user_vecs_source = {} #dict with key = user_id and value = vector of the user
        self.user_article_mask = None
        self.article_user_idxs = None
        self.device = device
        self.frac = 1
        self.database_name = "PHEME"
        self.initilize_data()
        


    def initilize_data(self):
        if self.save_path.exists():

            print("Loading precomputed data")
            self._load_precomputed_data()
            print("Data loaded")

            #build per-article user-index lists for subset-only scoring
            #coalesce(): merge duplicate entries and finalize the sparse tensor
            mask = self.user_article_mask.coalesce()
            # Print basic info about the sparse mask
            print(f"[Init] user_article_mask shape: {self.user_article_mask.shape}, "
                f"non-zero entries: {mask._nnz()}")

            #number of articles for correct sizing
            n_articles = self.user_article_mask.size(0)

            #indices(): returns a 2×nnz LongTensor row0=article_idx, row1=user_idx for each engagement
            thread_idxs, user_idxs = mask.indices()

            #Prepare: a Python list where each element is the list of engaged user indices for that article
            article_user_idxs = [[] for _ in range(n_articles)]
            for t, u in zip(thread_idxs.tolist(), user_idxs.tolist()):#iterate at the 2 list
                article_user_idxs[t].append(u)

            # Convert each list into a 1D LongTensor
            self.article_user_idxs = [torch.tensor(lst, dtype=torch.long)
                                    for lst in article_user_idxs]
            del self.user_article_mask  # free memory
            # Print summary of article_user_idxs
            print(f"Built article_user_idxs for {len(self.article_user_idxs)} articles.")
            print(f"Example: article 0 has {len(self.article_user_idxs[0])} engaged users.")


        else:

            print("Parse all threads")
            self._parse_all_threads()
            print("Threads parsed")

            print("Precompute embeddings")
            self._precompute_embeddings()
            print("Embeddings precomputed")

            print("Build user vectors")
            self._build_user_vectors()
            print("User vectors built")

            
            print("Build article sequences")
            self._build_threads_sequences()
            print("Article sequences built")

            print("Create user-article mask")
            self.create_user_article_mask()
            print("User-article mask created")

            #build per-article user-index lists for subset-only scoring
            #coalesce(): merge duplicate entries and finalize the sparse tensor
            mask = self.user_article_mask.coalesce()
            # Print basic info about the sparse mask
            print(f"[Init] user_article_mask shape: {self.user_article_mask.shape}, "
                f"non-zero entries: {mask._nnz()}")

            #number of articles for correct sizing
            n_articles = self.user_article_mask.size(0)

            #indices(): returns a 2×nnz LongTensor row0=article_idx, row1=user_idx for each engagement
            thread_idxs, user_idxs = mask.indices()

            #Prepare: a Python list where each element is the list of engaged user indices for that article
            article_user_idxs = [[] for _ in range(n_articles)]
            for t, u in zip(thread_idxs.tolist(), user_idxs.tolist()):#iterate at the 2 list
                article_user_idxs[t].append(u)

            # Convert each list into a 1D LongTensor
            self.article_user_idxs = [torch.tensor(lst, dtype=torch.long)
                                    for lst in article_user_idxs]
            del self.user_article_mask  # free memory
            # Print summary of article_user_idxs
            print(f"Built article_user_idxs for {len(self.article_user_idxs)} articles.")
            print(f"Example: article 0 has {len(self.article_user_idxs[0])} engaged users.")

            del self.tweets_df
            gc.collect()
            print("Free memory")


            print("Save precomputed data")
            self.save_precomputed_data()
            print("Data saved")

        # Safety check: ensure article_user_idxs is always built
        if self.article_user_idxs is None:
            print("Warning: article_user_idxs was None, rebuilding from user_article_mask")
            if hasattr(self, 'user_article_mask') and self.user_article_mask is not None:
                mask = self.user_article_mask.coalesce()
                n_articles = self.user_article_mask.size(0)
                thread_idxs, user_idxs = mask.indices()
                article_user_idxs = [[] for _ in range(n_articles)]
                for t, u in zip(thread_idxs.tolist(), user_idxs.tolist()):
                    article_user_idxs[t].append(u)
                self.article_user_idxs = [torch.tensor(lst, dtype=torch.long)
                                        for lst in article_user_idxs]
                print(f"Rebuilt article_user_idxs for {len(self.article_user_idxs)} articles.")
            else:
                raise RuntimeError("Cannot build article_user_idxs: user_article_mask is missing")

    def _parse_all_threads(self):
        """ need to parse all the json (each a tweet), each already grouped 
        (in directories) by the structure rumour->thread->source/or_reaction -> .json
        As described in build_user_vector_source/gloval and build_thread_seq,
        we need a dataframe with all the tweets and a dict with the info source
        tweet of each thread"""
        
        #loop on all json files
        for source_path in list(
            self.pheme_dataset_path.glob("*/*/*/source-tweet/*.json")
        ):
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

        self.tweets_df = pd.DataFrame(self.tweets)

        print(f"Parsed {len(self.tweets)} tweets")
        print(f"Number of threads: {len(self.threads_source)}")
        print(f"Number of users: {self.tweets_df['user_id'].nunique()}")
        

        #transform self.tweets to a dataframe for easier grouping
        

        print("DataFrame head: ",self.tweets_df.head()) #should be thread tweet_id parent_id user_id text ts label
        print("Types: ",self.tweets_df.dtypes)

        self.tweets = None # free memory
        gc.collect()

        return None
    

    def _build_user_vectors(self):
        """Build both global and source user vectors via single SVD on the user-thread incidence."""
        # map user and thread ids to indices
        user_ids = self.tweets_df["user_id"].unique()
        thread_ids = self.tweets_df["thread_id"].unique()
        u_idx = {u: i for i, u in enumerate(user_ids)}
        t_idx = {t: i for i, t in enumerate(thread_ids)}

        # build incidence matrix M (users × threads)
        rows = []
        cols = []
        for u, t in zip(self.tweets_df["user_id"], self.tweets_df["thread_id"]):
            rows.append(u_idx[u])
            cols.append(t_idx[t])
        data = np.ones(len(rows), dtype=np.float32)
        M = csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(thread_ids)), dtype=np.float32)

        # compute SVD with k = max(dim_x_u, dim_y_i)
        k = max(self.dim_x_u, self.dim_y_i)
        U, s, Vt = svds(M, k=k)

        # sort singular values and vectors in descending order
        idx = np.argsort(s)[::-1]
        U = U[:, idx]

        # assign global and source user vectors
        for u, i in u_idx.items():
            vec = U[i]
            self.user_vecs_global[u] = vec[: self.dim_x_u]
            self.user_vecs_source[u] = vec[: self.dim_y_i]

    def create_user_article_mask(self):
        """
        Create a sparse mask tensor for user-article engagements. to use it
        in score models, to get the user indices for each article."""
        threads_ids = list(self.threads_seq.keys())
        user_ids = list(self.user_vecs_source.keys())
        thread_idx_map = {tid: idx for idx, tid in enumerate(threads_ids)}
        user_idx_map = {uid: idx for idx, uid in enumerate(user_ids)}
        
        #dataframe with 2 colomn, thread id and user id for all tweets,
        #then remove the duplicates to get unique thread and user pairss
        pairs = self.tweets_df[['thread_id', 'user_id']].drop_duplicates()
        
        # add 2 new colomns idXXXXX ! map will transform the thread_id and user_id to their respective indexes
        pairs['thread_idx'] = pairs['thread_id'].map(thread_idx_map)
        
        pairs['user_idx'] = pairs['user_id'].map(user_idx_map)
        
        #just in case
        pairs = pairs.dropna(subset=['thread_idx', 'user_idx'])
        thread_idxs = pairs['thread_idx'].astype(int).to_numpy()
        user_idxs = pairs['user_idx'].astype(int).to_numpy()
        
        #build sparse tensor
        indices = torch.stack([
            torch.tensor(thread_idxs, dtype=torch.long),
            torch.tensor(user_idxs, dtype=torch.long)
        ]) # [[thread_idx1, thread_idx2, ...], [user_idx1, user_idx2, ...]] for COO matrix
        values = torch.ones(len(thread_idxs), dtype=torch.bool)
        sparse_mask = torch.sparse_coo_tensor(
            indices=indices, 
            values=values,
            size=torch.Size([len(threads_ids), len(user_ids)])
        )
        
        self.user_article_mask = sparse_mask
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
            "ts": self._to_ts(tweet["created_at"]),#from string to timestamp int
            "label": label
        })   

        return None 
    
    def _to_ts(self, tstr):
        """ "Wed Jan 07 11:06:08 +0000 2015" to 1420628768"""
        fmt = "%a %b %d %H:%M:%S %z %Y"
        return int(dt.datetime.strptime(tstr, fmt).timestamp())


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
        #print a check
        if thread_id == "rumours-552783238415265792":
                print("Thread source example: ",self.threads_source[thread_id])
                print("Tweet source example: ",self.tweets[-1])

        return None
    
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
        for i in tqdm(
            range(0, len(texts), batch_size), 
            desc="Encoding tweets"
        ):
            batch = texts[i : i + batch_size] #if i + batch_size > len(texts) doesn't go out of range
            batch_embeded = self._model.encode(
                    batch, convert_to_numpy=True, show_progress_bar=False
                )
            all_vecs.append(batch_embeded)

        #flat everything to have 
        all_vecs = np.vstack(all_vecs)

        print("Shape of all_vecs: ", all_vecs.shape)#suppose to be (Nb of tweets,384)

        #from dim 384 to dim 100 using random projection (preserve pair wise distances)
        #based on johnson lindenstrauss lemme
        """wiki:The lemma states that a set of points in a high-dimensional space can
        be embedded into a space of much lower dimension in such a way that distances
        between the points are nearly preserved. In the classical proof of the lemma,
        the embedding is a random (Normal dist) orthogonal projection. """
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
        """single thread to build its sequence , so we build here 
        delta_t, eta, x_u, x_tau"""
        #get source time , select the tweet associated to the thread_id
        #and sort them by time
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

        
        if thread_id == "non-rumours-498235547685756928":
            print(grp.groupby("bin").head())

        last_non_empty = None
        seq = []
        for b_idx, bin_df in grp.groupby("bin"): #bin_df is just a table of the tweet of a given thread in order of bin, subdivised in bin
            eta = len(bin_df)
            if last_non_empty is None:
                delta_t = 0
            else:
                delta_t = b_idx - last_non_empty
            last_non_empty = b_idx #ensure delta t is the time between the last non empty bin and the current one

            # process text embeddings in batches, ! bin_df.text.tolist() might be empty if weird text encoding
            
            
            x_tau = np.mean(bin_df.embed.tolist(), axis=0) #use the precomputed embedding
            
            #harvesrt user id of user present in the bin to make a mean of the user 
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
        """load model for embding (create x_tau at the end)
        in ram only when needed, will be used in _build_threads_sequences
        transform text to vector of dimension 384"""
        if self._model is None:
            device = torch.device("mps")
            #verifier pretained
            #self._model = SentenceTransformer("all-mpnet-base-v2").to(device)
            self._model = SentenceTransformer("all-MiniLM-L6-v2").to(device)


    def _load_precomputed_data(self):
        """if we precomputed the data before, we can load it"""
        with open(self.save_path, "rb") as f:
            d = pickle.load(f)  #load a dictionary wi
        self.threads_seq = d["threads_seq"] 
        self.labels = d["labels"]
        self.user_vecs_global = d["user_vecs_global"]
        self.user_vecs_source = d["user_vecs_source"]
        self.user_article_mask = d["user_article_mask"]
        self.threads_source = d["threads_source"]


        lengths = [len(seq) for seq in self.threads_seq.values()]
        self.T = max(lengths)
        print("Max sequence length: ", self.T)
        print("Min sequence length: ", min(lengths))
        print("Mean sequence length: ", np.mean(lengths))
        print("Std sequence length: ", np.std(lengths))
        
    def save_precomputed_data(self):
        """save the data """
        with open(self.save_path, "wb") as f:
            pickle.dump(
                obj={
                    "threads_seq": self.threads_seq,
                    "labels": self.labels,
                    "user_vecs_global": self.user_vecs_global,
                    "user_vecs_source": self.user_vecs_source, #dictionnaire cle = user_id, value = vector
                    "user_article_mask": self.user_article_mask,
                    "threads_source": self.threads_source
                },
                file=f
            )
        print("Data saved in ",self.save_path)

    #ajout d'antoine pour y acceder depuis trainer
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
    def __init__(
            self, lengths, article_features, user_features, 
            labels, article_user_idxs, device
    ):
        self.article_features    = article_features
        self.user_features       = user_features
        self.labels              = labels
        # New: store the per-article list of engaged-user indices
        self.article_user_idxs   = article_user_idxs
        self.lengths             = lengths

    def __getitem__(self, idx): #idx associated to the article/thread id
        # Fetch indices of engaged users for this article
        idxs = self.article_user_idxs[idx]
        # Gather their feature vectors (n_engaged x dim_y_i)
        y_is = self.user_features[idxs]
        return {
            'article_features': self.article_features[idx],
            'labels':          self.labels[idx],
            # y_is: tensor of shape (n_engaged, dim_y_i)
            'y_is':            y_is,
            'lengths':         self.lengths[idx]
        }

    def __len__(self):
        # Return the number of articles in the dataset
        return len(self.labels)
    
def rumor_collate(batch):
    """create a custom function to collate the data in a batch
    especially for the y_is which are not of the same size"""
    # batch: list of dicts
    article_feats = torch.stack([b['article_features'] for b in batch])
    lengths       = torch.tensor([b['lengths']       for b in batch], dtype=torch.long)
    labels        = torch.tensor([b['labels']        for b in batch], dtype=torch.long)

    """in a batch the y_is aren't of the same size, so we need to flatten 
    them and keep track of the source index for each y_is once it is flattened
    """
    y_blocks, src_index = [], []
    for art_idx, b in enumerate(batch):
        y_is = b['y_is']                      # (n_i, dim_y)
        y_blocks.append(y_is)
        src_index.append(
            torch.full((len(y_is),), art_idx, dtype=torch.long)
        )
    y_flat  = torch.cat(y_blocks, 0)         # (N, dim_y)
    src_idx = torch.cat(src_index, 0)        # (N,)
    
    return {
        'article_features': article_feats,
        'lengths': lengths,
        'labels': labels,
        'y_flat': y_flat,  # (N, dim_y)
        'src_idx': src_idx   # (N,)
    }