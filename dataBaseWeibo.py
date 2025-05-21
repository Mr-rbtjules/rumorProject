# weibo_database.py
"""
Light-weight replica of database.py, but for the Weibo Rumdect dataset.

🔹 It re-uses the loader you built in parseTest.py – so import order matters:
    1. parseTest.py runs, builds `df`, `events`, `threads_source` …
    2. We import those objects here and convert them to the structures
       the original DataBase class expected.

🔹 Main public attributes after instantiation
    ├─ self.threads_seq        dict[thread_id] -> list of time-binned dicts
    ├─ self.labels             dict[thread_id] -> 0/1
    ├─ self.user_vecs_global   dict[user_id]   -> np.ndarray(dim_x_u,)
    ├─ self.user_vecs_source   dict[user_id]   -> np.ndarray(dim_y_i,)
    └─ self.T                  max sequence length (for padding)

Usage
------
>>> from weibo_database import WeiboDataBase
>>> data = WeiboDataBase(bin_size=1, dim_x_u=20, dim_y_i=50, dim_x_tau=100)
>>> X_seq, label = data.article_sequence('1')   # example
"""
import math, random
import rumorProject as RP
from pathlib import Path
from collections import Counter, defaultdict
import math, random, gc, sys, json
import pickle
from collections import Counter

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.random_projection import GaussianRandomProjection
from tqdm.auto import tqdm

from sentence_transformers import SentenceTransformer
import torch



# ----------------------------------------------------------------------
# main class
# ----------------------------------------------------------------------
class WeiboDataBase:
    def __init__(
            self,
            bin_size: int = 1,
            dim_x_u : int = 20,
            dim_y_i : int = 50,
            dim_x_tau: int = 100,
            event_fraction: float = 0.2,
            save_df_file_name: str = None,
            save_precomputed_file_name: str = None,
            device: torch.device = None
    ):
        self.bin_size   = bin_size
        self.dim_x_u    = dim_x_u
        self.dim_y_i    = dim_y_i
        self.dim_x_tau  = dim_x_tau
        self.event_fraction = event_fraction
        self.save_df_file_name = save_df_file_name
        self.save_precomputed_file_name = save_precomputed_file_name

        df_base = save_df_file_name if save_df_file_name else "weibo_df"
        self.save_df_path = Path(RP.config.DATA_EXT_DIR) / f"{df_base}.pkl"
        pre_base = save_precomputed_file_name if save_precomputed_file_name else "weibo_precomputed"
        self.precomputed_data_path = Path(RP.config.DATA_EXT_DIR) / f"{pre_base}.pkl"

        # placeholders
        self.tweets_df = None
        self.events = None
        self.threads_source   = {}
        self.threads_seq      = {}
        self.labels           = {}
        self.user_vecs_global = {}
        self.user_vecs_source = {}
        self.T                = 0
        self.device = device
        random.seed(RP.config.SEED_RAND)

        self.initilize_data()

    def initilize_data(self):
        # DataFrame + events cache
        if self.save_df_path.exists():
            print("Loading cached DataFrame")
            with open(self.save_df_path, "rb") as f:
                self.tweets_df, self.events = pickle.load(f)
            print("Cached DataFrame loaded")
            print(f"DataFrame shape: {self.tweets_df.shape}, events: {len(self.events)}")
        else:
            print("Parse all threads")
            self.tweets_df, self.events = self._parse_all_threads(event_fraction=self.event_fraction)
            print("Caching parsed DataFrame")
            print(f"DataFrame parsed: {self.tweets_df.shape}, events: {len(self.events)}")
            with open(self.save_df_path, "wb") as f:
                pickle.dump((self.tweets_df, self.events), f)

        # After extracting threads_source, make events equivalent
        self._extract_thread_sources()
        self.events = self.threads_source

        if self.precomputed_data_path.exists():
            print("Loading precomputed data")
            with open(self.precomputed_data_path, "rb") as f:
                d = pickle.load(f)
            self.threads_seq       = d["threads_seq"]
            self.labels            = d["labels"]
            self.user_vecs_global  = d["user_vecs_global"]
            self.user_vecs_source  = d["user_vecs_source"]
            self.T = max(len(seq) for seq in self.threads_seq.values())
            print("Data loaded")
            print(f"Precomputed cache: {len(self.threads_seq)} threads; "
                  f"global vectors for {len(self.user_vecs_global)} users; "
                  f"source vectors for {len(self.user_vecs_source)} users; "
                  f"max sequence length {self.T}")
            return

        print("Build user vectors")
        self._build_user_vectors()
        print("User vectors built")

        print("Precompute embeddings")
        self._precompute_embeddings()
        print("Embeddings precomputed")

        print("Build article sequences")
        self._build_threads_sequences()
        print("Article sequences built")

        print("Free memory")
        gc.collect()

        print("Save precomputed data")
        self.save_precomputed_data()
        print("Data saved")


    # ------------------------------------------------------------------
    # 1. Source-post table  (threads_source & labels)
    # ------------------------------------------------------------------
    def _extract_thread_sources(self):
        src_rows = self.tweets_df[self.tweets_df["parent_id"].isna()]      # root posts
        for _, row in src_rows.iterrows():
            tid = row["thread_id"]
            self.threads_source[tid] = {
                "source_id": row["tweet_id"],
                "user_id"  : row["user_id"],
                "time"     : int(row["ts"]),
                "label"    : self.events[tid]["label"]
            }
            self.labels[tid] = self.events[tid]["label"]


    def _parse_all_threads(self, event_fraction: float = 0.2):
        """
        Replicates the data‑loading logic that previously lived in parseTest.py.

        Parameters
        ----------
        event_fraction : float
            Fraction of events to load into memory (0 < x ≤ 1).  A smaller
            value speeds up experimentation on limited hardware.

        Returns
        -------
        df : pd.DataFrame
            DataFrame with one row per post plus an 'event_id' column.
        events : dict
            Mapping: event_id -> {'label': int, 'post_ids': list[str]}
        """
        print("Loading Weibo data...")
        DATA_ROOT   = Path(RP.config.DATA_EXT_DIR) / 'weibo_dataset'
        LABEL_FILE  = DATA_ROOT / 'Weibo.txt'
        if not LABEL_FILE.exists():
            sys.exit(f'ERROR: {LABEL_FILE} not found ‒ make sure “rumdect” is extracted next to {Path(__file__).name}')



        events = {}
        with open(LABEL_FILE, encoding='utf-8') as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                # try comma‑first, then tab
                for sep in (',', '\t'):
                    parts = line.split(sep, 2)
                    if len(parts) >= 3:
                        break
                if len(parts) < 3:
                    continue

                event_str, label_str, post_ids_str = parts
                event_id  = event_str.split(':', 1)[1] if event_str.startswith('eid:')   else event_str
                label_val = label_str.split(':', 1)[1] if label_str.startswith('label:') else label_str

                try:
                    label_int = int(label_val)
                except ValueError:
                    continue

                post_ids = post_ids_str.strip().split()
                events[event_id] = {'label': label_int, 'post_ids': post_ids}

        if not events:
            sys.exit('ERROR: No events were parsed ‒ check the format of Weibo.txt (did you unzip fully and is it UTF‑8?)')
        else:
            print(f"Found {len(events)} events in metadata file.")

        # Sample a fraction of events for quick experiments
        if 0 < event_fraction < 1:
            n_events = max(1, math.ceil(len(events) * event_fraction))
            n_events = min(n_events, len(events))
            selected_event_ids = random.sample(list(events.keys()), n_events)
        else:
            selected_event_ids = list(events.keys())
        print(f"Selected {len(selected_event_ids)} events for loading "
              f"({event_fraction:.0%} of total).")

        all_posts = []
        for eid in selected_event_ids:
            with open(DATA_ROOT / 'Weibo' / f'{eid}.json', encoding='utf-8') as f:
                event_posts = json.load(f)
            for p in event_posts:
                p['event_id'] = eid
            all_posts.extend(event_posts)
        print(f"Loaded {len(all_posts)} posts… building DataFrame.")
        df = pd.DataFrame(all_posts)

        # Keep only the relevant raw columns
        required_cols = ["id", "uid", "parent", "t", "text", "event_id"]
        df = df[[c for c in required_cols if c in df.columns]]

        # Attach label column from events mapping
        df["label"] = df["event_id"].map(lambda eid: events[eid]["label"])

        # Rename to match PHEME DataBase.py conventions
        df.rename(columns={
            "id": "tweet_id",
            "uid": "user_id",
            "parent": "parent_id",
            "t": "ts",
            "event_id": "thread_id"
        }, inplace=True)

        print(f"Keeping columns: {df.columns.tolist()}")

        # summary stats
        label_dist = Counter(events[eid]['label'] for eid in selected_event_ids)
        print(f"Label distribution (0 = non‑rumor, 1 = rumor): {dict(label_dist)}")
        print("DataFrame head:\n", df.head())
        print("dtypes:\n", df.dtypes)

        return df, {eid: events[eid] for eid in selected_event_ids}




    def _precompute_embeddings(self, batch_size=256):
        model  = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ).to(self.device)
        texts  = self.tweets_df["text"].fillna("").tolist()
        batches = range(0, len(texts), batch_size)
        vecs = []
        for i in tqdm(batches, desc="Embedding Weibo posts"):
            enc = model.encode(
                texts[i:i+batch_size],
                convert_to_numpy=True,
                show_progress_bar=False,
                device=self.device
            )
            vecs.append(enc)
        all_vecs = np.vstack(vecs)

        del model; torch.mps.empty_cache(); gc.collect()
        # compress 384-d ⇒ dim_x_tau with random projection
        grp = GaussianRandomProjection(
            n_components=self.dim_x_tau, 
            random_state=RP.config.SEED_RAND
        )
        red = grp.fit_transform(all_vecs)
        self.tweets_df["embed"] = list(red)   # list[ndarray]
        print(f"Embeddings assigned: DataFrame now has {len(self.tweets_df)} rows "
              f"and embed dim {self.dim_x_tau}")

        print("head of tweets_df: ", self.tweets_df.head())
        print("tail of tweets_df: ", self.tweets_df.tail())

    def _build_user_vectors_global(self):
        user_ids   = self.tweets_df["user_id"].unique()
        thread_ids = self.tweets_df["thread_id"].unique()
        u_idx = {u:i for i,u in enumerate(user_ids)}
        t_idx = {t:i for i,t in enumerate(thread_ids)}

        print(f" {len(u_idx)} users and {len(t_idx)} threads")

        # incidence: rows=user, cols=thread
        pairs = {(u_idx[u], t_idx[t]) for u,t in zip(self.tweets_df["user_id"], self.tweets_df["thread_id"])}
        rows, cols = zip(*pairs)
        data = np.ones(len(rows))
        M = csr_matrix((data, (rows, cols)), shape=(len(u_idx), len(t_idx)))

        print(f"CSR matrix shape: {M.shape}")
        
        print(f"SVD with k={self.dim_x_u}")

        U, _, _ = svds(M, k=self.dim_x_u)

        print("SVD completed")
        for uid, i in u_idx.items():
            self.user_vecs_global[uid] = U[i]
        print(f"user_vecs_global built: {len(self.user_vecs_global)} users × {self.dim_x_u} dims")

    def _build_user_vectors_source(self):
        u_idx = {u:i for i,u in enumerate(self.tweets_df["user_id"].unique())}
        pairs_counter = Counter()

        # build weighted graph: common-thread co-occurrence
        for tid, grp in self.tweets_df.groupby("thread_id"):
            users = sorted({u_idx[u] for u in grp["user_id"]})
            for i in range(len(users)):
                for j in range(i+1, len(users)):
                    pairs_counter[(users[i], users[j])] += 1

        rows, cols, w = [],[],[]
        for (i,j), wt in pairs_counter.items():
            rows.extend((i,j))
            cols.extend((j,i))
            w.extend((wt,wt))

        A = csr_matrix((w, (rows, cols)), shape=(len(u_idx), len(u_idx)))
        U, _, _ = svds(A, k=self.dim_y_i)
        for uid, i in u_idx.items():
            self.user_vecs_source[uid] = U[i]
        print(f"user_vecs_source built: {len(self.user_vecs_source)} users × {self.dim_y_i} dims")

    # ------------------------------------------------------------------
    # 4. Thread sequences  (Capture module prep)
    # ------------------------------------------------------------------
    def _build_threads_sequences(self):
        seq_lengths = []
        for tid, grp in self.tweets_df.groupby("thread_id"):
            src_time = self.threads_source[tid]["time"]
            grp_sorted = grp.sort_values("ts")
            bins = ((grp_sorted["ts"] - src_time) // (self.bin_size*3600)).astype(int)

            seq = []
            last_non_empty = None
            for bin_idx, bin_df in grp_sorted.groupby(bins):
                eta = len(bin_df)
                delta_t = 0 if last_non_empty is None else bin_idx - last_non_empty
                last_non_empty = bin_idx

                # aggregate embeddings and user vectors
                x_tau = np.mean(np.vstack(bin_df["embed"]), axis=0)
                user_vecs = [self.user_vecs_global[u] for u in bin_df["user_id"]]
                x_u = np.mean(user_vecs, axis=0) if user_vecs else np.zeros(self.dim_x_u)

                seq.append({
                    "eta": int(eta),
                    "delta_t": int(delta_t),
                    "x_u": x_u,
                    "x_tau": x_tau
                })

            self.threads_seq[tid] = seq
            seq_lengths.append(len(seq))

        self.T = max(seq_lengths)
        print(f"threads_seq built: {len(self.threads_seq)} threads; max sequence length {self.T}")


    def article_ids(self):
        return list(self.threads_seq.keys())

    def article_sequence(self, art_id):
        return self.threads_seq[art_id], self.labels[art_id]

    def user_feature_matrix(self):
        ids = list(self.user_vecs_source.keys())
        mat = np.vstack([self.user_vecs_source[u] for u in ids])
        return ids, mat
    # ------------------------------------------------------------------
    #  Utility: write all heavy pre‑computations as a single .npy
    # ------------------------------------------------------------------
    def save_precomputed_data(self):
        cache = {
            "threads_seq": self.threads_seq,
            "labels": self.labels,
            "user_vecs_global": self.user_vecs_global,
            "user_vecs_source": self.user_vecs_source,
        }
        print(f"Saving precomputed data to {self.precomputed_data_path}")
        with open(self.precomputed_data_path, "wb") as f:
            pickle.dump(cache, f)


    def extract_embed_matrix(self) -> np.ndarray:
        """
        Return all embeddings as a standalone array (n_posts × dim_x_tau).
        Call this before heavy DataFrame operations to free memory if needed.
        """
        embeds = np.vstack(self.tweets_df["embed"].values)
        return embeds
    
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