import os
import json
import pandas as pd
import numpy as np
import re
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

class TwitterRumorDataset(Dataset):
    def __init__(self, article_features, user_features, user_article_masks, labels):
        self.article_features = article_features
        self.user_features = user_features
        self.user_article_masks = user_article_masks
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'article_features': self.article_features[idx],
            'user_features': self.user_features,
            'user_article_mask': self.user_article_masks[idx],
            'labels': self.labels[idx]
        }

def preprocess_text(text):
    """Clean and preprocess tweet text"""
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove user mentions
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags symbol (but keep the text)
    text = re.sub(r'#(\w+)', r'\1', text)
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Convert to lowercase and strip
    text = text.lower().strip()
    return text

def parse_tweet_time(time_str):
    """Parse tweet timestamp into datetime object"""
    try:
        return datetime.strptime(time_str, "%a %b %d %H:%M:%S %z %Y")
    except:
        try:
            return datetime.strptime(time_str, "%a %b %d %H:%M:%S +0000 %Y")
        except:
            return None

def extract_user_features(user_data):
    """Extract relevant user features from user object"""
    features = {
        'verified': 1 if user_data.get('verified', False) else 0,
        'followers_count': user_data.get('followers_count', 0),
        'friends_count': user_data.get('friends_count', 0),
        'statuses_count': user_data.get('statuses_count', 0),
        'favourites_count': user_data.get('favourites_count', 0),
        'listed_count': user_data.get('listed_count', 0),
        'account_age_days': 0,  # Will be calculated if 'created_at' is available
    }
    
    if 'created_at' in user_data:
        created_time = parse_tweet_time(user_data['created_at'])
        if created_time:
            account_age = datetime.now().replace(tzinfo=None) - created_time.replace(tzinfo=None)
            features['account_age_days'] = account_age.days
    
    return features

def load_rumor_data(base_dir, max_tweets_per_thread=100, max_vocab_size=5000):
    """
    Load and process rumor data from the specified directory structure
    
    Args:
        base_dir: Path to the directory containing all-rnr-annotated-threads
        max_tweets_per_thread: Maximum number of tweets to consider per thread
        max_vocab_size: Maximum vocabulary size for feature extraction
    
    Returns:
        processed_data: Dictionary containing processed data
    """
    data_path = os.path.join(base_dir, 'all-rnr-annotated-threads')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    
    events = os.listdir(data_path)
    
    all_threads = []
    all_source_tweets = []
    all_reactions = []
    all_users = {}
    labels = []
    
    # Iterate through all events
    for event in events:
        event_path = os.path.join(data_path, event)
        if not os.path.isdir(event_path):
            continue
            
        # Process rumor threads
        rumor_path = os.path.join(event_path, 'rumours')
        if os.path.exists(rumor_path):
            for thread_id in os.listdir(rumor_path):
                thread_path = os.path.join(rumor_path, thread_id)
                if os.path.isdir(thread_path):
                    thread_data = process_thread(thread_path, thread_id, event, True)
                    if thread_data:
                        all_threads.append(thread_data)
                        all_source_tweets.append(thread_data['source_tweet'])
                        all_reactions.extend(thread_data['reactions'])
                        labels.append(1)  # 1 for rumor
                        
                        # Store user information
                        for user_id, user_data in thread_data['users'].items():
                            if user_id not in all_users:
                                all_users[user_id] = user_data
        
        # Process non-rumor threads
        non_rumor_path = os.path.join(event_path, 'non-rumours')
        if os.path.exists(non_rumor_path):
            for thread_id in os.listdir(non_rumor_path):
                thread_path = os.path.join(non_rumor_path, thread_id)
                if os.path.isdir(thread_path):
                    thread_data = process_thread(thread_path, thread_id, event, False)
                    if thread_data:
                        all_threads.append(thread_data)
                        all_source_tweets.append(thread_data['source_tweet'])
                        all_reactions.extend(thread_data['reactions'])
                        labels.append(0)  # 0 for non-rumor
                        
                        # Store user information
                        for user_id, user_data in thread_data['users'].items():
                            if user_id not in all_users:
                                all_users[user_id] = user_data
    
    print(f"Loaded {len(all_threads)} threads ({sum(labels)} rumors, {len(labels) - sum(labels)} non-rumors)")
    print(f"Total source tweets: {len(all_source_tweets)}, reactions: {len(all_reactions)}")
    print(f"Total unique users: {len(all_users)}")
    
    # Process text data using TF-IDF
    vectorizer = TfidfVectorizer(max_features=max_vocab_size, stop_words='english')
    
    # Prepare text corpus from tweets
    all_tweet_texts = [tweet['text'] for tweet in all_source_tweets]
    all_tweet_texts.extend([tweet['text'] for tweet in all_reactions])
    
    # Fit and transform text data
    tfidf_matrix = vectorizer.fit_transform(all_tweet_texts)
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    
    # Create time series for each thread
    article_features = []
    user_article_masks = []
    
    for thread_idx, thread in enumerate(all_threads):
        # Get reactions sorted by time
        thread_reactions = sorted(thread['reactions'], key=lambda x: x['timestamp'])[:max_tweets_per_thread]
        
        if len(thread_reactions) == 0:
            # If no reactions, use only source tweet features
            source_tweet = thread['source_tweet']
            source_text_vec = vectorizer.transform([source_tweet['text']]).toarray()[0]
            
            # Create basic time series with only the source tweet
            ts_features = np.zeros((1, max_vocab_size + 2))  # +2 for engagement count and time delta
            ts_features[0, :max_vocab_size] = source_text_vec
            ts_features[0, max_vocab_size] = 1  # Engagement count (just the source)
            ts_features[0, max_vocab_size + 1] = 0  # Time delta (initial tweet)
            
            # Pad to ensure consistent sequence length
            padded_features = np.zeros((max_tweets_per_thread, max_vocab_size + 2))
            padded_features[0] = ts_features[0]
            
            article_features.append(padded_features)
        else:
            # Create time series for the thread
            ts_features = np.zeros((len(thread_reactions) + 1, max_vocab_size + 2))
            
            # Add source tweet as first entry
            source_tweet = thread['source_tweet']
            source_text_vec = vectorizer.transform([source_tweet['text']]).toarray()[0]
            ts_features[0, :max_vocab_size] = source_text_vec
            ts_features[0, max_vocab_size] = 1  # Initial engagement count
            ts_features[0, max_vocab_size + 1] = 0  # Time delta for source tweet
            
            # Source tweet time
            source_time = source_tweet['timestamp']
            
            # Add reactions
            for i, reaction in enumerate(thread_reactions):
                # Get text vector for this reaction
                reaction_text_vec = vectorizer.transform([reaction['text']]).toarray()[0]
                
                # Calculate time delta in hours
                time_delta = (reaction['timestamp'] - source_time).total_seconds() / 3600
                
                # Set features
                ts_features[i + 1, :max_vocab_size] = reaction_text_vec
                ts_features[i + 1, max_vocab_size] = i + 2  # Cumulative engagement count
                ts_features[i + 1, max_vocab_size + 1] = time_delta
            
            # Pad if necessary
            if len(thread_reactions) + 1 < max_tweets_per_thread:
                padded_features = np.zeros((max_tweets_per_thread, max_vocab_size + 2))
                padded_features[:len(thread_reactions) + 1] = ts_features
                article_features.append(padded_features)
            else:
                article_features.append(ts_features[:max_tweets_per_thread])
        
        # Create user-article mask
        thread_user_ids = [thread['source_tweet']['user_id']]
        thread_user_ids.extend([reaction['user_id'] for reaction in thread_reactions])
        
        # Create mask vector (1 for users involved in this thread)
        user_mask = np.zeros(len(all_users))
        for user_id in thread_user_ids:
            if user_id in all_users:
                user_idx = list(all_users.keys()).index(user_id)
                user_mask[user_idx] = 1
                
        user_article_masks.append(user_mask)
    
    # Process user features
    user_features_list = []
    for user_id, user_data in all_users.items():
        user_features = extract_user_features(user_data)
        user_features_list.append(list(user_features.values()))
    
    # Convert to numpy arrays
    user_features_array = np.array(user_features_list)
    article_features_array = np.array(article_features)
    user_article_masks_array = np.array(user_article_masks)
    labels_array = np.array(labels)
    
    # Normalize user features
    scaler = StandardScaler()
    user_features_normalized = scaler.fit_transform(user_features_array)
    
    # Convert to PyTorch tensors
    article_features_tensor = torch.FloatTensor(article_features_array)
    user_features_tensor = torch.FloatTensor(user_features_normalized)
    user_article_masks_tensor = torch.FloatTensor(user_article_masks_array)
    labels_tensor = torch.FloatTensor(labels_array)
    
    return {
        'article_features': article_features_tensor,
        'user_features': user_features_tensor,
        'user_article_masks': user_article_masks_tensor,
        'labels': labels_tensor,
        'vectorizer': vectorizer,
        'scaler': scaler,
        'threads': all_threads,
        'users': all_users
    }

def process_thread(thread_path, thread_id, event_name, is_rumor):
    """Process a single thread (source tweet and its reactions)"""
    # Load source tweet
    source_tweet_path = os.path.join(thread_path, 'source-tweets')
    source_files = [f for f in os.listdir(source_tweet_path) if f.endswith('.json')]
    
    if not source_files:
        return None
    
    source_file = os.path.join(source_tweet_path, source_files[0])
    
    try:
        with open(source_file, 'r', encoding='utf-8') as f:
            source_data = json.load(f)
    except:
        return None
    
    # Load annotation if available
    annotation_file = os.path.join(thread_path, 'annotation.json')
    annotation = None
    if os.path.exists(annotation_file):
        try:
            with open(annotation_file, 'r', encoding='utf-8') as f:
                annotation = json.load(f)
        except:
            pass
    
    # Load conversation structure
    structure_file = os.path.join(thread_path, 'structure.json')
    structure = None
    if os.path.exists(structure_file):
        try:
            with open(structure_file, 'r', encoding='utf-8') as f:
                structure = json.load(f)
        except:
            pass
    
    # Process source tweet
    source_tweet = {
        'id': thread_id,
        'event': event_name,
        'text': preprocess_text(source_data.get('text', '')),
        'user_id': str(source_data.get('user', {}).get('id', '')),
        'timestamp': parse_tweet_time(source_data.get('created_at', '')),
        'is_rumor': is_rumor
    }
    
    # Store user information
    users = {}
    if 'user' in source_data:
        users[source_tweet['user_id']] = source_data['user']
    
    # Process reactions
    reactions = []
    reactions_path = os.path.join(thread_path, 'reactions')
    if os.path.exists(reactions_path):
        for reaction_file in os.listdir(reactions_path):
            if reaction_file.endswith('.json'):
                try:
                    with open(os.path.join(reactions_path, reaction_file), 'r', encoding='utf-8') as f:
                        reaction_data = json.load(f)
                        
                    reaction = {
                        'id': reaction_data.get('id_str', ''),
                        'text': preprocess_text(reaction_data.get('text', '')),
                        'user_id': str(reaction_data.get('user', {}).get('id', '')),
                        'timestamp': parse_tweet_time(reaction_data.get('created_at', ''))
                    }
                    
                    # Add user data
                    if 'user' in reaction_data:
                        users[reaction['user_id']] = reaction_data['user']
                    
                    reactions.append(reaction)
                except:
                    continue
    
    # Create thread data
    thread_data = {
        'id': thread_id,
        'event': event_name,
        'is_rumor': is_rumor,
        'source_tweet': source_tweet,
        'reactions': reactions,
        'users': users,
        'annotation': annotation,
        'structure': structure
    }
    
    return thread_data

def create_data_loaders(processed_data, batch_size=16, train_ratio=0.8):
    """Create DataLoader objects for training and validation"""
    dataset = TwitterRumorDataset(
        processed_data['article_features'],
        processed_data['user_features'],
        processed_data['user_article_masks'],
        processed_data['labels']
    )
    
    # Split dataset into training and validation sets
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Example usage
    base_dir = "d:/cours/MA2/Projet_ML/datas"
    
    print("Loading and processing data...")
    processed_data = load_rumor_data(base_dir)
    
    print("\nCreating data loaders...")
    train_loader, val_loader = create_data_loaders(processed_data)
    
    print(f"\nDataset created successfully!")
    print(f"Article features shape: {processed_data['article_features'].shape}")
    print(f"User features shape: {processed_data['user_features'].shape}")
    print(f"User-article masks shape: {processed_data['user_article_masks'].shape}")
    print(f"Labels shape: {processed_data['labels'].shape}")
    
    print(f"\nTraining batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Save processed data (optional)
    torch.save(processed_data, 'd:/cours/MA2/Projet_ML/rumorProject/processed_rumor_data.pt')
    print("\nProcessed data saved to 'processed_rumor_data.pt'")