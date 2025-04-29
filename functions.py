import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from sklearn.feature_extraction.text import TfidfVectorizer


# Initialize BERT tokenizer and model
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Initialize TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()


def vectorize_text(text, method="bert"):
    """
    Vectorizes a text using either BERT embeddings or TF-IDF.
    
    Args:
        text (str): The input text to vectorize.
        method (str): The vectorization method ("bert" or "tfidf").
    
    Returns:
        torch.Tensor or np.ndarray: The vectorized representation of the text.
    """
    if method == "bert":
        # Tokenize and encode the text for BERT
        inputs = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = bert_model(**inputs)
        # Use the mean pooling of the last hidden state as the embedding
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Shape: [1, hidden_size]
        return embeddings.squeeze(0)  # Shape: [hidden_size]
    
    elif method == "tfidf":
        # Fit the TF-IDF vectorizer on the dataset (if not already fitted)
        if not hasattr(tfidf_vectorizer, 'vocabulary_'):
            all_texts = [msg for _, messages in data for msg, _ in messages]
            tfidf_vectorizer.fit(all_texts)
        # Transform the input text into a TF-IDF vector
        tfidf_vector = tfidf_vectorizer.transform([text])  # Shape: [1, vocab_size]
        return tfidf_vector.toarray().squeeze(0)  # Shape: [vocab_size]
    
    else:
        raise ValueError("Invalid method. Choose 'bert' or 'tfidf'.")


def average_message_vector(user_messages):
    if not user_messages:
        return torch.zeros(768)
    
    vectors = []
    for (text, _) in user_messages:
        vectors.append(vectorize_text(text))

    return torch.stack(vectors).mean(dim=0)


def extract_time_features(user_messages):
    timestamps = []
    for (_, t) in user_messages:
        timestamps.append(t)

    ts_tensor = torch.tensor(timestamps, dtype=torch.float)
    mean_ts = ts_tensor.mean()
    std_ts = ts_tensor.std()
    num_posts = len(timestamps)
    range_ts = ts_tensor.max() - ts_tensor.min()

    return torch.tensor([mean_ts, std_ts, num_posts, range_ts])


def user_embedding_layer(user2idx, embed_dim_user):
    num_users = len(user2idx)
    return nn.Embedding(num_users, embed_dim_user)


def build_user_feature(user_id, user_messages):
    
    user_embed = user_embedding_layer(torch.tensor([user_id]))  # [1, user_embed_dim]
    msg_vec = average_message_vector(user_messages).unsqueeze(0)  # [1, msg_dim]
    time_feat = extract_time_features(user_messages).unsqueeze(0) # [1, time_dim]
    
    return torch.cat([user_embed, msg_vec, time_feat], dim=-1)  # [1, total_dim]

