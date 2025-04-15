import torch
import torch.nn as nn


# Raw input
data = [
    ['alice', [['the earth is flat', 100]]],
    ['bob', [['government lies again', 120], ['fake news', 121]]],
    ['charlie', [['totally agree!', 101]]],
    ['dave', [['I disagree', 122], ['earth is not flat', 123]]]
]

# Build a dictionary: user -> user_index
user2idx = {}
for user, messages in data:
    if user not in user2idx:
        user2idx[user] = len(user2idx)  # Assign a unique index to each user
# Result: {'alice': 0, 'bob': 1, 'charlie': 2, 'dave': 3}

# Timestamp and normalization
timestamps = []
for user_id, messages in data:
    for message, timestamp in messages:
        timestamps.append(timestamp)

min_time, max_time = min(timestamps), max(timestamps)

# Flatten the data
flattened_data = []
for user_id, messages in data:
    user_id = user2idx[user_id]
    for message, timestamp in messages:
        flattened_data.append(user_id, message, (timestamp - min_time)/(max_time - min_time))

