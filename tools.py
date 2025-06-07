
import datetime as dt


def _to_ts(tstr):
    """ "Wed Jan 07 11:06:08 +0000 2015" to 1420628768"""
    fmt = "%a %b %d %H:%M:%S %z %Y"
    return int(dt.datetime.strptime(tstr, fmt).timestamp())

def testPackage() -> None:
    """ fct to test the package """
    print("Package is working")
    return None


from collections import Counter
import matplotlib.pyplot as plt

def plot_sequence_length_distribution(lengths):
    # Count frequencies of each length
    length_counts = Counter(lengths)

    # Create a list of counts for each length from 1 to 27
    counts = [length_counts.get(i, 0) for i in range(min(lengths), max(lengths) + 1)]

    # Create bar plot
    plt.figure(figsize=(10, 5))
    plt.bar(range(min(lengths), max(lengths)+1), counts)
    plt.xlabel('Sequence Length')
    plt.ylabel('Count')
    plt.title('Distribution of Sequence Lengths')
    plt.xticks(range(min(lengths), max(lengths)+1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

