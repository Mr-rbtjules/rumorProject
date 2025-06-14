import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "./data/logs")
MODELS_DIR = os.path.join(BASE_DIR, "./data/saved_models")
DATA_EXT_DIR = os.path.join(BASE_DIR, "../data_ext")
RAW_DATA_DIR = os.path.join(BASE_DIR, "./data/pheme-rnr-dataset")
CACHE_DATA_DIR = os.path.join(BASE_DIR, "./data/cache")
FIG_DIR = os.path.join(BASE_DIR, "./data/figures")


SEED_NP = 4
SEED_RAND = 5
SEED_SHUFFLE = 6
SEED_TORCH = 42