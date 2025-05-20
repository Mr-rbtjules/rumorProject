import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "./data/logs")
MODELS_DIR = os.path.join(BASE_DIR, "./data/saved_models")
FIG_DIR = os.path.join(BASE_DIR, "./data/figures")

# Chemin absolu vers les données
RAW_DATA_DIR = r"D:\cours\MA2\Projet_ML\datas\all-rnr-annotated-threads"

# Chemin pour le cache - assurez-vous que ce dossier existe
CACHE_DATA_DIR = os.path.join(BASE_DIR, "./data/cache")
os.makedirs(CACHE_DATA_DIR, exist_ok=True)

SEED_NP = 4
SEED_RAND = 5
SEED_SHUFFLE = 6
SEED_TORCH = 7