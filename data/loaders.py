import pandas as pd
import pickle
from pathlib import Path

def load_datasets(data_paths):
    """Load all datasets from specified paths"""
    datasets = {}
    for name, path in data_paths.items():
        datasets[name] = pd.read_csv(path, encoding_errors='ignore').dropna()
    return datasets

def load_embeddings(embedding_paths):
    """Load precomputed embeddings"""
    embeddings = {}
    for name, path in embedding_paths.items():
        with open(path, 'rb') as f:
            embeddings[name] = pickle.load(f)
    return embeddings
