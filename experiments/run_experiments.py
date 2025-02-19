import os
import json
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from tensorflow import keras
from collections import defaultdict
from gensim.models import Word2Vec
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score
from sklearn.metrics import roc_auc_score, matthews_corrcoef
import random
import tensorflow as tf

from config.experiments import EXPERIMENT_CONFIG
from data.loaders import load_datasets, load_embeddings
from models.architectures import create_model
from utils.gpu_manager import clear_gpu_memory
import time

import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set seeds for reproducibility
random_seed = 42
tf.random.set_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

## Optional: Set deterministic operations if using GPU\n",
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExperimentRunner:
    def __init__(self, config):
        self.config = config
        self.base_path = Path(config['base_path'])
        self.results = defaultdict(list)
    
    def split_valid_data(self, df):
        labeles = df.Label
        text = df.Description
        return (text, labeles)

    def load_tfidf_weights(self):
        # Load tf-idf weights from json file
        with open(self.config['tfidf_weights_path'], 'r') as f:
            return json.load(f)
    
    def load_bayesian_weights(self):
        # Load Bayesian weights from json file
        with open(self.config['bayesian_weights_path'], 'r') as f:
            return json.load(f)
        
    def load_ntfidf_weights(self):
        # Load ntf-idf weights from json file
        with open(self.config['ntfidf_weights_path'], 'r') as f:
            return json.load(f)
    
    def load_tfigm_weights(self):
        # Load tf-igm weights from json file
        with open(self.config['tfigm_weights_path'], 'r') as f:
            return json.load(f)

    def _setup_directories(self, technique, architecture, batch_size):
        """Create directory structure for results"""
        exp_dir = self.base_path / technique / architecture / str(batch_size)
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir
    
    def create_weighted_embeddings(self, texts, labels, words_weights, model, max_len, embedding_dim, technique='none'):
        embeddings = []
        for text, label in tqdm(zip(texts, labels)):
            words = text.split()  # Split the text into words
            embedding = []
            for word in words:
                if word in model.wv:  # Check if the word exists in the model's vocabulary
                    if technique == 'tf-idf' or technique == 'tf-igm':
                        if word in words_weights:
                            embedding.append(model.wv[word]*words_weights[word])
                        else:
                            embedding.append(model.wv[word])
                    elif technique == 'bayesian':
                        if word in words_weights[label]:
                            embedding.append(model.wv[word]*words_weights[label][word])
                        else:
                            embedding.append(model.wv[word])
                    elif technique == 'ntf-idf':
                        if word in words_weights[label]:
                            embedding.append(model.wv[word]*max(words_weights['esv'][word], words_weights['gpsv'][word], words_weights['ambiguous'][word]))
                        else:
                            embedding.append(model.wv[word]*0.5)
                    elif technique == 'none':
                        embedding.append(model.wv[word])
                else:
                    embedding.append(np.zeros(embedding_dim))  # Create a zero vector for OOV words
            # Pad or truncate the sequence to max_len
            embedding = np.array(embedding)
            if len(embedding) < max_len:
                padding = np.zeros((max_len - len(embedding), embedding_dim))
                embedding = np.concatenate([embedding, padding])
            else:
                embedding = embedding[:max_len]  # Truncate if too long
            embeddings.append(embedding)
        return np.array(embeddings)
    
    def _get_weight_matrix(self, technique):
        """Load or compute weighting matrix for technique"""
        # Implement your weighting matrix loading logic here
        # Example placeholder:
        if technique == 'tf-idf':
            return self.load_tfidf_weights()
        elif technique == 'bayesian':
            return self.load_bayesian_weights()
        elif technique == 'ntf-idf':
            return self.load_ntfidf_weights()
        elif technique == 'tf-igm':
            return self.load_tfigm_weights()
        return None
    
    def _create_callbacks(self, exp_dir):
        """Create training callbacks"""
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config['patience'],
            restore_best_weights=True
        )
        
        checkpoint = keras.callbacks.ModelCheckpoint(
            str(exp_dir / 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.0001
        )
        
        return [early_stop, checkpoint, reduce_lr]
    
    def _evaluate_model(self, model, test_sets, exp_dir):
        """Evaluate model on all test sets and save results"""
        metrics = {}
        for test_name, (X_test, y_test) in test_sets.items():
            y_pred = model.predict(X_test)
            test_metrics = self._calculate_metrics(y_test, y_pred)
            metrics[test_name] = test_metrics
            
            # Save predictions
            pd.DataFrame(y_pred).to_csv(exp_dir / f'{test_name}_predictions.csv')
        
        # Save metrics
        with open(exp_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f)
            
        return metrics
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate evaluation metrics"""
        y_pred_labels = np.argmax(y_pred, axis=1)
        return {
            'accuracy': accuracy_score(y_true, y_pred_labels),
            'precision': precision_score(y_true, y_pred_labels, average='macro'),
            'recall': recall_score(y_true, y_pred_labels, average='macro'),
            'f1': f1_score(y_true, y_pred_labels, average='macro'),
            'mcc': matthews_corrcoef(y_true, y_pred_labels),
            'auc_roc': roc_auc_score(y_true, y_pred, multi_class="ovr", average="macro")
        }
    
    def run_experiments(self):
        """Main method to run all experiments"""
        # Load datasets and embeddings
        data_paths = {
            'variot': self.config['variot_path'],
            'cnnvd': self.config['cnnvd_path'],
            'cnvd': self.config['cnvd_path'],
            'unseen': self.config['unseen_path'],
            # Add all other dataset paths
        }
        datasets = load_datasets(data_paths)

        # Split test data into descriptions and labels
        X_cnnvd, y_cnnvd = self.split_valid_data(datasets['cnnvd_prep'])
        X_cnvd, y_cnvd = self.split_valid_data(datasets['cnvd_prep'])
        X_variot, y_variot = self.split_valid_data(datasets['variot_prep'])
        X_unseen2, y_unseen2 = self.split_valid_data(datasets['unseen_prep'])

        # Map labels to ids
        label_mapping = {'esv': 0, 'gpsv': 1, 'ambiguous': 2}
        y_unseen2_numeric = y_unseen2.map(label_mapping)
        y_cnnvd_numeric = y_cnnvd.map(label_mapping)
        y_cnvd_numeric = y_cnvd.map(label_mapping)
        y_variot_numeric = y_variot.map(label_mapping)
        
        embedding_paths = {
            'X_train': self.config['X_train_path'],
            'X_valid': self.config['X_valid_path'],
            'y_train': self.config['y_train_path'],
            'y_valid': self.config['y_valid_path'],
        }
        embeddings = load_embeddings(embedding_paths)
        # Assuming y_train is your label column
        label_mapping = {'esv': 0, 'gpsv': 1, 'ambiguous': 2}
        y_train_numeric = embeddings['y_train'].map(label_mapping)
        # Assuming y_train is your label column
        label_mapping = {'esv': 0, 'gpsv': 1, 'ambiguous': 2}
        y_test_numeric = embeddings['y_valid'].map(label_mapping)
        # Load vuln2vec embeddings
        vuln2vec_model = Word2Vec.load(self.config['vuln2vec_path'])
        
        # Main experiment loop
        for technique in tqdm(self.config['weighting_techniques'], desc='Weighting Techniques'):
            weight_matrix = self._get_weight_matrix(technique)
            
            for architecture in tqdm(self.config['architectures'], desc='Architectures'):
                for batch_size in tqdm(self.config['batch_sizes'], desc='Batch Sizes'):
                    try:
                        # Create embeddings with current weighting
                        train_embeddings = self.create_weighted_embeddings( 
                            embeddings['X_train'],
                            embeddings['y_train'],
                            weight_matrix,
                            vuln2vec_model,
                            self.config['max_len'],
                            self.config['embedding_dim'],
                            technique
                        )
                        valid_embeddings = self.create_weighted_embeddings(
                            embeddings['X_valid'],
                            embeddings['y_valid'],
                            weight_matrix,
                            vuln2vec_model,
                            self.config['max_len'],
                            self.config['embedding_dim'],
                            technique
                        )

                        unseen_embeddings = self.create_weighted_embeddings(
                            X_unseen2,
                            y_unseen2,
                            weight_matrix,
                            vuln2vec_model,
                            self.config['max_len'],
                            self.config['embedding_dim'],
                            technique
                        )

                        cnnvd_embeddings = self.create_weighted_embeddings(
                            X_cnnvd,
                            y_cnnvd,
                            weight_matrix,
                            vuln2vec_model,
                            self.config['max_len'],
                            self.config['embedding_dim'],
                            technique
                        )

                        cnvd_embeddings = self.create_weighted_embeddings(
                            X_cnvd,
                            y_cnvd,
                            weight_matrix,
                            vuln2vec_model,
                            self.config['max_len'],
                            self.config['embedding_dim'],
                            technique
                        )

                        variot_embeddings = self.create_weighted_embeddings(
                            X_variot,
                            y_variot,
                            weight_matrix,
                            vuln2vec_model,
                            self.config['max_len'],
                            self.config['embedding_dim'],
                            technique
                        )
                        
                        # Setup directories
                        exp_dir = self._setup_directories(technique, architecture, batch_size)
                        
                        # Create and train model
                        model = create_model(
                            architecture=architecture,
                            input_shape=(self.config['max_len'], self.config['embedding_dim']),
                            n_classes=self.config['n_classes']
                        )

                        start_time = time.time()
                        history = model.fit(
                            train_embeddings,
                            y_train_numeric,
                            batch_size=batch_size,
                            epochs=self.config['epochs'],
                            validation_data=(valid_embeddings, y_test_numeric),
                            callbacks=self._create_callbacks(exp_dir),
                            verbose=1
                        )

                        total_time = (time.time() - start_time) / 60  # minutes
                        # Evaluate and save results
                        test_sets = {
                            'cnnvd': (cnnvd_embeddings, y_cnnvd_numeric),
                            'cnvd': (cnvd_embeddings, y_cnvd_numeric),
                            'unseen': (unseen_embeddings, y_unseen2_numeric),
                            'variot': (variot_embeddings, y_variot_numeric),
                        }
                        metrics = self._evaluate_model(model, test_sets, exp_dir)
                        
                        # Save training history
                        pd.DataFrame(history.history).to_csv(exp_dir / 'training_history.csv')
                        
                        # Store results for final report
                        self._store_results(technique, architecture, batch_size, metrics, total_time)
                        
                    except Exception as e:
                        logger.error(f"Error in {technique}-{architecture}-{batch_size}: {str(e)}")
                    finally:
                        clear_gpu_memory()
                        
        # Save final consolidated results
        self._save_final_report()
    
    def _store_results(self, technique, architecture, batch_size, metrics, model_time):
        """Aggregate results across all experiments"""
        for test_set, values in metrics.items():
            self.results['technique'].append(technique)
            self.results['architecture'].append(architecture)
            self.results['batch_size'].append(batch_size)
            self.results['test_set'].append(test_set)
            self.results['total_time'].append(model_time)
            for metric, value in values.items():
                self.results[metric].append(value)
                
    def _save_final_report(self):
        """Save all results to a consolidated CSV"""
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(self.base_path / 'experiment_results.csv', index=False)
        logger.info("Saved consolidated results to experiment_results.csv")

if __name__ == "__main__":
    runner = ExperimentRunner(EXPERIMENT_CONFIG)
    runner.run_experiments()