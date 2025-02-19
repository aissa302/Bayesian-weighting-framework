EXPERIMENT_CONFIG = {
    "weighting_techniques": [
        'bayesian'#, 'tf-idf', 'ntf-idf', 'tf-igm', 'none'
    ],
    "bayesian_weights_path": "../data/bayes_weights_2_11_24.json",
    "ntfidf_weights_path": "../data/ntfidf_weights_07_02_25.json",
    "tfigm_weights_path": "../data/tfigm_weights_06_02_25.json",
    "tfidf_weights_path": "../data/tfidf_weights_07_02_25.json",
    "X_train_path": "../data/X_train_full.pkl",
    "y_train_path": "../data/y_train_full.pkl",
    "X_valid_path": "../data/X_test_full.pkl",
    "y_valid_path": "../data/y_test_full.pkl",
    "unseen_path": "../data/unseen2.csv",
    "cnnvd_path": "../data/cnnvd_ambiguous_test_data_2024.csv",
    "cnvd_path": "../data/cnvd_ambiguous_test_data_2024.csv",
    "variot_path": "../data/variot_ambiguous_test_data_2024.csv",
    "vuln2vec_path": "../data/vuln2vec-300-vulnDBv2.model",
    "architectures": ['BiLSTM'],#, 'BiGRU'],
    "batch_sizes": [128],
    "base_path": "../code_test",
    "max_len": 256,
    "embedding_dim": 300,
    "n_classes": 3,
    "epochs": 100,
    "patience": 5
}