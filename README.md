
# Bayesian Word Weighting for Text Classification

## Overview

This repository presents a novel Bayesian word weighting technique tailored for software vulnerability classification. Traditional methods like TF-IDF, TF-IGM, and N-TF-IDF assign a uniform weight to each word across all categories, often overlooking the nuances of category-specific significance. As a result, these approaches may undervalue rare but highly discriminative words that are critical for accurate vulnerability detection. Our method leverages Bayes' theorem to dynamically adjust word weights based on their relevance within each category. By integrating prior knowledge with observed data, the proposed Bayesian framework assigns unique, context-sensitive weights to words, thereby enhancing the representation of textual features.

## Key Features

- **Bayesian Weighting**: Calculates word weights by considering category-specific distributions, ensuring more accurate representation for classification tasks.
- **Word2Vec Integration**: Combines domain-specific Word2Vec embeddings with Bayesian weights for a robust feature representation.
- **Enhanced Classification Performance**: Handles unique category words effectively, even when they appear in only a few documents.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/Bayesian-weighting-framework.git
   cd Bayesian-weighting-framework
   ```

## Usage

### 1. Prepare Your Dataset
Ensure your dataset is in a suitable format (e.g., CSV) with columns for text and corresponding category labels.

### 2. Configuration
Edit config/experiments.py to set up experiments:
```python
EXPERIMENT_CONFIG = {
    # Weighting techniques to compare
    "weighting_techniques": ['bayesian', 'tf-idf', 'ntf-idf', 'tf-igm', 'none'],
    
    # Model architectures
    "architectures": ['BiLSTM', 'BiGRU'],
    
    # Training parameters
    "batch_sizes": [16, 32, 64, 128],
    "max_len": 256,
    "embedding_dim": 300,
    
    # Path configurations
    "base_path": "experiment_results",
    "data_paths": {
        'train': "data/Train_data.csv",
        'test': "data/Test_data.csv"
    }
}
```
### 4. Running Experiments
```python
python -m experiments.run_experiments
```

## Algorithm

For a detailed explanation of the Bayesian word weighting algorithm, see the [Algorithm Documentation](docs/algorithm.md).

## File Structure

```
Bayesian-weighting-framework/
├── config/
│   ├── __init__.py
│   └── experiments.py
├── data/
│   ├── __init__.py
│   └── loaders.py
├── experiments/
│   └── run_experiments.py
├── models/
│   ├── __init__.py
│   └── architectures.py
└── utils/
    ├── __init__.py
    └── gpu_manager.py
```

## Contributing

We welcome contributions! If you have ideas for improvements or find a bug, feel free to open an issue or create a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or support, please reach out to `ai.benyahya@edu.umi.ac.ma`.


