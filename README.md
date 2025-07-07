# megashtein

In their paper ["Deep Squared Euclidean Approximation to the Levenshtein Distance for DNA Storage"](https://arxiv.org/abs/2207.04684), Guo et al. explore techniques for using a neural network to embed sequences in such a way that the squared Euclidean distance between embeddings approximates the Levenshtein distance between the original sequences. This implementation also takes techniques from ["Levenshtein Distance Embeddings with Poisson Regression for DNA Storage" by Wei et al. (2023)](https://arxiv.org/pdf/2312.07931v1).

This is valuable because there are excellent libraries for doing fast GPU accelerated searches for the K nearest neighbors of vectors, like [faiss](https://github.com/facebookresearch/faiss). Algorithms like [HNSW](https://en.wikipedia.org/wiki/Hierarchical_navigable_small_world) allow us to do these searches in logarithmic time, where a brute force levenshtein distance based fuzzy search would need to run in exponential time.

This repo contains a PyTorch implementation of the core ideas from Guo's paper, adapted for ASCII sequences rather than DNA sequences. The implementation includes:

- A convolutional neural network architecture for sequence embedding
- Training using Poisson regression loss (PNLL) as described in the paper
- Synthetic data generation with controlled edit distance relationships
- Model saving and loading functionality

The trained model learns to embed ASCII strings such that the squared Euclidean distance between embeddings approximates the true Levenshtein distance between the strings.

## Model Architecture

- **Base Architecture**: Convolutional Neural Network with embedding layer
- **Input**: ASCII sequences up to 80 characters (padded with null characters)
- **Output**: 80-dimensional dense embeddings
- **Vocab Size**: 128 (ASCII character set)
- **Embedding Dimension**: 140

The model uses a 5-layer CNN with average pooling followed by fully connected layers to produce fixed-size embeddings from variable-length ASCII sequences.

## Installation

```bash
pip install megashtein
```

## Usage

```python
import torch
from megashtein import load_model, embed_string

# Load the pre-trained model
model = load_model()

# Example usage
text1 = "hello world"
text2 = "hello word"

# Embed strings
emb1 = embed_string(model, text1)
emb2 = embed_string(model, text2)

# Compute approximate edit distance
approx_distance = torch.sum((emb1 - emb2) ** 2).item()
print(f"Approximate edit distance: {approx_distance}")
```

## Training Details

The model is trained using:
- **Loss Function**: Poisson Negative Log-Likelihood (PNLL)
- **Optimizer**: AdamW with learning rate 0.000817
- **Batch Size**: 32
- **Sequence Length**: 80 characters (fixed)
- **Synthetic Data**: Pairs of ASCII strings with known Levenshtein distances

## Use Cases

- **Fuzzy String Search**: Find similar strings in large text collections
- **Text Clustering**: Group similar texts based on edit distance
- **Data Deduplication**: Identify near-duplicate text entries
- **Approximate String Matching**: Fast similarity search with controllable accuracy

## Limitations

- **Fixed Length**: Input sequences must be exactly 80 characters (padded/truncated)
- **ASCII Only**: Limited to ASCII character set (0-127)
- **Approximation**: Provides approximate rather than exact edit distances
