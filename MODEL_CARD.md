---
license: mit
language:
- en
library_name: pytorch
tags:
- sequence-embedding
- levenshtein-distance
- edit-distance
- neural-network
- string-similarity
- deep-learning
pipeline_tag: feature-extraction
---

# Megashtein: Deep Squared Euclidean Approximation to Levenshtein Distance

## Model Description

Megashtein is a neural network model that learns to embed ASCII sequences such that the squared Euclidean distance between embeddings approximates the Levenshtein (edit) distance between the original sequences. This approach is based on the paper ["Deep Squared Euclidean Approximation to the Levenshtein Distance for DNA Storage"](https://arxiv.org/abs/2207.04684) by Guo et al., adapted for ASCII text sequences.

The model transforms string similarity computation from expensive dynamic programming (O(nm)) to fast vector distance calculation (O(d)), enabling efficient fuzzy search and clustering operations on large text collections.

## Model Architecture

- **Base Architecture**: Convolutional Neural Network with embedding layer
- **Input**: ASCII sequences up to 80 characters (padded with null characters)
- **Output**: 80-dimensional dense embeddings
- **Vocab Size**: 128 (ASCII character set)
- **Embedding Dimension**: 140
- **Total Parameters**: ~500K

### Architecture Details

```
EditDistanceModel(
  (embedding): Embedding(128, 140)
  (conv_layers): Sequential(
    (0): Conv1d(140, 64, kernel_size=3, stride=1, padding=1)
    (1): AvgPool1d(kernel_size=2)
    (2): ReLU()
    (3): Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
    (4): AvgPool1d(kernel_size=2)
    (5): ReLU()
    (6): Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
    (7): AvgPool1d(kernel_size=2)
    (8): ReLU()
    (9): Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
    (10): AvgPool1d(kernel_size=2)
    (11): ReLU()
    (12): Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
    (13): AvgPool1d(kernel_size=2)
    (14): ReLU()
  )
  (fc_layers): Sequential(
    (0): Linear(in_features=64, out_features=200)
    (1): ReLU()
    (2): Linear(in_features=200, out_features=80)
    (3): BatchNorm1d(80)
  )
)
```

## Training Details

### Training Data
- **Synthetic Data Generation**: Pairs of ASCII strings with known Levenshtein distances
- **Homologous Pairs**: Original strings with controlled edit operations (insertions, deletions, substitutions)
- **Non-homologous Pairs**: Random string pairs
- **Sequence Length**: 80 characters (fixed)
- **Batch Size**: 32

### Training Procedure
- **Loss Function**: Poisson Negative Log-Likelihood (PNLL) as described in Wei et al.
- **Optimizer**: AdamW with weight decay (1e-5)
- **Learning Rate**: 0.000817
- **Gradient Clipping**: Max norm 2.463
- **Training Steps**: 1000
- **Scheduler**: StepLR (step_size=200, gamma=0.5)

### Training Formula
The model uses the Poisson regression loss from the original paper:
```
PNLL(d̂, d) = d̂ - d * ln(d̂)
```
where d̂ is the predicted distance and d is the true Levenshtein distance.

## Intended Use

### Primary Use Cases
- **Fuzzy String Search**: Find similar strings in large text collections
- **Text Clustering**: Group similar texts based on edit distance
- **Approximate String Matching**: Fast similarity search with controllable accuracy
- **Data Deduplication**: Identify near-duplicate text entries

### Direct Use
```python
import torch
from models import EditDistanceModel

# Load the model
model = EditDistanceModel(embedding_dim=140)
model.load_state_dict(torch.load('megashtein_trained_model.pth'))
model.eval()

# Embed strings
def embed_string(text, max_length=80):
    # Pad and convert to tensor
    padded = (text + '\0' * max_length)[:max_length]
    indices = [min(ord(c), 127) for c in padded]
    tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)
    
    with torch.no_grad():
        embedding = model(tensor)
    return embedding

# Example usage
text1 = "hello world"
text2 = "hello word"

emb1 = embed_string(text1)
emb2 = embed_string(text2)

# Compute approximate edit distance
approx_distance = torch.sum((emb1 - emb2) ** 2).item()
print(f"Approximate edit distance: {approx_distance}")
```

### Downstream Use
The embeddings can be used with:
- **FAISS**: For efficient k-nearest neighbor search
- **HNSW**: For hierarchical navigable small world search
- **Standard clustering algorithms**: K-means, DBSCAN, etc.

## Performance

The model learns to approximate Levenshtein distances with the following characteristics:
- **Speed**: O(d) distance computation vs O(nm) for dynamic programming
- **Accuracy**: Embeddings preserve relative distance relationships
- **Scalability**: Enables sub-linear search in large text collections

## Limitations

- **Fixed Length**: Input sequences must be exactly 80 characters (padded/truncated)
- **ASCII Only**: Limited to ASCII character set (0-127)
- **Approximation**: Provides approximate rather than exact edit distances
- **Training Domain**: Optimized for general ASCII text patterns

## Bias and Fairness

The model is trained on synthetic data with random string generation, which may not reflect real-world text distributions. Performance may vary across different:
- Languages and writing systems
- Text domains and genres
- String length distributions
- Character frequency patterns

## Citation

If you use this model, please cite the original paper:

```bibtex
@article{guo2022deep,
  title={Deep Squared Euclidean Approximation to the Levenshtein Distance for DNA Storage},
  author={Guo, Alan JX and Liang, Cong and Hou, Qing-Hu},
  journal={arXiv preprint arXiv:2207.04684},
  year={2022}
}
```

## Model Card Authors

This model card was created by the megashtein project contributors.

## Model Card Contact

For questions about this model, please open an issue in the [megashtein repository](https://github.com/username/megashtein).