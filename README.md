# megashtein

In their paper ["Deep Squared Euclidean Approximation to the Levenshtein Distance for DNA Storage"](https://arxiv.org/abs/2207.04684), Guo et al. explore techniques for using a neural network to embed sequences in such a way that the squared Euclidean distance between embeddings approximates the Levenshtein distance between the original sequences.

This is valuable because there are excellent libraries for doing fast GPU accelerated searches for the K nearest neighbors of vectors, like [faiss](https://github.com/facebookresearch/faiss). Algorithms like [HNSW](https://en.wikipedia.org/wiki/Hierarchical_navigable_small_world) allow us to do these searches in logarithmic time, where a brute force levenshtein distance based fuzzy search would need to run in exponential time.

This repo contains a PyTorch implementation of the core ideas from Guo's paper, adapted for ASCII sequences rather than DNA sequences. The implementation includes:

- A convolutional neural network architecture for sequence embedding
- Training using Poisson regression loss (PNLL) as described in the paper
- Synthetic data generation with controlled edit distance relationships
- Model saving and loading functionality

The trained model learns to embed ASCII strings such that the squared Euclidean distance between embeddings approximates the true Levenshtein distance between the strings.
