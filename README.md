# megashtein

In their paper [Guo et al.](https://arxiv.org/abs/2207.04684) explore techniques for using a neural network to hash
DNA sequences in such a way that you can do "fuzzy" searches over them using
vector similarity. The distance between two DNA sequence vectors is an approximation
of their [levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance).

This is cool because there are excellent libraries for doing fast GPU accelerated searches
for the K nearest neighbors of vectors, like [faiss](https://github.com/facebookresearch/faiss). Algorithms
like [HNSW](https://en.wikipedia.org/wiki/Hierarchical_navigable_small_world) allow us to do these searches in logarithmic time,
where a brute force levenshtein distance based fuzzy search would need to run in exponential time.

In this repo I'm working on reimplementing Guo's paper in PyTorch, applying it to ASCII sequences,
not DNA sequences.
