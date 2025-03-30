# megashtein

In their paper [Guo et al.](https://arxiv.org/abs/2207.04684) explore techniques for using a neural network to hash
DNA sequences in such a way that you can do "fuzzy" searches over them using
vector similarity. The distance between two DNA sequence vectors is an approximation
of their [levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance).

This is interesting because there are excellent libraries for doing fast GPU accelerated K nearest neighbor
searches, like [faiss](https://github.com/facebookresearch/faiss). Algorithms
like [HNSW](https://en.wikipedia.org/wiki/Hierarchical_navigable_small_world) allow us to do these searches in logarithmic time,
where a brute force levenshtein distance based fuzzy search would need to run in quadratic time.

In this repo I'm working on reimplementing Guo's paper in PyTorch, applying it to ASCII sequences, to see if this 
research is applicable for general fuzzy searching needs.

## Some details

### Network setup

The authors explore a couple of network models, here I'm initially focusing on their CNN-ED-5 architecture.

### Data

The authors pass in DNA sequences as tensors of one-hot encoded vectors to generate embeddings. This works for 
DNA since DNA sequences have a very small alphabet, not so much for ASCII.

My current approach is to encode ASCII sequences as a tensor of vectors of character embeddings.

My current character embedding algorithm is pretty simple, embeddings are simply vectors over the character's ASCII binary representation.

I.e. 'A' = \[0, 1, 0, 0, 0, 0, 0, 1\]

In the paper the author's work with an actual dataset of DNA sequences. This dataset contains sequences and their accompanying "reads", which contain random errors.
In this repo I generate synthetic data, mainly focusing on generating what the authors refer to as "homologous" and "non-homologous" pairs.

**Homologous pairs** are statistically related. In this repo homologous pairs consist of a source string, and a "mangled" string. The mangled string is the source string after being modified by a random sequence of edits.

**Non-homologous pairs** are not statistically related. In this repo they consist of two random strings.

### Training

The model produces embeddings of sequences, where the squared euclidean distance between the embedding vectors approximates the levenshtein distance. The authors derive a loss
function, $\textrm{RE}_{\chi ^ 2}$ which operates on the measured distance between the vectors and the actual levenshtein distance between the sequences.
