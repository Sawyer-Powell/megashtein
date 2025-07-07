"""
Megashtein: Deep Squared Euclidean Approximation to Levenshtein Distance

A PyTorch implementation for embedding ASCII sequences such that squared Euclidean
distance between embeddings approximates Levenshtein distance between original sequences.
"""

import torch
from pathlib import Path

from .models import EditDistanceModel

__version__ = "0.1.0"
__all__ = ["embed_string", "load_model", "EditDistanceModel"]


def load_model():
    """
    Load the trained model.

    Args:
        model_path: Path to the model file. If None, uses default path.

    Returns:
        Loaded model instance
    """

    model_path = Path(__file__).parent / "megashtein_trained_model.pth"

    model = EditDistanceModel(embedding_dim=140)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    return model


def embed_string(model: EditDistanceModel, text: str) -> torch.Tensor:
    """
    Convert a string to its embedding vector.

    Args:
        text: Input string (ASCII characters only)
        max_length: Maximum sequence length (default: 80)

    Returns:
        torch.Tensor: 80-dimensional embedding vector

    Raises:
        RuntimeError: If model is not loaded
        ValueError: If string contains non-ASCII characters
    """

    # Validate input
    try:
        text.encode('ascii')
    except UnicodeEncodeError:
        raise ValueError("Input string contains non-ASCII characters")

    max_length = 80

    # Pad with null characters and truncate to max_length
    padded = (text + '\0' * max_length)[:max_length]

    # Convert to tensor
    indices = [min(ord(c), 127) for c in padded]
    tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    # Get embedding
    with torch.no_grad():
        embedding = model(tensor)

    return embedding.squeeze(0)
