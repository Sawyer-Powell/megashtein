#!/usr/bin/env python3
"""
Tests for the megashtein package.
"""

import torch
from pathlib import Path
import sys

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from megashtein import embed_string, load_model, EditDistanceModel

model = load_model()

def test_embed_string():
    """Test that embed_string returns correct shape and type."""
    print("Testing embed_string...")

    # Test basic functionality
    text = "hello world"
    embedding = embed_string(model, text)

    # Check return type
    assert isinstance(embedding, torch.Tensor), f"Expected torch.Tensor, got {type(embedding)}"

    # Check shape
    expected_shape = (80,)
    assert embedding.shape == expected_shape, f"Expected shape {expected_shape}, got {embedding.shape}"

    # Check that different strings produce different embeddings
    text2 = "goodbye world"
    embedding2 = embed_string(model, text2)

    # They should be different
    assert not torch.equal(embedding, embedding2), "Different strings should produce different embeddings"

    print("✓ embed_string test passed")


def test_distance_approximation():
    """Test that similar strings have smaller distances than dissimilar ones."""
    print("Testing distance approximation...")

    # Test strings with known relationships
    original = "hello world"
    similar = "hello word"  # 1 edit (l->empty)
    different = "xyz abc"   # completely different

    # Get embeddings
    emb_original = embed_string(model, original)
    emb_similar = embed_string(model, similar)
    emb_different = embed_string(model, different)

    # Calculate squared euclidean distances
    dist_similar = torch.sum((emb_original - emb_similar) ** 2).item()
    dist_different = torch.sum((emb_original - emb_different) ** 2).item()

    # Similar strings should have smaller distance than different strings
    assert dist_similar < dist_different, f"Similar distance ({dist_similar}) should be less than different distance ({dist_different})"

    print("✓ Distance approximation test passed")
    print(f"  Original vs Similar: {dist_similar:.3f}")
    print(f"  Original vs Different: {dist_different:.3f}")


def test_model_loading():
    """Test that the model can be loaded properly."""
    print("Testing model loading...")

    # Test that we can create a model instance
    model = EditDistanceModel(embedding_dim=140)
    assert isinstance(model, torch.nn.Module), "Model should be a PyTorch module"

    # Test that the model has the expected structure
    assert hasattr(model, 'embedding'), "Model should have embedding layer"
    assert hasattr(model, 'conv_layers'), "Model should have conv layers"
    assert hasattr(model, 'fc_layers'), "Model should have fully connected layers"

    print("✓ Model loading test passed")


def test_string_validation():
    """Test that non-ASCII strings are properly rejected."""
    print("Testing string validation...")

    # Test that ASCII strings work
    ascii_text = "hello world 123!@#"
    embedding = embed_string(model, ascii_text)
    assert embedding.shape == (80,), "ASCII string should work"

    # Test that non-ASCII strings raise ValueError
    try:
        embed_string(model, "héllo wørld")
        assert False, "Non-ASCII string should raise ValueError"
    except ValueError as e:
        assert "non-ASCII characters" in str(e), "Should raise ValueError about non-ASCII characters"

    print("✓ String validation test passed")


def test_padding_and_truncation():
    """Test that strings are properly padded and truncated."""
    print("Testing padding and truncation...")

    # Test short string (should be padded)
    short_text = "hi"
    embedding_short = embed_string(model, short_text)
    assert embedding_short.shape == (80,), "Short string should be padded to correct length"

    # Test long string (should be truncated)
    long_text = "a" * 100
    embedding_long = embed_string(model, long_text)
    assert embedding_long.shape == (80,), "Long string should be truncated to correct length"

    # Test exact length string
    exact_text = "a" * 80
    embedding_exact = embed_string(model, exact_text)
    assert embedding_exact.shape == (80,), "Exact length string should work"

    print("✓ Padding and truncation test passed")


def main():
    """Run all tests."""
    print("Running megashtein package tests...")
    print("=" * 50)

    try:
        test_embed_string()
        test_distance_approximation()
        test_model_loading()
        test_string_validation()
        test_padding_and_truncation()

        print("=" * 50)
        print("✓ All tests passed!")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
