#!/usr/bin/env python3
"""
Training script for the megashtein model.

This script trains the EditDistanceModel using the training utilities.
"""

import torch
from pathlib import Path

from ..models import EditDistanceModel
from .train import run_experiment


def train():
    """Main training function."""
    print("Starting megashtein model training...")

    # Model parameters
    embedding_dim = 140

    # Training parameters
    learning_rate = 0.000817
    num_steps = 1000
    size = 80
    batch_size = 32
    use_gradient_clipping = True
    max_grad_norm = 2.463
    distance_metric = "euclidean"

    print("Model parameters:")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Input size: {size}")
    print("Training parameters:")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Number of steps: {num_steps}")
    print(f"  Batch size: {batch_size}")
    print(f"  Distance metric: {distance_metric}")
    print()

    # Create model
    model = EditDistanceModel(embedding_dim=embedding_dim)

    # Run training
    final_loss, final_approx_error = run_experiment(
        embedding_dim=embedding_dim,
        model=model,
        learning_rate=learning_rate,
        num_steps=num_steps,
        size=size,
        batch_size=batch_size,
        use_gradient_clipping=use_gradient_clipping,
        max_grad_norm=max_grad_norm,
        distance_metric=distance_metric,
    )

    print("\nTraining completed!")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Final approximation error: {final_approx_error:.4f}")

    # Save the trained model to the package root
    package_root = Path(__file__).parent.parent.parent.parent
    model_path = package_root / "megashtein_trained_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    train()
