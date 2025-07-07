"""
Training utilities for the megashtein package.

This module contains functions for training the EditDistanceModel.
"""

from .train import (
    pad_with_null,
    string_to_tensor,
    random_char,
    random_str,
    mangle_string,
    get_random_edit_distance,
    get_homologous_pair,
    get_non_homologous_pair,
    squared_euclidean_distance,
    get_batch,
    estimate_M,
    get_distances,
    approximation_error,
    get_loss,
    validate_training_data,
    run_experiment,
)

__all__ = [
    "pad_with_null",
    "string_to_tensor",
    "random_char",
    "random_str",
    "mangle_string",
    "get_random_edit_distance",
    "get_homologous_pair",
    "get_non_homologous_pair",
    "squared_euclidean_distance",
    "get_batch",
    "estimate_M",
    "get_distances",
    "approximation_error",
    "get_loss",
    "validate_training_data",
    "run_experiment",
]
