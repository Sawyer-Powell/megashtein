#!/usr/bin/env python3
"""
Command line training script for the megashtein model.

This script provides a simple command line interface for training the model.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from megashtein.training.train_model import train

if __name__ == "__main__":
    train()
