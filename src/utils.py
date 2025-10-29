"""
Utility functions for the Predictive Delivery Optimizer
"""
import pandas as pd
import numpy as np
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    import random
    random.seed(seed)

def print_section(title: str) -> None:
    """Print a formatted section title"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def save_json(data: Dict[Any, Any], filepath: str) -> None:
    """Save dictionary to JSON file"""
    import json
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(filepath: str) -> Dict[Any, Any]:
    """Load JSON file to dictionary"""
    import json
    with open(filepath, 'r') as f:
        return json.load(f)
