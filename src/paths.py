"""
Path utilities for the stock sentiment analysis project
"""
import os

# Get the project root directory (parent of src)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define data and models directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def get_data_path(filename):
    """Get full path for a data file"""
    return os.path.join(DATA_DIR, filename)

def get_model_path(filename):
    """Get full path for a model file"""
    return os.path.join(MODELS_DIR, filename)
