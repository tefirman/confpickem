import os
import sys

# Get the absolute path of the project root directory
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, "src"))