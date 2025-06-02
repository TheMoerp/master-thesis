import sys
import os

# Add the env_packages directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "env_packages"))

# Try importing the packages
try:
    import torch
    import numpy as np
    import nibabel as nib
    import matplotlib.pyplot as plt
    import sklearn
    import tqdm
    import pandas as pd
    import seaborn as sns
    
    print("All packages imported successfully!")
    
    # Print versions
    print(f"torch version: {torch.__version__}")
    print(f"numpy version: {np.__version__}")
    print(f"nibabel version: {nib.__version__}")
    print(f"matplotlib version: {plt.matplotlib.__version__}")
    print(f"sklearn version: {sklearn.__version__}")
    print(f"pandas version: {pd.__version__}")
    print(f"seaborn version: {sns.__version__}")
    
except ImportError as e:
    print(f"Error importing packages: {e}") 