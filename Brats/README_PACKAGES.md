# Package Setup Instructions

This project uses a local package directory (`env_packages`) to avoid Windows long path issues.

## Setup Information

The packages have been installed locally in the `env_packages` directory using:
```
pip install --target="C:\Users\Matthias Arbeit\Documents\GitHub\master-thesis\Brats\env_packages" torch numpy nibabel matplotlib scikit-learn tqdm pandas seaborn
```

## How to Use the Packages

### Option 1: Import in Python Scripts 
Add this code at the beginning of your Python scripts:

```python
import sys
import os

# Add the env_packages directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "env_packages"))

# Now you can import the packages normally
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import sklearn
import tqdm
import pandas as pd
import seaborn as sns
```

### Option 2: Set PYTHONPATH Environment Variable

You can set the PYTHONPATH environment variable before running your scripts:

```
set PYTHONPATH=%PYTHONPATH%;C:\Users\Matthias Arbeit\Documents\GitHub\master-thesis\Brats\env_packages
```

Then run your Python scripts normally.

## Alternative Long-Term Solution

If possible, enable Windows Long Path support:

1. Press Win+R, type "gpedit.msc" and press Enter
2. Navigate to Computer Configuration → Administrative Templates → System → Filesystem
3. Find and enable "Enable Win32 long paths"
4. Reboot your computer

Alternatively, you can enable it using the registry:
```
reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f
```

Note: This requires administrative privileges.

## Installed Package Versions

- torch: 2.7.0+cpu
- numpy: 2.2.5
- nibabel: 5.3.2
- matplotlib: 3.10.3
- scikit-learn: 1.6.1
- pandas: 2.2.3
- seaborn: 0.13.2
- tqdm: 4.67.1 