---
icon: folder-arrow-down
description: Here, we provide the required steps in order to install TMLL on your machine.
---

# Installation

### Install from PyPI

TMLL is currently available through the **Test PyPI** repository. To install it, you may use the following command (from _pip_):

```bash
pip3 install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple tmll
```

### Install from the Source Code

If you are eager to install TMLL with its latest changes in its repository, you may follow the below steps to install it properly:

```bash
# Clone TMLL from Git
git clone https://github.com/eclipse-tracecompass/tmll.git
cd tmll

# Clone its submodule(s)
git submodule update --init

# Create a virtual environment (if haven't already)
python3 -m venv venv
source venv/bin/activate # If Linux or MacOS
.\venv\Scripts\activate # If Windows 

# Install the required dependencies
(venv) pip3 install -r requirements.txt
```

If you install TMLL from the source code, you need to add these lines before importing TMLL's library (i.e., in your source code).

```python
import sys
sys.path.append("tmll/tsp")
```

