import os
import sys

import torch

# No GPU -> run kernels in Triton's CPU interpreter (correctness only, not speed).
if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
