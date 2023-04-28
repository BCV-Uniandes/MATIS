import traceback
import numpy as np
import json
from tqdm import tqdm
import os.path as osp

def eval_classification(preds):
    """
    Presence recognition evaluation is not implemented. Return 0.1 always.
    """
    return 0.1, 0.1, 0.1