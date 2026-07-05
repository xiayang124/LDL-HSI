import json
from typing import Dict, List, Tuple

import numpy as np
import scipy.io as io
from sklearn.decomposition import PCA
import torch
import random
from itertools import product
from pathlib import Path


def read_mat(mat_dir: str) \
        -> np.ndarray:
    """
    Get mat data.
    -----------------------------------------
    Arguments:
        mat_dir(str): Dir about mat file.
    -----------------------------------------
    Returns: ndarray data.
    """
    mat = io.loadmat(mat_dir)
    for data in mat.values():
        if type(data) is np.ndarray:
            numpy_data = np.array(data)
            return numpy_data
    raise Exception("Wrong mat file!")

def ensure_dir(path: str):
    """
    Ensure the directory exists, if not, create it.
    -------------------------------------------------
    Arguments:
        dir_path: Directory path.
    """
    Path(path).mkdir(parents=True, exist_ok=True)

def normalize(data, if_255=False) \
        -> np.ndarray:
    """
    Norm data and if set range to [0, 255].
    -------------------------------------------------
    Arguments:
        data: Source data.
    -------------------------------------------------
    Returns: 
        norm_data: Norm data or norm to [0, 255] data
    """
    norm_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return (norm_data * 255).astype(np.uint8) if if_255 else norm_data


def get_train_test_num(gt, train_num, least=15) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Get per class train num and test num.
    -------------------------------------------------
    Arguments:
        gt: All ground truth
        train_num: The train num of per class.
        least: The least of train num
    -------------------------------------------------
    Returns: 
        Train num and test num of per class.
    """
    # Get class number(int(ndarray)) and per class num(ndarray)
    class_num = np.max(gt)
    per_class_num = np.array([np.sum(gt == classes) for classes in range(1, class_num + 1)])
    
    # if train_num is less than 1, treat it as a ratio
    if train_num < 1:
        theo_num = (per_class_num * train_num).astype(int)
        train_list = np.where(per_class_num > theo_num + 50, theo_num, least)
    # if train_num is 1, set all train to 1
    elif train_num == 1:
        train_list = np.ones(class_num, int)
    # if train_num is greater than 1, set it as a fixed number
    elif train_num > 1:
        tn = int(train_num)
        train_list = np.where(per_class_num > tn, tn, (per_class_num // 2).astype(int))
    else:
        raise ValueError("train_num must be a positive number or ratio less than 1")
    test_list = per_class_num - train_list
    return train_list, test_list

def split_gt(gt, train_list) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Split ground truth into train and test set.
    -------------------------------------------------
    Arguments:
        gt: Ground truth data.
        train_list: Train number of each class.
    -------------------------------------------------
    Returns:
        out_train: Train ground truth set of ground truth.
        out_test: Test ground truth set of ground truth.
    """
    # Get coordinates and labels
    coords = np.argwhere(gt > 0)
    labels = gt[tuple(coords.T)]
    out_train, out_test = np.zeros_like(gt), np.zeros_like(gt)

    # Split the coordinates into train and test sets
    for lbl, num in enumerate(train_list, start=1):
        idx = np.flatnonzero(labels == lbl)
        np.random.shuffle(idx)
        out_train[tuple(coords[idx[:num]].T)] = lbl
        out_test[tuple(coords[idx[num:]].T)]  = lbl
    return out_train, out_test

# Convert gpu tensor to numpy array
def gpu2numpy(data: torch.Tensor):
    """
    Convert a PyTorch tensor to a NumPy array.
    """
    return data.detach().cpu().numpy()

# Read json config file
def get_json_config(dirs: str):
    """
    Read json config file and return as a dictionary.
    """
    with open(dirs, "r") as f:
        params = json.loads(f.read())
    return params

# Generate combinations of multiple arguments
def combinations(multiple_args: Dict[any, List[any]]):
    """
    Generate combinations of multiple arguments.
    """
    return (dict(zip(multiple_args, v)) for v in product(*multiple_args.values()))

def fix_seed(seed: int = 42):
    """
    Fix random seed for reproducibility.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def apply_pca(image, n_components=3):
    """
    Apply PCA to the input image.
    """
    # Reshape the image to 2D
    h, w, c = image.shape
    image_2d = image.reshape(-1, c)

    # Apply PCA
    pca = PCA(n_components=n_components)
    image_pca = pca.fit_transform(image_2d)

    # Reshape back to original
    image_pca = image_pca.reshape(h, w, n_components)
    return image_pca
