import importlib
import os
from omegaconf import OmegaConf
import torch
import numpy as np
import pickle

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def get_saved(saved_folder_path, name, log=False, save_path=None, is_backup=False):
    path = os.path.join(saved_folder_path, f"{name}.pkl")
    with open(path, "rb") as f:
        obj =  pickle.load(f)
    if log and save_path is not None and is_backup:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        path = os.path.join(save_path, f"{name}.pkl")
        with open(path, 'wb') as file:
            pickle.dump(obj, file)                
    return obj