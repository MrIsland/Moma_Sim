import yaml
import numpy as np
import open3d as o3d
import json
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import logging
import pickle as pkl
import os, sys
import cv2
import torch
import time

import rospy
import rospkg

logger = logging.getLogger(__name__)
def load_config(path, default_path=None):
    """
    Loads config file.
    Args:
        path (str): path to config file.
        default_path (str, optional): whether to use default path. Defaults to None.
    Returns:
        cfg (dict): config dict.
    """
    # load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.full_load(f)
    
    # check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')
    package_name = cfg_special.get('package_name')
    # if yes, load this config first as default
    # if no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.full_load(f)
    else:
        cfg = dict()

    # include main configuration
    update_recursive(cfg, cfg_special)


    if package_name is not None:
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path(package_name)
        resolve_relative_paths(cfg, pkg_path)

    return cfg

def update_recursive(dict1, dict2):
    """
    Update two config dictionaries recursively.
    Args:
        dict1 (dict): first dictionary to be updated.
        dict2 (dict): second dictionary which entries should be used.
    """
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


def resolve_relative_paths(cfg, base_path):
    """
    Recursively resolve relative paths in cfg using base_path.
    
    Any value that is a string and starts with ./ or ../ or is just a relative path
    will be replaced by an absolute path based on base_path.
    """
    for k, v in cfg.items():
        if isinstance(v, str):
            if not os.path.isabs(v):
                # Skip ROS topic names like '/camera/image_raw'
                if v.startswith('./') or v.startswith('../'):
                    cfg[k] = os.path.normpath(os.path.join(base_path, v))
        elif isinstance(v, dict):
            resolve_relative_paths(v, base_path)
        elif isinstance(v, list):
            for i in range(len(v)):
                if isinstance(v[i], str) and not os.path.isabs(v[i]) and not v[i].startswith('/'):
                    v[i] = os.path.normpath(os.path.join(base_path, v[i]))
                elif isinstance(v[i], dict):
                    resolve_relative_paths(v[i], base_path)

def load_opt_from_config_files(conf_files):
    """
    Load opt from the config files, settings in later files can override those in previous files.

    Args:
        conf_files (list): a list of config file paths

    Returns:
        dict: a dictionary of opt settings
    """
    opt = {}
    for conf_file in conf_files:
        with open(conf_file, encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        load_config_dict_to_opt(opt, config_dict)

    return opt


def load_config_dict_to_opt(opt, config_dict):
    """
    Load the key, value pairs from config_dict to opt, overriding existing values in opt
    if there is any.
    """
    if not isinstance(config_dict, dict):
        raise TypeError("Config must be a Python dictionary")
    for k, v in config_dict.items():
        k_parts = k.split('.')
        pointer = opt
        for k_part in k_parts[:-1]:
            if k_part not in pointer:
                pointer[k_part] = {}
            pointer = pointer[k_part]
            assert isinstance(pointer, dict), "Overriding key needs to be inside a Python dict."
        ori_value = pointer.get(k_parts[-1])
        pointer[k_parts[-1]] = v
        if ori_value:
            logger.warning(f"Overrided {k} from {ori_value} to {pointer[k_parts[-1]]}")

def to_tensor(numpy_array, device=None):
    if isinstance(numpy_array, torch.Tensor):
        return numpy_array
    if device is None:
        return torch.from_numpy(numpy_array)
    else:
        return torch.from_numpy(numpy_array).to(device)

def to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()