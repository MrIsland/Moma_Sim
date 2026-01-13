import os, sys
import pdb

import numpy as np
import random
import hydra
import isaacgymenvs
from utils.dagger import set_np_formatting
from utils.process_sarl import process_dagger_with_dataset, process_dagger_with_model
from omegaconf import DictConfig, OmegaConf

def train(cfg):
    print("Algorithm:    {}".format(cfg.learn.algo))
    if cfg.learn.algo in ['dagger_with_model', 'dagger_with_dataset']:
        sarl = eval('process_{}'.format(cfg.learn.algo))(cfg)
        iterations = cfg['learn']['max_iterations']
        sarl.run(log_interval=cfg.learn.save_interval)
    else:
        print("Unrecognized algorithm!")

@hydra.main(version_base="1.1", config_path="./cfg", config_name="config_dagger")
def launch_rlg_hydra(cfg: DictConfig):
    set_np_formatting()
    train(cfg)

if __name__ == '__main__':
    launch_rlg_hydra()
