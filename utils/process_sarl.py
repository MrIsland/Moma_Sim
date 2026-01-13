import os, sys
import pdb

import numpy as np
from datetime import datetime
import time

from omegaconf import DictConfig, OmegaConf
from utils.dagger import DAggerWithDatasetAgent, DAggerWithModelAgent
from isaacgymenvs.utils.utils import set_np_formatting, set_seed
import gym
import isaacgymenvs
from rl_games.common import env_configurations, vecenv
from isaacgymenvs.tasks import isaacgym_task_map
from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, MultiObserver, ComplexObsRLGPUEnv


def process_dagger_with_dataset(cfg):
    learn_cfg = cfg.learn
    is_train = not cfg.learn.test
    experiment_dir = None
    if is_train:
        experiment_dir = os.path.join('runs', 'dagger', '{}'.format(learn_cfg.algo),
                                      cfg.student.task.name + '_{date:%d-%H-%M-%S}'.format(date=datetime.now()))

        os.makedirs(experiment_dir, exist_ok=True)
        with open(os.path.join(experiment_dir, 'config_dagger.yaml'), 'w') as f:
            f.write(OmegaConf.to_yaml(cfg))

    dagger_with_dataset_agent = DAggerWithDatasetAgent(
        cfg=cfg,
        num_learning_epochs=learn_cfg.noptepochs,
        num_mini_batches=learn_cfg.nminibatches,
        learning_rate=learn_cfg.lr,
        print_log=is_train,
        is_train=is_train,
        logdir=experiment_dir,
        device=cfg.student.rl_device,
    )

    return dagger_with_dataset_agent


def init_sim(cfg):
    global_rank = int(os.getenv("RANK", "0"))
    cfg.student.seed = set_seed(cfg.student.seed, torch_deterministic=cfg.student.torch_deterministic, rank=global_rank)
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{cfg.student.wandb_name}_{time_str}"
    cfg_student = cfg.student
    # cfg.student.task.physics_engine = cfg.physics_engine
    # cfg.student.task.env.numEnvs = cfg.num_envs
    # cfg.student.task.sim.use_gpu_pipeline = cfg.pipeline

    def create_isaacgym_env(**kwargs):
        envs = isaacgymenvs.make(
            cfg_student.seed,
            cfg_student.task_name,
            cfg_student.num_envs,
            cfg_student.sim_device,
            cfg_student.rl_device,
            cfg_student.graphics_device_id,
            cfg_student.headless,
            cfg_student.multi_gpu,
            cfg_student.capture_video,
            cfg_student.force_render,
            cfg_student,
            **kwargs,
        )
        if cfg_student.capture_video:
            envs.is_vector_env = True
            envs = gym.wrappers.RecordVideo(
                envs,
                f"videos/{run_name}",
                step_trigger=lambda step: step % cfg.capture_video_freq == 0,
                video_length=cfg.capture_video_len,
            )
        return envs


    env_configurations.register('rlgpu', {
        'vecenv_type': 'RLGPU',
        'env_creator': lambda **kwargs: create_isaacgym_env(**kwargs),
    })

    vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))


def process_dagger_with_model(cfg):
    learn_cfg = cfg.learn
    is_train = not cfg.learn.test
    experiment_dir = None

    if is_train:
        experiment_dir = os.path.join('runs', 'dagger', '{}'.format(learn_cfg.algo),
                                      cfg.student.task.name + '_{date:%d-%H-%M-%S}'.format(date=datetime.now()))

        os.makedirs(experiment_dir, exist_ok=True)
        with open(os.path.join(experiment_dir, 'config_dagger.yaml'), 'w') as f:
            f.write(OmegaConf.to_yaml(cfg))

    init_sim(cfg)

    dagger_with_model_agent = DAggerWithModelAgent(
        cfg=cfg,
        num_learning_epochs=learn_cfg.noptepochs,
        num_mini_batches=learn_cfg.nminibatches,
        learning_rate=learn_cfg.lr,
        print_log=is_train,
        is_train=is_train,
        logdir=experiment_dir,
        device=cfg.student.rl_device,
    )

    return dagger_with_model_agent
