"""
Use SAC on BAC tasks
"""
import hydra
import torch
import gym
import bax.envs
from bax.util.misc_util import Dumper
import numpy as np

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.core import logger
from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.gaussian_strategy import GaussianStrategy
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.td3.td3 import TD3Trainer
from rlkit.torch.networks import FlattenMlp, TanhMlpPolicy
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm


def experiment(env_name, variant):
    eval_env = NormalizedBoxEnv(gym.make(env_name))
    expl_env = NormalizedBoxEnv(gym.make(env_name))
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    M = variant['layer_size']
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        **variant['policy_kwargs']
    )
    target_policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        **variant['policy_kwargs']
    )
    es = GaussianStrategy(
        action_space=expl_env.action_space,
        max_sigma=0.1,
        min_sigma=0.1,  # Constant sigma
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    eval_path_collector = MdpPathCollector(
        eval_env,
        policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        exploration_policy,
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    trainer = TD3Trainer(
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        target_policy=target_policy,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


@hydra.main(config_path='cfg', config_name='rlkit')
def main(config):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    variant = dict(
        algorithm="TD3",
        version="normal",
        layer_size=config.alg.layer_size,
        replay_buffer_size=int(1E6),
        algorithm_kwargs=dict(
            num_epochs=config.num_epochs,
            num_eval_steps_per_epoch=1000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=0,
            num_train_loops_per_epoch=1,
            max_path_length=config.env.max_path_length,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=config.discount,
        ),
        policy_kwargs=dict(
            hidden_sizes=[config.alg.layer_size, config.alg.layer_size],
        ),
    )
    logger.reset()
    setup_logger(config.name, variant=variant, log_dir='.')
    ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(config.env.name, variant)



if __name__ == "__main__":
    main()
