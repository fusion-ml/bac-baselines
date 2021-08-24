import gym
import bax.util.envs
from bax.util.misc_util import Dumper
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
import hydra


@hydra.main(config_path='cfg', config_name='config')
def main(config):
    env = gym.make(config.env.name)
    eval_env = gym.make(config.env.name)
    check_env(env)
    model_classes = {'PPO': PPO, 'SAC': SAC, 'TD3': TD3}
    model_class = model_classes[config.alg.name]
    model = model_class('MlpPolicy', env)
    eval_callback = EvalCallback(eval_env, best_model_save_path='.',
                                 log_path='.', eval_freq=config.eval_frequency,
                                 deterministic=True, render=False, n_eval_episodes=config.num_eval_trials)
    model.learn(total_timesteps=config.env.timesteps, callback=eval_callback)


if __name__ == '__main__':
    main()
