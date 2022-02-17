import hydra
import gym
import numpy as np
from dragonfly import maximise_function
import barl.envs
from barl.util.misc_util import Dumper
from rlkit.envs.wrappers import NormalizedBoxEnv



def eval_function(action_sequence, env, start_state):
    action_dim = env.action_space.low.size
    action_sequence = np.array(action_sequence).reshape(env.horizon, action_dim)
    obs = env.reset(start_state)
    # todo maybe reset aciton sequence
    total_rew = 0.
    for action in action_sequence:
        obs, rew, done, info = env.step(action)
        total_rew += rew
    return total_rew



@hydra.main(config_path="cfg", config_name="bayes_opt")
def main(config):
    np.random.seed(config.seed)
    env.seed(config.seed)
    env = NormalizedBoxEnv(gym.make(config.env_name))
    action_dim = env.action_space.low.size
    horizon = env.horizon
    domain = [[-1, 1]] * action_dim * horizon
    start_state = env.reset()
    dumper = Dumper()
    objfn = partial(eval_function, env=env, start_state=start_state)
    for capital in range(config.max_episodes):
        max_val, max_pt, history = maximise_function(objfn, domain, capital)
        dumper.add("Eval Returns", [max_val])
        dumper.add("Eval ndata", horizon * capital)
        dumper.save()
