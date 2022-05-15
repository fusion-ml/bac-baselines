import hydra
import gym
import numpy as np
# from dragonfly import maximise_function
from skopt import gp_minimize
import barl.envs
from functools import partial
from tqdm import trange
from barl.util.misc_util import Dumper
from rlkit.envs.wrappers import NormalizedBoxEnv



def eval_function(action_sequence, env, start_state):
    action_dim = env.action_space.low.size
    action_sequence = np.array(action_sequence).reshape(env.horizon, action_dim)
    obs = env.reset(obs=start_state)
    # todo maybe reset aciton sequence
    total_rew = 0.
    for action in action_sequence:
        obs, rew, done, info = env.step(action)
        total_rew += rew
    return total_rew

def neg_eval_function(action_sequence, env, start_state):
    return -1 * eval_function(action_sequence, env, start_state)


@hydra.main(config_path="cfg", config_name="bayes_opt")
def main(config):
    import barl.envs
    np.random.seed(config.seed + 15)
    env = NormalizedBoxEnv(gym.make(config.env.name))
    env.seed(config.seed)
    start_state = env.reset()
    action_dim = env.action_space.low.size
    horizon = env.horizon
    domain = [[-1, 1]] * action_dim * horizon
    print(f"{start_state=}")
    dumper = Dumper(config.name)
    objfn = partial(neg_eval_function, env=env, start_state=start_state)
    for capital in trange(1, config.max_episodes):
        res = gp_minimize(objfn,
                          domain,
                          n_initial_points=1,
                          n_calls=capital,
                          random_state=config.seed)
        # max_val, max_pt, history = maximise_function(objfn, domain, capital)
        # dumper.add("Eval Returns", [max_val])
        dumper.add("Eval Returns", [-1 * res.fun])
        dumper.add("Eval ndata", horizon * capital)
        dumper.save()

if __name__ == '__main__':
    main()
