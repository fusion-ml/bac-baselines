import hydra
import numpy as np
import gym
import barl.envs
from barl.util.misc_util import Dumper
from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController
from pilco.rewards import ExponentialReward
import tensorflow as tf
from tqdm import trange
from gpflow import set_trainable
# from tensorflow import logging

def rollout(env, pilco, timesteps, verbose=True, random=False, SUBS=1, render=False):
        X = []; Y = [];
        x = env.reset()
        ep_return_full = 0
        ep_return_sampled = 0
        for timestep in range(timesteps):
            if render: env.render()
            u = policy(env, pilco, x, random)
            for i in range(SUBS):
                x_new, r, done, _ = env.step(u)
                ep_return_full += r
                if done: break
                if render: env.render()
            if verbose:
                print("Action: ", u)
                print("State : ", x_new)
                print("Return so far: ", ep_return_full)
            X.append(np.hstack((x, u)))
            Y.append(x_new - x)
            ep_return_sampled += r
            x = x_new
            if done: break
        return np.stack(X), np.stack(Y), ep_return_sampled, ep_return_full


def policy(env, pilco, x, random):
    if random:
        return env.action_space.sample()
    else:
        return pilco.compute_action(x[None, :])[0, :]

@hydra.main(config_path='cfg', config_name='pilco')
def main(config):
    # set seeds
    np.random.seed(config.seed)
    tf.random.set_seed(config.seed)
    env = gym.make(config.env.name)
    dumper = Dumper(config.name)
    horizon = config.env.max_path_length
    # Initial random rollouts to generate a dataset
    X,Y, _, _ = rollout(env=env, pilco=None, random=True, timesteps=horizon, render=False, verbose=False)
    for i in range(config.init_random_rollouts):
        X_, Y_, _, _ = rollout(env=env, pilco=None, random=True,  timesteps=horizon, render=False, verbose=False)
        X = np.vstack((X, X_))
        Y = np.vstack((Y, Y_))


    state_dim = Y.shape[1]
    control_dim = X.shape[1] - state_dim
    controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=config.num_basis_functions)
    # controller = LinearController(state_dim=state_dim, control_dim=control_dim)

    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    pilco = PILCO((X, Y), controller=controller, horizon=horizon)
    # Example of user provided reward function, setting a custom target state
    # R = ExponentialReward(state_dim=state_dim, t=np.array([0.1,0,0,0]))
    # pilco = PILCO(X, Y, controller=controller, horizon=40, reward=R)

    for rollouts in range(config.num_rl_trials):
        pilco.optimize_models()
        pilco.optimize_policy()
        eval_returns = []
        pbar = trange(config.num_eval_trials)
        for _ in trange(config.num_eval_trials):
            X_new, Y_new, _, ep_return = rollout(env=env, pilco=pilco, timesteps=horizon, render=False, verbose=False)
            eval_returns.append(ep_return)
        stats = {"Mean Return": np.mean(eval_returns), "Std Return:": np.std(eval_returns)}
        pbar.set_postfix(stats)
        dumper.add("Eval Returns", eval_returns)
        dumper.add("Eval ndata", X.shape[0])
        dumper.save()
        # Update dataset
        X = np.vstack((X, X_new)); Y = np.vstack((Y, Y_new))
        pilco.mgpr.set_data((X, Y))


if __name__ == '__main__':
    main()
