import numpy as np
import gym
from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController
from pilco.rewards import ExponentialReward
import barl.envs
import tensorflow as tf
from gpflow import set_trainable
np.random.seed(0)

# NEEDS a different initialisation than the one in gym (change the reset() method),
# to (m_init, S_init), modifying the gym env

# Introduces subsampling with the parameter SUBS and modified rollout function
# Introduces priors for better conditioning of the GP model
# Uses restarts

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

class myCartpole():
    def __init__(self):
        self.env = gym.make('pilcocartpole-v0')
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        obs = self.env.reset()
        return obs

    def render(self):
        self.env.render()

if __name__=='__main__':
    SUBS=1
    bf = 30
    maxiter=50
    max_action=1.0
    target = np.array([0, 0, 0, 0]).astype(np.float64)
    weights = np.diag([1.0, 0, 1, 0]).astype(np.float64)
    m_init = np.reshape([0, 0, np.pi, 0.0], (1,4)).astype(np.float64)
    S_init = np.diag([0.02, 0.02, 0.02, 0.02]).astype(np.float64)
    T = 25
    T_sim = T
    J = 4
    N = 30
    restarts = 2

    env = myCartpole()

    # Initial random rollouts to generate a dataset
    X, Y, _, _ = rollout(env, None, timesteps=T, random=True, SUBS=SUBS, render=False)
    for i in range(1,J):
        X_, Y_, _, _ = rollout(env, None, timesteps=T, random=True, SUBS=SUBS, verbose=True, render=False)
        X = np.vstack((X, X_))
        Y = np.vstack((Y, Y_))

    state_dim = Y.shape[1]
    control_dim = X.shape[1] - state_dim

    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=bf, max_action=max_action)
    R = ExponentialReward(state_dim=state_dim, t=target, W=weights)

    pilco = PILCO((X, Y), controller=controller, horizon=T, reward=R, m_init=m_init, S_init=S_init)

    # for numerical stability, we can set the likelihood variance parameters of the GP models
    for model in pilco.mgpr.models:
        model.likelihood.variance.assign(0.001)
        set_trainable(model.likelihood.variance, False)

    r_new = np.zeros((T, 1))
    for rollouts in range(N):
        print("**** ITERATION no", rollouts, " ****")
        pilco.optimize_models(maxiter=maxiter, restarts=2)
        pilco.optimize_policy(maxiter=maxiter, restarts=2)

        X_new, Y_new, _, _ = rollout(env, pilco, timesteps=T_sim, verbose=True, SUBS=SUBS, render=False)
        breakpoint()
        X_new = X_new.astype(np.float64)
        Y_new = Y_new.astype(np.float64)

        # Since we had decide on the various parameters of the reward function
        # we might want to verify that it behaves as expected by inspection
        for i in range(len(X_new)):
                r_new[:, 0] = R.compute_reward(X_new[i,None,:-1], 0.001 * np.eye(state_dim))[0]
        total_r = sum(r_new)
        _, _, r = pilco.predict(X_new[0,None,:-1], 0.001 * S_init, T)
        print("Total ", total_r, " Predicted: ", r)

        # Update dataset
        X = np.vstack((X, X_new)); Y = np.vstack((Y, Y_new))
        pilco.mgpr.set_data((X, Y))
