import numpy as np
import gym
import bax.envs  # NOQA
import torch
from bax.envs.pilco_cartpole import CartPoleSwingUpEnv


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)


def get_pole_pos(x):
    xpos = x[..., 0]
    theta = x[..., 2]
    pole_x = CartPoleSwingUpEnv.POLE_LENGTH*torch.sin(theta)
    pole_y = CartPoleSwingUpEnv.POLE_LENGTH*torch.cos(theta)
    position = torch.stack((xpos + pole_x, pole_y))
    return position.T


def pets_cartpole_reward(actions, next_obs):
    device = actions.device
    position = get_pole_pos(next_obs)
    goal = torch.Tensor([[0.0, CartPoleSwingUpEnv.POLE_LENGTH]]).to(device)
    squared_distance = torch.sum((position - goal)**2, axis=-1)
    squared_sigma = 0.25**2
    costs = 1 - torch.exp(-0.5*squared_distance/squared_sigma)
    return -costs[:, None]


def pets_pend_reward(actions, next_obs):
    th = next_obs[..., 0]
    thdot = next_obs[..., 1]
    u = actions[..., 0]
    costs = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)
    return -costs[None, :]


def pets_reacher_reward(actions, next_obs):
    pass


reward_functions = {
        'bacpendulum-v0': pets_pend_reward,
        'pilcocartpole-v0': pets_cartpole_reward,
        'reacher-v0': pets_reacher_reward,
        }


def test_env(name, fn):
    env = gym.make(name)
    env.reset()
    next_obss = []
    actions = []
    rewards = []
    for _ in range(env.horizon):
        action = env.action_space.sample()
        next_obs, rew, done, info = env.step(action)
        next_obss.append(next_obs)
        actions.append(action)
        rewards.append(rew)
    next_obss = torch.Tensor(next_obss)
    actions = torch.Tensor(actions)
    rewards = torch.Tensor(rewards)[None, :]
    pred_rewards = fn(actions, next_obss)
    assert torch.allclose(pred_rewards, rewards), f'pred_rewards: {pred_rewards}\nrewards: {rewards}'


def test_cartpole():
    pass


def test_reacher():
    pass


if __name__ == '__main__':
    test_env('pilcocartpole-v0', pets_cartpole_reward)
    test_env('bacpendulum-v0', pets_pend_reward)
    test_env('reacher-v0', pets_reacher_reward)
