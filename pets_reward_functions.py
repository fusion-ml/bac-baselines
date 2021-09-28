import numpy as np
import gym
import bax.envs  # NOQA
import torch
from torch import Tensor
from bax.envs.pilco_cartpole import CartPoleSwingUpEnv
from bax.envs.lava_path import in_lava, LavaPathEnv


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
    goal = Tensor([[0.0, CartPoleSwingUpEnv.POLE_LENGTH]]).to(device)
    squared_distance = torch.sum((position - goal)**2, axis=-1)
    squared_sigma = 0.25**2
    costs = 1 - torch.exp(-0.5*squared_distance/squared_sigma)
    return -costs[:, None]


def pets_pend_reward(actions, next_obs):
    th = next_obs[..., 0]
    thdot = next_obs[..., 1]
    u = actions[..., 0]
    costs = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)
    return -costs[:, None]


def pets_reacher_reward(actions, next_obs):
    reward_ctrl = -torch.square(actions).sum(axis=-1)
    vec = next_obs[..., -2:]
    reward_dist = -torch.linalg.norm(vec, dim=-1)
    reward = reward_ctrl + reward_dist
    return reward[:, None]


def in_lava(x):
    lava_pits = [Tensor([[-10, -8], [-0.5, 8]]).to(x.device), Tensor([[0.5, -8], [10, 8]]).to(x.device)]
    lava = Tensor([0] * x.shape[0]).bool().to(x.device)
    for pit in lava_pits:
        above_bottom = torch.all(x > pit[0, :], dim=1)
        below_top = torch.all(x < pit[1, :], dim=1)
        lava |= (above_bottom & below_top)
    return lava

def pets_lava_path_reward(actions, next_obs):
    device = actions.device
    x_prob = next_obs[..., :2]
    lava = in_lava(x_prob)
    goal = Tensor(LavaPathEnv.goal).to(device)
    reward = -torch.sum((x_prob - goal) ** 2, axis=-1) / 800 + LavaPathEnv.lava_penalty * lava.int()
    return reward[:, None]


def pets_beta_tracking_reward(actions, next_obs, target=2):
    BETA_IDX = 0
    betas = next_obs[..., BETA_IDX]
    iqr = 0.8255070447921753
    median = 1.622602
    betas = betas * iqr + median
    return -1 * torch.abs(betas - target)[:, None]


reward_functions = {
        'bacpendulum-v0': pets_pend_reward,
        'pilcocartpole-v0': pets_cartpole_reward,
        'bacreacher-v0': pets_reacher_reward,
        'bacreacher-tight-v0': pets_reacher_reward,
        'lavapath-v0': pets_lava_path_reward,
        'betatracking-v0': pets_beta_tracking_reward,
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
    next_obss = Tensor(next_obss)
    actions = Tensor(actions)
    rewards = Tensor(rewards)[:, None]
    pred_rewards = fn(actions, next_obss)
    assert torch.allclose(pred_rewards, rewards), f'pred_rewards: {pred_rewards}\nrewards: {rewards}'
    print(f'passed for {name}')




if __name__ == '__main__':
    test_env('lavapath-v0', pets_lava_path_reward)
    test_env('pilcocartpole-v0', pets_cartpole_reward)
    test_env('bacpendulum-v0', pets_pend_reward)
    test_env('bacreacher-v0', pets_reacher_reward)
    test_env('betatracking-v0', pets_beta_tracking_reward)
