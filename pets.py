import hydra
import logging
import gym
import torch
import numpy as np
import bax.envs  # NOQA
from tqdm import trange
from bax.util.misc_util import Dumper
import mbrl.models as models
import mbrl.planning as planning
import mbrl.util.common as common_util
from mbrl.env.termination_fns import no_termination
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)


def pets_pend_reward(actions, next_obs):
    th = next_obs[..., 0]
    thdot = next_obs[..., 1]
    u = actions[..., 0]
    costs = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)
    return -costs[:, None]


def pets_cartpole_reward(actions, next_obs):
    pass


def pets_reacher_reward(actions, next_obs):
    pass


reward_functions = {
        'bacpendulum-v0': pets_pend_reward,
        'pilcocartpole-trig-v0': pets_cartpole_reward,
        'reacher-v0': pets_reacher_reward,
        }




@hydra.main(config_path='cfg', config_name='pets')
def main(config):
    dumper = Dumper(config.name)
    config.dynamics_model.model.device = device
    config.agent.optimizer_cfg.device = device
    env = gym.make(config.env.name)
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    # Create a 1-D dynamics model for this environment
    dynamics_model = common_util.create_one_dim_tr_model(config, obs_shape, act_shape)

    # TODO
    reward_fn = reward_functions[config.env.name]
    # Create a gym-like environment to encapsulate the model
    model_env = models.ModelEnv(env, dynamics_model, no_termination, reward_fn)

    replay_buffer = common_util.create_replay_buffer(config, obs_shape, act_shape)

    common_util.rollout_agent_trajectories(
        env,
        config.env.max_path_length,  # initial exploration steps
        planning.RandomAgent(env),
        {},  # keyword arguments to pass to agent.act()
        replay_buffer=replay_buffer,
        trial_length=config.env.max_path_length,
    )

    print("# samples stored", replay_buffer.num_stored)

    agent_cfg = config.agent
    action_lb = env.action_space.low
    action_ub = env.action_space.high
    agent_cfg.optimizer_cfg.lower_bound = np.tile(action_lb, (agent_cfg.planning_horizon, 1)).tolist()
    agent_cfg.optimizer_cfg.upper_bound = np.tile(action_ub, (agent_cfg.planning_horizon, 1)).tolist()
    agent = planning.create_trajectory_optim_agent_for_model(
        model_env,
        agent_cfg,
        num_particles=20
    )
    train_losses = []
    val_scores = []

    def train_callback(_model, _total_calls, _epoch, tr_loss, val_score, _best_val):
        train_losses.append(tr_loss)
        val_scores.append(val_score.mean().item())   # this returns val score per ensemble model

    # Create a trainer for the model
    model_trainer = models.ModelTrainer(dynamics_model, optim_lr=1e-3, weight_decay=5e-5)
    # Main PETS loop
    for trial in range(config.num_trials):
        eval_returns = []
        pbar = trange(config.num_eval_trials)
        for etrial in pbar:
            obs = env.reset()
            agent.reset()

            done = False
            total_reward = 0.0
            steps_trial = 0
            while not done:
                # --------------- Model Training -----------------
                if etrial == 0:
                    if steps_trial == 0:
                        dynamics_model.update_normalizer(replay_buffer.get_all())  # update normalizer stats

                        dataset_train, dataset_val = common_util.get_basic_buffer_iterators(
                            replay_buffer,
                            batch_size=config.overrides.model_batch_size,
                            val_ratio=config.overrides.validation_ratio,
                            ensemble_size=config.dynamics_model.model.ensemble_size,
                            shuffle_each_epoch=True,
                            bootstrap_permutes=False,  # build bootstrap dataset using sampling with replacement
                        )

                        model_trainer.train(
                            dataset_train,
                            dataset_val=dataset_val,
                            num_epochs=50,
                            patience=50,
                            callback=train_callback)

                    # --- Doing env step using the agent and adding to model dataset ---

                    next_obs, reward, done, _ = common_util.step_env_and_add_to_buffer(
                        env, obs, agent, {}, replay_buffer)
                else:
                    action = agent.act(obs)
                    next_obs, reward, done, _ = env.step(action)

                obs = next_obs
                total_reward += reward
                steps_trial += 1

                if steps_trial == config.env.max_path_length:
                    break
            eval_returns.append(total_reward)
            stats = {"Mean Return": np.mean(eval_returns), "Std Return:": np.std(eval_returns)}
            pbar.set_postfix(stats)

        logging.info(f"Trial {trial}, returns_mean={np.mean(eval_returns):.2f}, returns_std={np.std(eval_returns):.2f}, ndata={len(replay_buffer)}")  # NOQA
        dumper.add('Eval Returns', eval_returns)
        dumper.add('Eval ndata', len(replay_buffer))
        dumper.save()


if __name__ == '__main__':
    main()
