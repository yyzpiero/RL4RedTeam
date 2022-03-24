from itertools import zip_longest
import numpy as np
import gym
from typing import Dict, Iterable, Optional, Union
import torch


def test_game(env, agent, n_eval_episodes):
    """
    Test the agent to measure rewards.
    It evlautes the agent with the environment for some epidoses,
    and record totol rewards.

    :param env: environment to intereact with
    :param agent: agent
    :param n_eval_episodes: number of episodes
    """
    all_episode_reward = []
    all_episode_steps = []
    for _ in range(n_eval_episodes):
        obs = env.reset()
        episode_rewards = []
        done = False
        steps = 0
        while not done:
            action = agent.e_greedy_act(obs)
            #action = agent.act(obs)
            next_obs, reward, done, _ = env.step(action)
            obs = next_obs
            # print("action:{},reward{}".format(action, reward))

            episode_rewards.append(reward)
            steps += 1

        all_episode_reward.append(sum(episode_rewards))
        all_episode_steps.append(steps)

    mean_episode_reward = np.mean(all_episode_reward)
    mean_episode_reward_std = np.std(all_episode_reward)
    mean_episode_step = np.mean(all_episode_steps)
    mean_episode_step_std = np.std(all_episode_steps)
    # obs = env.reset()

    # if rewards >= 500:
    #     reward_games.append(rewards)
    #     obs = env.reset()
    #     break
    # print(reward_games)
    return mean_episode_reward, mean_episode_reward_std, mean_episode_step, mean_episode_step_std


def get_device(device: Union[torch.device, str] = "auto") -> torch.device:
    """
    Retrieve PyTorch device.
    It checks that the requested device is available first.
    For now, it supports only cpu and cuda.
    By default, it tries to use the gpu.
    :param device: One for 'auto', 'cuda', 'cpu'
    :return:
    """
    # Cuda by default
    if device == "auto":
        device = "cuda"
    # Force conversion to th.device
    device = torch.device(device)

    # Cuda not available
    if device.type == torch.device(
            "cuda").type and not torch.cuda.is_available():
        return torch.device("cpu")

    return device


def zip_strict(*iterables: Iterable) -> Iterable:
    """
    ``zip()`` function but enforces that iterables are of equal length.
    Raises ``ValueError`` if iterables not of equal length.
    Code inspired by Stackoverflow answer for question #32954486.
    :param \*iterables: iterables to ``zip()``
    """
    # As in Stackoverflow #32954486, use
    # new object for "empty" in case we have
    # Nones in iterable.
    sentinel = object()
    for combo in zip_longest(*iterables, fillvalue=sentinel):
        if sentinel in combo:
            raise ValueError("Iterables have different lengths")
        yield combo


def polyak_update(
    params: Iterable[torch.nn.Parameter],
    target_params: Iterable[torch.nn.Parameter],
    tau: float = 1,
) -> None:
    """
    Perform a Polyak average update on ``target_params`` using ``params``:
    target parameters are slowly updated towards the main parameters.
    ``tau``, the soft update coefficient controls the interpolation:
    ``tau=1`` corresponds to copying the parameters to the target ones whereas nothing happens when ``tau=0``.
    The Polyak update is done in place, with ``no_grad``, and therefore does not create intermediate tensors,
    or a computation graph, reducing memory cost and improving performance.  We scale the target params
    by ``1-tau`` (in-place), add the new weights, scaled by ``tau`` and store the result of the sum in the target
    params (in place).
    See https://github.com/DLR-RM/stable-baselines3/issues/93
    :param params: parameters to use to update the target params
    :param target_params: parameters to update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    """
    with torch.no_grad():
        # zip does not raise an exception if length of parameters does not match.
        for param, target_param in zip_strict(params, target_params):
            target_param.data.mul_(1 - tau)
            torch.add(target_param.data,
                      param.data,
                      alpha=tau,
                      out=target_param.data)
