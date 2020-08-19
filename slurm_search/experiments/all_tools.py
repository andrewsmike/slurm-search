from pprint import pprint
from time import sleep
from random import random

import numpy as np

from all.environments import AtariEnvironment, GymEnvironment
from all.experiments import SingleEnvExperiment, ParallelEnvExperiment
from all.presets import atari, classic_control, continuous

from slurm_search.experiment import accepts_param_names


## ALL target function
def run_results(env, agent, hp, run_params, run_seed):

    if agent.startswith("debug"):
        sleep(1)
        return {
            "return_mean": random(),
        }

    train_frames = run_params["train_frames"]
    train_episodes = run_params["train_episodes"]
    test_episodes = run_params["test_episodes"]

    agent_type, agent_name = agent.split(":")
    env_type, env_name = env.split(":")

    env_func = {
        "classic": GymEnvironment,
        "continuous": GymEnvironment,
        "atari": AtariEnvironment,
    }[env_type]

    env = env_func(env_name, device="cuda")
    if env_name == "MountainCar-v0":
        env._env._max_episode_steps = 20000

    agent_mod = {
        "classic": classic_control,
        "continuous": continuous,
        "atari": atari,
    }[agent_type]
    agent_func = getattr(agent_mod, agent_name)

    int_hp_keys = {
        "minibatch_size",
        "update_frequency",
    }

    hp = {
        hp_key: (int(hp_value) if hp_key.startswith("n_") or hp_key in int_hp_keys else hp_value)
        for hp_key, hp_value in hp.items()
    }

    agent = agent_func(
        device="cuda",
        **hp,
    )

    if isinstance(agent, tuple):
        experiment = ParallelEnvExperiment(
            agent,
            env,
            render=False,
            quiet=True,
            write_loss=False,
        )
    else:
        experiment = SingleEnvExperiment(
            agent,
            env,
            render=False,
            quiet=True,
            write_loss=False,
        )

    experiment.train(
        frames=train_frames,
        episodes=train_episodes or np.inf,
    )
    returns = experiment.test(
        episodes=test_episodes,
    )[:test_episodes] # ALL BUG: May return >test_episodes episodes.
    del experiment

    returns = np.array(returns)

    return {
        "return_mean": returns.mean(),
        "return_var": returns.std(),
        "returns_cdf": np.sort(returns),
    }

@accepts_param_names
def return_mean(env, agent, hp, run_params, run_seed):
    return run_results(env, agent, hp, run_params, run_seed)["return_mean"]

@accepts_param_names
def return_var(env, agent, hp, run_params, run_seed):
    return run_results(env, agent, hp, run_params, run_seed)["return_var"]

@accepts_param_names
def return_cdf(env, agent, hp, run_params, run_seed):
    return run_results(env, agent, hp, run_params, run_seed)["return_cdf"]

