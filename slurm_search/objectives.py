"""
Objectives to search over.
Each objective has some callable evaluation function and a CLI invocation
decoding function that generates the appropriate search session config.

ALL OBJECTIVES MUST BE IMPORTED INTO THE MAIN MODULE FOR PICKLE LOADING.
"""
from csv import reader
from math import log
from os.path import join
from pprint import pprint
from time import sleep

from hyperopt import hp
import numpy as np

from all.environments import AtariEnvironment, GymEnvironment
from all.experiments import SingleEnvExperiment, ParallelEnvExperiment
from all.presets import atari, classic_control, continuous

from slurm_search.params import unflattened_params

__all__ = [
    "ale_objective",
    "demo_objective",
    "search_session_args",
]

def search_session_args(session_type, *args):
    return {
        "demo": demo_search_session_args,
        "ale": ale_search_session_args,
    }[session_type](*args)

# Demonstration search: Optimize x^2 + y^2 in [-2, 2]^2 with 4s per eval point.
demo_space = [
    hp.uniform("x", -2, 2),
    hp.uniform("y", -2, 2),
]

def demo_objective(spec):
    x, y = spec
    sleep(4)
    return {
        "loss": x ** 2 + y ** 2,
        "status": "ok",
    }

def demo_search_session_args(*args):
    return {
        "space": demo_space,
        "objective": demo_objective,
        "algo": "tpe",
        "max_evals": 400,
    }

def ale_objective(spec):
    spec, = spec # Unwrap the spec.

    agent_name = spec["agent"]
    env_name = spec["env"]
    agent_type = spec["type"]

    env_func = {
        "classic": GymEnvironment,
        "continuous": GymEnvironment,
        "atari": AtariEnvironment,
    }[agent_type]

    env = env_func(env_name, device="cuda")

    agent_mod = {
        "classic": classic_control,
        "continuous": continuous,
        "atari": atari,
    }[agent_type]
    agent_func = getattr(agent_mod, agent_name)

    agent = agent_func(
        device="cuda",
        **spec["agent_args"],
    )

    returns = []
    for i in range(spec["runs_per_setting"]):
        experiment = ParallelEnvExperiment(
            agent,
            env,
            render=False,
            quiet=True,
            write_loss=False,
        )

        experiment.train(
            frames=spec["frames"],
            episodes=spec.get("episodes", 0) or np.inf,
        )
        returns.append(experiment.test(
            episodes=spec["test_episodes"],
        )[:spec["test_episodes"]])
        # BUG: Sometimes parallel env returns more than this number.
        del experiment

    returns = np.array(returns)
    print(f"Returns: {returns.mean()} +/- {returns.std()}")

    run_axis, return_axis = 0, 1
    return {
        "run_return_means": returns.mean(axis=return_axis),
        "run_return_vars": returns.std(axis=return_axis),
        "returns_mean": returns.mean(),
        "returns_std": returns.std(),
        "loss": - returns.mean(),
        "status": "ok",
    }

def parsed_value(value):
    try:
        return int(value)
    except:
        pass

    try:
        return float(value)
    except:
        pass

    return value

def config_from_args(args):
    return {
        key.lstrip("-"): parsed_value(value)
        for arg in args
        for key, value in (arg.strip().split("="), )
    }


# Example:
# slurm_search.py start ale type=classic agent=a2c env=CartPole-v0
def ale_search_session_args(*args):
    space_spec = {
        "frames": 200000,
        #"episodes": 1000000, # Default of np.inf is best.
        "test_episodes": 100,
        "runs_per_setting": 16,

        "agent_args": {
            "lr": hp.loguniform("lr", log(0.0001), log(0.01)),
            "entropy_loss_scaling": hp.uniform("els", 0.0, 0.1),
        },
    }

    space_spec.update(
        unflattened_params(config_from_args(args), delim=":")
    )

    search_args = space_spec.get("search", {})
    del space_spec["search"]

    config = {
        "objective": ale_objective,
        "algo": "rand",
        "space": [space_spec], # Wrapped because hyperopt is weird.
        "max_evals": 32,
    }

    config.update(search_args)

    return config
