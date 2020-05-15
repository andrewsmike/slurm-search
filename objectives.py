"""
Objectives to search over.
Each objective has some callable evaluation function and a CLI invocation
decoding function that generates the appropriate search session config.

ALL OBJECTIVES MUST BE IMPORTED INTO THE MAIN MODULE FOR PICKLE LOADING.
"""
from csv import reader
from os.path import join
from time import sleep

from hyperopt import hp
import numpy as np

from all.environments import AtariEnvironment, GymEnvironment
from all.experiments import SingleEnvExperiment, ParallelEnvExperiment
from all.presets import atari, classic_control, continuous

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
            episodes=spec["episodes"],
        )
        returns.extend(experiment.test(
            episodes=spec["test_episodes"],
        ))
        del experiment

    returns = np.array(returns)
    print(f"Returns: {returns.mean()} +/- {returns.std()}")

    return {
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

def unflattened_dict(flattened_dict, delim=":"):
    result = {}
    for path, value in flattened_dict.items():
        path_parts = path.split(delim)
        key, rest = path_parts[0], path_parts[1:]
        result.setdefault(key, {})[delim.join(rest)] = value

    return {
        key: (
            unflattened_dict(value)
            if "" not in value else
            value[""]
        )
        for key, value in result.items()
    }


# Example:
# slurm_search.py start ale type=classic agent=a2c env=CartPole-v0
def ale_search_session_args(*args):
    space_spec = {
        "frames": 100 * (1000),
        "episodes": 1000,
        "test_episodes": 100,
        "runs_per_setting": 32,

        "agent_args": {
            "lr": 1 - hp.loguniform("one_minus_lr", -4, -1),
        },
    }

    space_spec.update(
        unflattened_dict(config_from_args(args), delim=":")
    )

    search_args = space_spec.get("search", {})
    del space_spec["search"]

    config = {
        "objective": ale_objective,
        "algo": "tpe",
        "max_evals": 12,
        "space": [space_spec], # Wrapped because hyperopt is weird.
    }

    config.update(search_args)

    return config
