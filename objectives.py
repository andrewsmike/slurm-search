"""
Objectives to search over.
Each objective has some callable evaluation function and a CLI invocation
decoding function that generates the appropriate search session config.
"""
from csv import reader
from os.path import join
from time import sleep

from hyperopt import hp
import numpy as np

from all.environments import AtariEnvironment, GymEnvironment
from all.experiments import SingleEnvExperiment
from all.presets import atari, classic_control, continuous

__all__ = [
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

# Invoking an ALE runner.
def last_100_returns_mean(writer):
    records_path = join(writer.log_dir, writer.env_name, "returns100.csv")
    with open(records_path, "r") as f:
        lines = list(reader(f))

    last_frame, last_100_returns_mean, last_100_returns_std = lines[-1]
    return last_100_returns_mean

def ale_objective(spec):
    spec, = spec
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

    experiment = SingleEnvExperiment(
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
    returns = experiment.test(
        episodes=spec["test_episodes"],
    )

    return value

def config_from_args(args):
    config = {
        key: value
        for arg in args
        for key, val in (arg.strip().split("="), )
    }

def state_from_config(config):
    return {
        "agent_args": {
            "lr": hp.uniform(0.95, 0.9999),
        }
    }
    # "a2c"
    # "CartPole-v0"
    # "gym"
    # "lr"

def ale_search_session_args(*args):
    # agent, env, env_type to start.
    default_config = {
        "objective": ale_objective,
        "algo": "tpe",
        "max_evals": 16,

        "frames": 1 * (1000 * 1000),
        "episodes": 1,
        "test_episodes": 100,
    }

    config = default_config.update(
        config_from_args(args)
    )

    config["state"] = state_from_config(config)

    return [config]
