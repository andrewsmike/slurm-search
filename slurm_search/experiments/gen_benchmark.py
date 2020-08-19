from csv import reader
from math import log
from os.path import expanduser
from pickle import dump

from hyperopt import hp
import numpy as np

from slurm_search.experiment import (
    accepts_param_names,
    maximizing_sampling,
    random_sampling,
    use,
)
from slurm_search.experiments.all_tools import return_mean
from slurm_search.experiments.env_baselines import (
    env_min_max_scores,
    env_names,
)
from slurm_search.experiments.display_tools import (
    display_cdfs,
    display_setting_surface,
    display_setting_cdf_surface,
    display_setting_samples,
)


benchmark_agents = [
    "classic:a2c",
    "classic:ppo",
    "classic:vsarsa",
    "classic:dqn",
]

benchmark_envs = [
    "classic:CartPole-v1",
    "classic:MountainCar-v0",
    "classic:Acrobot-v1",
    # "classic:Pendulum-v0",
]

def gen_benchmark():
    agent_search_samples = {}
    agent_best_hp = {}
    agent_best_hp_samples = {}
    for agent in benchmark_agents:
        agent_short_name = agent.split(":")[1]

        agent_search_samples[agent] = maximizing_sampling(
            ("hp", f"{agent_short_name}_hp_space"),
            random_sampling(
                "run_seed",
                random_sampling(
                    "env",
                    return_mean("env", agent, "hp", "run_params", "run_seed"),

                    sample_count="search:run_samples_per_setting_env",
                    method="inline",
                ),
                sample_count="search:env_samples_per_setting",
                method="inline",
            ),
            sample_count="search:setting_samples",
            maximize_measure="mean:mean",

            method="search:method",
            threads="search:threads",
        )

        agent_best_hp[agent] = agent_search_samples[agent]["argmax:mean:mean"]

        agent_best_hp_samples[agent] = use("hp", agent_best_hp[agent])[
            random_sampling(
                "run_seed",
                random_sampling(
                    "env",
                    return_mean("env", agent, "hp", "run_params", "run_seed"),

                    method="inline",
                    sample_count="eval:run_samples_per_env",
                ),
                sample_count="eval:env_samples",
                method="eval:method",
                threads="eval:threads",
            )
        ]

    return {
        "agent_search_samples": agent_search_samples,
        "best_hp": agent_best_hp,
        "best_hp_samples": agent_best_hp_samples,
    }
        
import hyperopt.pyll
from hyperopt.pyll import scope

if not hasattr(scope, "bounded"):
    @scope.define
    def bounded(val, minimum=None, maximum=None):
        if minimum is not None:
            val = max(val, minimum)
        if maximum is not None:
            val = min(val, maximum)
        return val

gen_benchmark_config = {
    "env_space": hp.choice(
        "env",
        benchmark_envs,
    ),

    "a2c_hp_space": {
        "clip_grad": scope.bounded(hp.normal("clip_grad", 0.4, 0.1), minimum=0.001, maximum=1),
        "lr": scope.bounded(hp.lognormal("lr", log(1e-3), (log(1e-3) - log(1e-4))/3), minimum=0, maximum=1),
        "entropy_loss_scaling": scope.bounded(hp.normal("els", 0.06, 0.01), minimum=0),
        "n_envs": scope.bounded(hp.qlognormal("n_envs", log(32)/2, log(32)/8, 1), minimum=2, maximum=32),
        "n_steps": scope.bounded(hp.qlognormal("n_steps", log(16)/2, log(16)/8, 1), minimum=2, maximum=32),
    },

    "ppo_hp_space": {
        "clip_grad": scope.bounded(hp.normal("clip_grad", 0.4, 0.1), minimum=0.001, maximum=1),
        "lr": scope.bounded(hp.lognormal("lr", log(1e-3), (log(1e-3) - log(1e-4))/3), minimum=0, maximum=1),
        "entropy_loss_scaling": scope.bounded(hp.normal("els", 0.06, 0.01), minimum=0),
        "n_envs": scope.bounded(hp.qlognormal("n_envs", log(32)/2, log(32)/8, 1), minimum=2, maximum=32),
        "n_steps": scope.bounded(hp.qlognormal("n_steps", log(16)/2, log(16)/8, 1), minimum=2, maximum=32),
    },

    "vsarsa_hp_space": {
        "lr": scope.bounded(hp.lognormal("lr", log(1e-3), (log(1e-3) - log(1e-4))/3), minimum=0, maximum=1),
        "eps": scope.bounded(hp.lognormal("eps", log(1e-3), (log(1e-3) - log(1e-4))/3), minimum=0, maximum=1),
        "epsilon": scope.bounded(hp.lognormal("epsilon", log(0.1), (log(0.1) - log(0.02)) / 3), minimum=0.0001, maximum=0.25),
    },

    "dqn_hp_space": {
        "lr": scope.bounded(hp.lognormal("lr", log(1e-3), (log(1e-3) - log(1e-4))/3), minimum=0, maximum=1),
        "minibatch_size": scope.bounded(hp.qlognormal("minibatch_size", log(48), (log(48) - log(8)) / 3, 1), minimum=4, maximum=128),
    },

    "run_seed_space": hp.quniform("run_seed", 0, 2 ** 31, 1),

    "run_params": {
        "train_frames": 300000,
        "train_episodes": np.inf,
        "test_episodes": 200,
    },
    "search": {
        "method": "slurm",
        "threads": 10,

        "setting_samples": 16,
        "env_samples_per_setting": 8,
        "run_samples_per_setting_env": 1,
    },
    "eval": {
        "method": "slurm",
        "threads": 10,

        "env_samples": 64,
        "run_samples_per_env": 1,
    },
}

gen_benchmark_debug_overrides = {
    "search": {
        "method": "inline",
        "setting_samples": 3,
        "env_samples_per_setting": 2,
    },
    "eval": {
        "method": "inline",
        "env_samples": 6,
    },
    "agent": "debug",
}

def write_benchmark(session_name, benchmark_data):
    with open(expanduser(f"~/benchmark_{session_name}.pkl"), "wb") as f:
        dump(benchmark_data, f)

def display_gen_benchmark(session_name, params, results):
    agent_hp_env_returns = [
        (agent, hp, env, env_return)
        for agent, agent_results in results["agent_search_samples"].items()
        for hp, agent_hp_results in agent_results["point_values"]
        for seed, agent_seed_results in agent_hp_results["point_values"]
        for env, env_return in agent_seed_results["point_values"]
    ]

    env_returns = dict()
    for agent, hp, env, env_return in agent_hp_env_returns:
        env_returns.setdefault(env, []).append(env_return)

    env_cdfs = {
        env: np.sort(env_returns)
        for env, env_returns in env_returns.items()
    }

    write_benchmark(session_name, {
        "env_cdfs": env_cdfs,
    })

    for env, env_cdf in env_cdfs.items():
        print(f"[{env}] {env_cdf.min()} <= {env_cdf.mean()} <= {env_cdf.max()}")


gen_benchmark_exp = {
    "config": gen_benchmark_config,
    "debug_overrides": gen_benchmark_debug_overrides,
    "display_func": display_gen_benchmark,
    "experiment_func": gen_benchmark,
}
