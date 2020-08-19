from csv import reader
from functools import partial
from math import log
from os.path import expanduser

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
    benchmark_env_cdf,
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

benchmark_name = "exp:foamy_narrow_path"

@accepts_param_names
def percentile_env_norm(env, return_mean):
    env_cdf = benchmark_env_cdf(benchmark_name, env)
    return np.searchsorted(
        env_cdf,
        return_mean,
    ) / len(env_cdf)

def multienv_benchmark_tuning(benchmark_agents):
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
                    percentile_env_norm(
                        "env",
                        return_mean("env", agent, "hp", "run_params", "run_seed"),
                    ),
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

multienv_benchmark_tuning_config = {
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

multienv_benchmark_tuning_debug_overrides = {}

def display_multienv_benchmark_tuning(session_name, params, results):
    display_setting_surface(
        results["space_hp_returns_mean"],
        setting_dims=["entropy_loss_scaling", "lr"],
        zlabel="Mean return",
        fig_name=f"{session_name}_setting_mean_return",
    )

    try:
        display_setting_cdf_surface(
            results["space_hp_returns_cdf"],
            zlabel="Return",
            fig_name=f"{session_name}_setting_percentile_return",
        )
    except:
        pass

    best_flattened = np.array([
        result
        for env, env_results in results["best_hp_results"]["point_values"]
        for seed, result in env_results["point_values"]
    ])
    best_mean = best_flattened.mean()
    best_std = best_flattened.std()
    best_cdf = np.sort(best_flattened)

    for label, mean, std in (
            ("trial best", best_mean, best_std),
            ("entire space", results["space_returns_mean"], results["space_returns_std"]),
    ):
        print(f"Performance for {label}: {mean} +/- {std}")

    label_cdfs = {
        "space": results["space_returns_cdf"],
        "trial_best": best_cdf,
    }

    labels, cdfs = zip(*label_cdfs.items())

    display_cdfs(
        cdfs,
        labels,
        title="Performance CDFs of various HPs.",
        fig_name=f"{session_name}_setting_cdfs",
    )

    hp_cdfs = results["space_hp_returns_cdf"]
    space_hp_cdfs = np.array([
        cdf
        for hp, cdf in hp_cdfs
    ])

    hp_std = space_hp_cdfs.mean(axis=1).std()
    trial_std = space_hp_cdfs.std(axis=1).mean()
    print(f"[{session_name}] Trial STD / HP std = {trial_std:0.3} / {hp_std:0.3} = {trial_std/hp_std:0.3}")

    setting_names = [
        "clip_grad",
        "lr",
        "entropy_loss_scaling",
        "value_loss_scaling",
        "n_envs",
        "n_steps",
    ]

    setting_point_group = {
        setting_name: [
            (hp[setting_name], run_result)
            for env, env_results in results["space_results"]["point_values"]
            for hp, hp_results in env_results["point_values"]
            for seed, run_result in hp_results["point_values"]
        ]
        for setting_name in setting_names
    }
    for setting_name, point_group in setting_point_group.items():
        display_setting_samples(
            point_groups=[point_group],
            labels=[setting_name],
            title=f"Multienv {setting_name} performance.",
            fig_name=f"{session_name}_{setting_name}_performance",
            show=False,
        )



multienv_benchmark_tuning_exp = {
    "config": multienv_benchmark_tuning_config,
    "debug_overrides": multienv_benchmark_tuning_debug_overrides,
    "display_func": display_multienv_benchmark_tuning,
    "experiment_func": partial(multienv_benchmark_tuning, benchmark_agents),
}

multienv_a2c_benchmark_tuning_exp = {
    "config": multienv_benchmark_tuning_config,
    "debug_overrides": multienv_benchmark_tuning_debug_overrides,
    "display_func": display_multienv_benchmark_tuning,
    "experiment_func": partial(multienv_benchmark_tuning, ["classic:a2c"]),
}

multienv_ppo_benchmark_tuning_exp = {
    "config": multienv_benchmark_tuning_config,
    "debug_overrides": multienv_benchmark_tuning_debug_overrides,
    "display_func": display_multienv_benchmark_tuning,
    "experiment_func": partial(multienv_benchmark_tuning, ["classic:ppo"]),
}

multienv_vsarsa_benchmark_tuning_exp = {
    "config": multienv_benchmark_tuning_config,
    "debug_overrides": multienv_benchmark_tuning_debug_overrides,
    "display_func": display_multienv_benchmark_tuning,
    "experiment_func": partial(multienv_benchmark_tuning, ["classic:vsarsa"]),
}

multienv_dqn_benchmark_tuning_exp = {
    "config": multienv_benchmark_tuning_config,
    "debug_overrides": multienv_benchmark_tuning_debug_overrides,
    "display_func": display_multienv_benchmark_tuning,
    "experiment_func": partial(multienv_benchmark_tuning, ["classic:dqn"]),
}
