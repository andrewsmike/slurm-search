from math import log

from hyperopt import hp
import numpy as np

from slurm_search.experiment import (
    random_sampling,
    enumeration_sampling,
    use,
)
from slurm_search.experiments.all_tools import return_mean
from slurm_search.experiments.display_tools import (
    display_cdfs,
)

def compare_agents():
    agent_perf_samples = enumeration_sampling(
        "agent",
        random_sampling(
            "hp",
            random_sampling(
                "run_seed",
                return_mean("env", "agent", "hp", "run_params", "run_seed"),
                sample_count="search:run_samples_per_hp",
                method="inline",
            ),
            sample_count="search:hp_samples_per_agent",
            method="inline",
        ),
        method="search:method",
        threads="search:threads",
    )

    # agent<#agents>:hp<many>:runs<1>
    return {
        "highest_scoring_agent": agent_perf_samples["argmax:mean:mean"],
        "most_consistent_agent": agent_perf_samples["argmin:std:mean"],
        "agent_cdfs": agent_perf_samples["point_values:cdf:mean"],
        "agent_means": agent_perf_samples["point_values:mean:mean"],
    }


compare_agents_config = {
    "agent_space": [
        "classic:a2c",
        "classic:ppo",
    ],
    "env": "classic:CartPole-v1",

    "hp_space": {
        "lr": hp.loguniform("lr", log(0.0001), log(0.01)),
        "entropy_loss_scaling": hp.uniform("els", 0.0, 0.1),
    },

    "run_seed_space": hp.quniform("run_seed", 0, 2 ** 31, 1),

    "run_params": {
        "train_frames": 100000,
        "train_episodes": np.inf,
        "test_episodes": 100,
    },

    "search": {
        "hp_samples_per_agent": 30,
        "run_samples_per_hp": 1,

        "method": "slurm",
        "threads": 8,
    },
}

compare_agents_debug_overrides = {
    "search": {"method": "inline"},
    "agent_space": ["debug1", "debug2"],
}

def display_compare_agents(session_name, params, results):
    agent_labels, cdfs = zip(*results["agent_cdfs"])

    env = params["env"].split(":")[1]
    display_cdfs(
        cdfs,
        labels=agent_labels,
        title=f"Agent returns CDF comparison [{env}]",
        ylabel="Agent Return",
        fig_name=f"{session_name}_setting_percentile_return",
    )

compare_agents_exp = {
    "config": compare_agents_config,
    "debug_overrides": compare_agents_debug_overrides,
    "display_func": display_compare_agents,
    "experiment_func": compare_agents,
}
