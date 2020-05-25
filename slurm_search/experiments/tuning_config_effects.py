from math import log

from hyperopt import hp
import numpy as np

from slurm_search.experiment import (
    random_sampling,
    use,
)
from slurm_search.experiments.all_tools import return_mean
from slurm_search.experiments.display_tools import (
    display_setting_surface,
    display_setting_cdf_surface,
)

def search_return_cdf(search_hp, search_seed):
    space_samples = random_sampling(
        "hp",
        random_sampling(
            "run_seed",
            return_mean("env", "agent", "hp", "run_params", "run_seed"),
            sample_count="search_hp:runs_per_setting",
            method="inline",
        ),
        sample_count="search_hp:setting_samples",
        method="search:method",
        threads="search:threads",
    )

    best_hp = space_samples["argmax:mean"]

    best_hp_samples = use("hp", best_hp)[
        random_sampling(
            "run_seed",
            return_mean("env", "agent", "hp", "run_params", "run_seed"),
            sample_count="eval:run_samples",
            method="search:method",
            threads="eval:threads",
        )
    ]

    return best_hp_samples["cdf"]

def tuning_config_effects():
    search_samples = random_sampling(
        ("search", "search_space"),
        search_return_cdf(
            "search_hp",
            "search_seed",
        ),
        sample_count="meta:search_count",
        method="inline",
    )

    search_hp_mean = search_samples["point_values:mean"]
    search_hp_var = search_samples["point_values:var"]

    return {
        "search_samples": search_samples,
        "search_hp_mean": search_hp_mean,
        "search_hp_var": search_hp_var,
    }

tuning_config_effects_config = {
    "agent": "classic:a2c",
    "env": "classic:CartPole-v1",

    "search_space": {
        "lr": hp.loguniform("lr", log(0.0001), log(0.01)),
        "entropy_loss_scaling": hp.uniform("entropy_loss_scaling", 0.0, 0.1),
    },
    "search_seed_space": hp.quniform("search_seed", 0, 2 ** 31, 1),
    "hp_space": {
        "runs_per_sample": hp.uniform("runs_per_sample", 8, 24),
        "setting_samples": hp.uniform("setting_samples", 12, 96),
    },
    "run_seed_space": hp.quniform("run_seed", 0, 2 ** 31, 1),

    "run": {
        "train_frames": 100000,
        "train_episodes": np.inf,
        "test_episodes": 100,
    },

    "search": {
        "threads": 8,
        "method": "slurm",
    },

    "eval": {
        "run_samples": 24,
        "threads": 8,
    },
}

tuning_config_effects_debug_overrides = {
    "search": {"method": "inline"},
    "agent": "debug",
}


def display_tuning_config_effects(session_name, results):
    display_setting_surface(
        results["space_hp_mean"],
        setting_dims=["entropy_loss_scaling", "lr"],
       zlabel="Mean return",
        fig_name=f"{session_name}_setting_mean_return",
    )
    display_setting_cdf_surface(
        results["space_hp_cdf"],
        zlabel="Return",
        fig_name=f"{session_name}_setting_percentile_return",
    )

tuning_config_effects_exp = {
    "config": tuning_config_effects_config,
    "debug_overrides": tuning_config_effects_debug_overrides,
    "display_func": display_tuning_config_effects,
    "experiment_func": tuning_config_effects,
}
