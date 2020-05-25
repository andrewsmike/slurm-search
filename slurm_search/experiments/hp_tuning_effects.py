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

def hp_tuning_effects():
    space_samples = random_sampling(
        "hp",
        random_sampling(
            "run_seed",
            return_mean("env", "agent", "hp", "run_params", "run_seed"),
            sample_count="search:run_samples_per_setting",
            method="inline",
        ),
        sample_count="search:setting_samples",
        method="search:method",
        threads="search:threads",
    )

    best_hp = space_samples["argmax:mean"]
    worst_hp = space_samples["argmin:mean"]
    space_returns_mean_std = space_samples["mean:mean", "std:mean"]

    best_hp_samples = use("hp", best_hp)[
        random_sampling(
            "run_seed",
            return_mean("env", "agent", "hp", "run_params", "run_seed"),
            sample_count="eval:run_samples",
            method="search:method",
            threads="eval:threads",
        )
    ]

    worst_hp_samples = use("hp", worst_hp)[
        random_sampling(
            "run_seed",
            return_mean("env", "agent", "hp", "run_params", "run_seed"),
            sample_count="eval:run_samples",
            method="search:method",
            threads="eval:threads",
        )
    ]

    best_hp_returns_mean_std = best_hp_samples["mean", "std"]
    worst_hp_returns_mean_std = worst_hp_samples["mean", "std"]

    best_hp_cdf = best_hp_samples["cdf"]
    worst_hp_cdf = worst_hp_samples["cdf"]

    space_hp_cdf = space_samples["point_values:cdf"]
    space_hp_mean = space_samples["point_values:mean"]


    return {
        "best_hp_samples": best_hp_samples,
        "best_hp": best_hp,
        "best_hp_returns_mean_std": best_hp_returns_mean_std,
        "best_hp_cdf": best_hp_cdf,

        "space_hp_samples": space_samples,
        "space_returns_mean_std": space_returns_mean_std,
        "space_hp_cdf": space_hp_cdf,
        "space_hp_mean": space_hp_mean,

        "worst_hp_samples": worst_hp_samples,
        "worst_hp": worst_hp,
        "worst_hp_returns_mean_std": worst_hp_returns_mean_std,
        "worst_hp_cdf": worst_hp_cdf,
    }


hp_tuning_effects_config = {
    "agent": "classic:a2c",
    "env": "classic:CartPole-v1",

    "hp_space": {
        "lr": hp.loguniform("lr", log(0.0001), log(0.01)),
        "entropy_loss_scaling": hp.uniform("els", 0.0, 0.1),
    },

    "run_seed_space": hp.quniform("run_seed", 0, 2 ** 31, 1),

    "run": {
        "train_frames": 100000,
        "train_episodes": np.inf,
        "test_episodes": 100,
    },
    "search": {
        "setting_samples": 96,
        "run_samples_per_setting": 12,

        "method": "slurm",
        "threads": 8,
    },
    "eval": {
        "run_samples": 24,

        "threads": 8,
    },
}

hp_tuning_effects_debug_overrides = {
    "search": {"method": "inline"},
    "agent": "debug",
}

def display_hp_tuning_effects(session_name, results):
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

hp_tuning_effects_exp = {
    "config": hp_tuning_effects_config,
    "debug_overrides": hp_tuning_effects_debug_overrides,
    "display_func": display_hp_tuning_effects,
    "experiment_func": hp_tuning_effects,
}
