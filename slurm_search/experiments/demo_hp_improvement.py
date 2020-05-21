from code import interact
from math import log
from pprint import pprint

from hyperopt import hp
import numpy as np

from slurm_search.experiment import (
    experiment_details,
    random_sampling,
    run_experiment,
    use,
)
from slurm_search.experiments.all_tools import return_mean
from slurm_search.experiments.display_tools import (
    display_setting_surface,
    display_setting_cdf_surface,
)

def demo_hp_effects():
    space_samples = random_sampling(
        "hp",
        random_sampling(
            "run_seed",
            return_mean("env", "agent", "hp", "run_params", "run_seed"),
            sample_count="search:run_samples_per_setting",
            method="inline",
        ),
        sample_count="search:setting_samples",
        method="slurm",
        threads=12,
    )

    best_hp = space_samples["argmax:mean"]
    worst_hp = space_samples["argmin:mean"]
    space_returns_mean_std = space_samples["mean:mean", "std:mean"]

    best_hp_samples = use("hp", best_hp)[
        random_sampling(
            "run_seed",
            return_mean("env", "agent", "hp", "run_params", "run_seed"),
            sample_count="eval:run_samples",
            method="slurm",
            threads=12,
        )
    ]

    worst_hp_samples = use("hp", worst_hp)[
        random_sampling(
            "run_seed",
            return_mean("env", "agent", "hp", "run_params", "run_seed"),
            sample_count="eval:run_samples",
            method="slurm",
            threads=12,
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


default_hp_space = {
    "lr": hp.loguniform("lr", log(0.0001), log(0.01)),
    "entropy_loss_scaling": hp.uniform("els", 0.0, 0.1),
}

default_demo_hp_effects_config = {
    "agent": "classic:a2c",
    "env": "classic:CartPole-v1",

    "hp_space": default_hp_space,

    "run_seed_space": hp.quniform("run_seed", 0, 2 ** 31, 1),

    "run": {
        "train_frames": 100000,
        "train_episodes": np.inf,
        "test_episodes": 100,
    },
    "search": {
        "setting_samples": 96,
        "run_samples_per_setting": 12,
    },
    "eval": {
        "run_samples": 24,
    },
}

def demo_hp_improvement():
    details = experiment_details(
        demo_hp_effects,
        defaults=default_demo_hp_effects_config,
        overrides={
            "agent": "continuous:a2c",
            "env":  "continuous:CartPole-v1",
        },
    )
    for key, value in details.items():
        print(f"{key}:")
        print(value)

    session_name, results = run_experiment(
        demo_hp_effects,
        defaults=default_demo_hp_effects_config,
        overrides={
            "agent": "classic:a2c",
            "env": "classic:CartPole-v1",
        },
    )

    print(session_name)
    pprint(results)

    display_setting_surface(
        results["space_hp_mean"],
        setting_dims=["els", "lr"],
        zlabel="Mean return",
        fig_name="setting_mean_return",
    )
    display_setting_cdf_surface(
        results["space_hp_cdf"],
        zlabel="Return",
        fig_name="setting_percentile_return",
    )

    interact(local=locals())
