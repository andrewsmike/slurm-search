from code import interact
from math import log
from pprint import pprint
from sys import argv

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
from slurm_search.params import (
    params_from_args,
    unflattened_params,
    updated_params,
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
        method="slurm",
        threads="search:threads",
    )

    best_hp = space_samples["argmax:mean"]

    best_hp_samples = use("hp", best_hp)[
        random_sampling(
            "run_seed",
            return_mean("env", "agent", "hp", "run_params", "run_seed"),
            sample_count="eval:run_samples",
            method="slurm",
            threads="eval:threads",
        )
    ]

    return best_hp_samples["cdf"]


def demo_hp_effects():
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

default_demo_tuning_curve_config = {
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
    },

    "eval": {
        "run_samples": 24,
        "threads": 8,
    },
}

def demo_hp_improvement():
    overrides = {
        "agent": "classic:a2c",
        "env":  "classic:CartPole-v1",
    }
    overrides = updated_params(
        overrides,
        unflattened_params(params_from_args(argv[1:]))
    )
    print("Override params:")
    pprint(overrides)
    print("All params:")
    pprint(updated_params(default_demo_hp_effects_config, overrides))

    details = experiment_details(
        demo_tuning_curve,
        defaults=default_demo_tuning_curve_config,
        overrides=overrides,
    )
    for key, value in details.items():
        print(f"{key}:")
        print(value)

    return 0
    session_name, results = run_experiment(
        demo_tuning_curve,
        defaults=default_demo_hp_effects_config,
        overrides=overrides,
    )

    print(session_name)
    pprint(results)

    display_setting_surface(
        results["space_hp_mean"],
        setting_dims=["entropy_loss_scaling", "lr"],
        zlabel="Mean return",
        fig_name="setting_mean_return",
    )
    display_setting_cdf_surface(
        results["space_hp_cdf"],
        zlabel="Return",
        fig_name="setting_percentile_return",
    )

    interact(local=locals())
