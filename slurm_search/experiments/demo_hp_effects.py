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
from all_tools import return_mean

def demo_hp_effects():
    typical_hp_samples = random_sampling(
        "hp",
        random_sampling(
            "run_seed",
            return_mean("env", "agent", "hp", "run_params", "run_seed"),
            sample_count="search:run_samples_per_setting",
            method="inline",
        )["mean"],
        sample_count="search:setting_samples",
        method="search:method",
        threads="search:threads",
    )

    best_hp = typical_hp_samples["argmax"]
    worst_hp = typical_hp_samples["argmin"]
    typical_hp_returns_mean_std = typical_hp_samples["mean", "std"]

    best_hp_runs = use("hp", best_hp)[
        random_sampling(
            "run_seed",
            return_mean("env", "agent", "hp", "run_params", "run_seed"),
            sample_count="eval:run_samples",
            method="eval:method",
            threads="eval:threads",
        )
    ]

    worst_hp_runs = use("hp", worst_hp)[
        random_sampling(
            "run_seed",
            return_mean("env", "agent", "hp", "run_params", "run_seed"),
            sample_count="eval:run_samples",
            method="eval:method",
            threads="eval:threads",
        )
    ]

    best_hp_returns_mean_std = best_hp_runs["mean", "std"]
    worst_hp_returns_mean_std = worst_hp_runs["mean", "std"]

    return {
        "best_hparams": best_hp,
        "best_returns_mean_std": best_hp_returns_mean_std,

        "typical_returns_mean_std": typical_hp_returns_mean_std,

        "worst_hparams": worst_hp,
        "worst_returns_mean_std": worst_hp_returns_mean_std,
    }


default_hp_space = {
    "lr": hp.loguniform("lr", log(0.0001), log(0.01)),
    "entropy_loss_scaling": hp.uniform("els", 0.0, 0.1),
}

default_demo_hp_effects_config = {
    "agent": "continuous:a2c",
    "env": "continuous:CartPole-v1",

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

        "method": "inline",
        "threads": 12,
    },
    "eval": {
        "run_samples": 24,

        "method": "inline",
        "threads": 12,
    },
}

def main():
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
            "agent": "continuous:a2c",
            "env": "continuous:CartPole-v1",
        },
    )

    pprint(results)
    import pdb
    pdb.set_trace()

if __name__ == "__main__":
    main()
