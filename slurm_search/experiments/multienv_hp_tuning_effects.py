from csv import reader
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
from slurm_search.experiments.env_baselines import env_min_max_scores
from slurm_search.experiments.display_tools import (
    display_cdfs,
    display_setting_surface,
    display_setting_cdf_surface,
)

@accepts_param_names
def linear_env_norm(env, return_mean):
    low, high = env_min_max_scores()[env]

    return (return_mean - low) / (high - low)

def multienv_hp_tuning_effects():
    space_samples = maximizing_sampling(
        "hp",
        random_sampling(
            "env",
            random_sampling(
                "run_seed",
                linear_env_norm(
                    "env",
                    return_mean("env", "agent", "hp", "run_params", "run_seed")
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

    best_hp = space_samples["argmax:mean:mean"]

    best_hp_samples = use("hp", best_hp)[
        random_sampling(
            "env",
            random_sampling(
                "run_seed",
                linear_env_norm(
                    "env",
                    return_mean("env", "agent", "hp", "run_params", "run_seed"),
                ),
                sample_count="eval:run_samples_per_env",
                method="inline",
                threads="eval:threads",
            ),
            sample_count="eval:env_samples",
            method="eval:method",
        )
    ]

    return {
        "best_hp": best_hp,

        "best_hp_results": best_hp_samples,

        "space_returns_mean": space_samples["mean:mean:mean"],
        "space_returns_std": space_samples["std:mean:mean"],
        "space_returns_cdf": space_samples["cdf:mean:mean"],
        "space_hp_returns_mean": space_samples["point_values:mean:mean"],
        "space_hp_returns_cdf": space_samples["point_values:cdf:mean"],
    }


multienv_hp_tuning_effects_config = {
    "agent": "classic:a2c",

    #"env": "classic:CartPole-v1",
    "env_space": hp.choice(
        "env",
        [
            "classic:CartPole-v0",
            "classic:CartPole-v1",
        ],
    ),

    "hp_space": {
        "lr": hp.loguniform("lr", log(0.0001), log(0.01)),
        "entropy_loss_scaling": hp.uniform("els", 0.0, 0.1),
    },

    "run_seed_space": hp.quniform("run_seed", 0, 2 ** 31, 1),

    "run_params": {
        "train_frames": 2000000,
        "train_episodes": np.inf,
        "test_episodes": 200,
    },
    "search": {
        "method": "slurm",
        "threads": 18,

        "setting_samples": 12,
        "env_samples_per_setting": 6,
        "run_samples_per_setting_env": 1,
    },
    "eval": {
        "method": "slurm",
        "threads": 18,

        "env_samples": 18,
        "run_samples_per_env": 1,
    },
}

multienv_hp_tuning_effects_debug_overrides = {
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

def display_multienv_hp_tuning_effects(session_name, params, results):
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

    for label, mean, std in (
            ("trial best", results["best_hp_results"]["mean"], results["best_hp_results"]["std"]),
            ("entire space", results["space_returns_mean"], results["space_returns_std"]),
    ):
        print(f"Performance for {label}: {mean} +/- {std}")

    label_cdfs = {
        "space": results["space_returns_cdf"],
        "trial_best": results["best_hp_results"]["cdf"],
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



multienv_hp_tuning_effects_exp = {
    "config": multienv_hp_tuning_effects_config,
    "debug_overrides": multienv_hp_tuning_effects_debug_overrides,
    "display_func": display_multienv_hp_tuning_effects,
    "experiment_func": multienv_hp_tuning_effects,
}
