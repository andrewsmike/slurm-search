from math import log
from random import choices

from hyperopt import hp
import numpy as np

from slurm_search.experiment import (
    enumeration_sampling,
    random_sampling,
    use,
)
from slurm_search.experiments.all_tools import return_mean
from slurm_search.experiments.display_tools import (
    display_setting_surface,
    display_setting_cdf_surface,
)

def hp_setting_run_samples():
    space_samples = random_sampling(
        "hp",
        random_sampling(
            "run_seed",
            return_mean("env", "agent", "hp", "run_params", "run_seed"),
            sample_count="search:runs_per_setting",
            method="inline",
        ),
        sample_count="search:setting_samples",
        method="search:method",
        threads="search:threads",
    )

    return space_samples["point_values"]

tuning_config_effects_config = {
    "agent": "classic:a2c",
    "env": "classic:CartPole-v1",

    "hp_space": {
        "lr": hp.loguniform("lr", log(0.0001), log(0.01)),
        "entropy_loss_scaling": hp.uniform("entropy_loss_scaling", 0.0, 0.1),
    },
    "run_seed_space": hp.quniform("run_seed", 0, 2 ** 31, 1),

    "run_params": {
        "train_frames": 75000,
        "train_episodes": np.inf,
        "test_episodes": 100,
    },

    "search": {
        "setting_samples": 112,
        "runs_per_setting": 32,

        "threads": 16,
        "method": "slurm",
    },
}

tuning_config_effects_debug_overrides = {
    "search": {"method": "inline"},
    "agent": "debug",
}



def alist_get(alist, lookup_key):
    result, = [
        value
        for key, value in alist
        if key == lookup_key
    ]

    return result

def bootstrap_search_best_hp(
        setting_run_samples,
        setting_samples,
        runs_per_setting,
):
    settings = [
        setting
        for setting, run_samples in setting_run_samples
    ]
    selected_settings = choices(settings, k=setting_samples)

    selected_setting_samples = [
        (
            selected_setting,
            choices(
                alist_get(setting_run_samples, selected_setting),
                k=runs_per_setting,
            ),
        )
        for selected_setting in selected_settings
    ]

    return max(
        selected_setting_samples,
        key=lambda setting_sample: sum(setting_sample[1]),
    )[0]

def display_tuning_config_effects(session_name, params, results):
    """
    Perform bootstrapping to analyzepppp setting_samples, runs_per_setting's effects
    on search performance.
    """

    # How we evaluate the results.
    setting_means = [
        (setting, result["mean"])
        for setting, result in results
    ]

    # How we generate the results.
    setting_run_samples = [
        (setting, result["cdf"])
        for setting, result in results
    ]

    setting_samples_values = list(range(8, 136+1, 4))
    runs_per_sample_values = list(range(4, 36+1, 2))

    bootstrap_trials_per_setting = 256

    S_vals = len(setting_samples_values)
    R_vals = len(runs_per_sample_values)
    T_vals = bootstrap_trials_per_setting
    S_val_axis, R_val_axis, T_axis = 0, 1, 2
    setting_searches_cdf = np.array([
        [
            [
                alist_get(
                    setting_means,
                    bootstrap_search_best_hp(
                        setting_run_samples,
                        setting_samples=setting_samples,
                        runs_per_setting=runs_per_sample,
                    ),
                )
                for trial_index in range(bootstrap_trials_per_setting)
            ]
            for runs_per_sample in runs_per_sample_values
        ]
        for setting_samples in setting_samples_values
    ])

    setting_search_means = setting_searches_cdf.mean(axis=T_axis)

    setting_search_points = [
        (
            {
                "Setting samples": setting_samples_values[S_val_index],
                "Runs per sample": runs_per_sample_values[R_val_index],
            },
            setting_search_means[S_val_index][R_val_index]
        )
        for S_val_index in range(S_vals)
        for R_val_index in range(R_vals)
    ]

    display_setting_surface(
        setting_search_points,
        setting_dims=["Setting samples", "Runs per sample"],
        zlabel="Mean return",
        fig_name=f"tuning_config_mean_{session_name}",
        product_contours=True,
    )

    display_setting_cdf_surface(
        setting_searches_cdf.reshape((S_vals * R_vals), T_vals),
        zlabel="Mean return",
        fig_name=f"tuning_config_cdf_{session_name}",
    )


tuning_config_effects_exp = {
    "config": tuning_config_effects_config,
    "debug_overrides": tuning_config_effects_debug_overrides,
    "display_func": display_tuning_config_effects,
    "experiment_func": hp_setting_run_samples,
}
