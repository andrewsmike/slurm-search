from math import log

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
        "train_frames": 100000,
        "train_episodes": np.inf,
        "test_episodes": 100,
    },

    "search": {
        "setting_samples": 128,
        "runs_per_setting": 32,

        "threads": 16,
        "method": "slurm",
    },
}

tuning_config_effects_debug_overrides = {
    "search": {"method": "inline"},
    "agent": "debug",
}


def display_tuning_config_effects(session_name, params, results):
    """
    Perform bootstrapping to analyze setting_samples, runs_per_setting's effects
    on search performance.
    """
    import pdb
    pdb.set_trace()

    # How we generate the results.
    setting_run_samples = [
        (setting, result["cdf"])
        for setting, result in results
    ]


    # How we evaluate the results.
    setting_means = [
        (setting, result["mean"])
        for setting, result in results
    ]
    setting_stds = [
        (setting, result["stds"])
        for setting, result in results
    ]



tuning_config_effects_exp = {
    "config": tuning_config_effects_config,
    "debug_overrides": tuning_config_effects_debug_overrides,
    "display_func": display_tuning_config_effects,
    "experiment_func": hp_setting_run_samples,
}
