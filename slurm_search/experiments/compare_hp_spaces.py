from math import log

from hyperopt import hp
import numpy as np

from slurm_search.experiment import (
    maximizing_sampling,
    random_sampling,
    use,
)
from slurm_search.experiments.all_tools import return_mean
from slurm_search.experiments.display_tools import (
    display_cdfs,
    display_setting_surface,
    display_setting_cdf_surface,
)

def compare_hp_spaces():
    space_results = {}
    for hp_space_name in [
            "loguniform_hp_space",
            "lognormal_hp_space",
            "lognormal_smaller_hp_space",
            "lognormal_smallerer_hp_space",
    ]:
        space_samples = maximizing_sampling(
            ("hp", hp_space_name),
             random_sampling(
                 "run_seed",
                 return_mean("env", "agent", "hp", "run_params", "run_seed"),
                 sample_count="search:run_samples_per_setting",
                 method="inline",
             ),
             sample_count="search:setting_samples",
             maximize_measure="mean",

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

        space_results[hp_space_name] = {
            "best_hp": best_hp,

            "best_hp_results": best_hp_samples,

            "space_returns_mean": space_samples["mean:mean"],
            "space_returns_std": space_samples["std:mean"],
            "space_returns_cdf": space_samples["cdf:mean"],
            "space_hp_returns_mean": space_samples["point_values:mean"],
            "space_hp_returns_cdf": space_samples["point_values:cdf"],
        }

    return space_results

import hyperopt.pyll
from hyperopt.pyll import scope

@scope.define
def bounded(val, minimum=None, maximum=None):
    if minimum is not None:
        val = max(val, minimum)
    if maximum is not None:
        val = min(val, maximum)

    return val

compare_hp_spaces_config = {
    "agent": "classic:a2c",
    "env": "classic:CartPole-v1",

    "loguniform_hp_space": {
        "clip_grad": scope.bounded(hp.loguniform("clip_grad", log(0.1), log(1)), minimum=0.001, maximum=1),
        "lr": scope.bounded(hp.loguniform("lr", log(1e-4), log(1e-2)), minimum=0, maximum=1),
        "entropy_loss_scaling": scope.bounded(hp.uniform("els", 0, 0.1), minimum=0),
        #"value_loss_scaling": hp.uniform("vls", 0.2, 1.2), # Unavailable for classic.
        "n_envs": scope.bounded(hp.qloguniform("n_envs", log(1), log(32), 1), minimum=1, maximum=32),
        "n_steps": scope.bounded(hp.qloguniform("n_steps", log(1), log(16), 1), minimum=1, maximum=32),
    },

    # Set 3std ~= min or max
    "lognormal_hp_space": {
        "clip_grad": scope.bounded(hp.normal("clip_grad", 0.4, 0.1), minimum=0.001, maximum=1),
        "lr": scope.bounded(hp.lognormal("lr", log(1e-3), (log(1e-3) - log(1e-4))/3), minimum=0, maximum=1),
        "entropy_loss_scaling": scope.bounded(hp.normal("els", 0.06, 0.01), minimum=0),
        #"value_loss_scaling": hp.uniform("vls", 0.2, 1.2), # Unavailable for classic.
        "n_envs": scope.bounded(hp.qlognormal("n_envs", log(32)/2, log(32)/8, 1), minimum=1, maximum=32),
        "n_steps": scope.bounded(hp.qlognormal("n_steps", log(16)/2, log(16)/8, 1), minimum=1, maximum=32),
    },

    "lognormal_smaller_hp_space": {
        "clip_grad": scope.bounded(hp.normal("clip_grad", 0.15, 0.04), minimum=0.001, maximum=1),
        "lr": scope.bounded(hp.lognormal("lr", log(7e-4), (log(7e-4) - log(1e-4))/6), minimum=0, maximum=1),
        "entropy_loss_scaling": scope.bounded(hp.normal("els", 0.03, 0.008), minimum=0),
        #"value_loss_scaling": hp.uniform("vls", 0.2, 1.2), # Unavailable for classic.
        "n_envs": scope.bounded(hp.qlognormal("n_envs", log(32)/2, log(32)/8, 1), minimum=1, maximum=32),
        "n_steps": scope.bounded(hp.qlognormal("n_steps", log(5), log(5)/6, 1), minimum=1, maximum=32),
    },

    "lognormal_smallerer_hp_space": {
        "clip_grad": scope.bounded(hp.normal("clip_grad", 0.12, 0.02), minimum=0.001, maximum=1),
        "lr": scope.bounded(hp.lognormal("lr", log(7e-4), (log(7e-4) - log(1e-4))/12), minimum=0, maximum=1),
        "entropy_loss_scaling": scope.bounded(hp.normal("els", 0.03, 0.004), minimum=0),
        #"value_loss_scaling": hp.uniform("vls", 0.2, 1.2), # Unavailable for classic.
        "n_envs": scope.bounded(hp.qlognormal("n_envs", log(32)/2, log(32)/16, 1), minimum=1, maximum=32),
        "n_steps": scope.bounded(hp.qlognormal("n_steps", log(5), log(5)/8, 1), minimum=1, maximum=32),
    },

    "run_seed_space": hp.quniform("run_seed", 0, 2 ** 31, 1),

    "run_params": {
        "train_frames": 100000,
        "train_episodes": np.inf,
        "test_episodes": 100,
    },
    "search": {
        "setting_samples": 96,
        "run_samples_per_setting": 12,

        "method": "slurm",
        "threads": 16,
    },
    "eval": {
        "run_samples": 24,

        "threads": 8,
    },
}

compare_hp_spaces_debug_overrides = {
    "search": {"method": "inline"},
    "agent": "debug",
}

def display_compare_hp_spaces(session_name, params, results):
    for space_name, space_results in results.items():
        display_setting_surface(
            space_results["space_hp_returns_mean"],
            setting_dims=["entropy_loss_scaling", "lr"],
            zlabel="Mean return",
            fig_name=f"{session_name}_{space_name}setting_mean_return",
        )

        try:
            display_setting_cdf_surface(
                space_results["space_hp_returns_cdf"],
                zlabel="Return {space_name}",
                fig_name=f"{session_name}_{space_name}_setting_percentile_return",
            )
        except:
            pass

        for label, mean, std in (
                ("trial best", space_results["best_hp_results"]["mean"], space_results["best_hp_results"]["std"]),
                ("entire space", space_results["space_returns_mean"], space_results["space_returns_std"]),
        ):
            print(f"{space_name} performance for {label}: {mean} +/- {std}")

        label_cdfs = {
            "space": space_results["space_returns_cdf"],
            "trial_best": space_results["best_hp_results"]["cdf"],
        }

        labels, cdfs = zip(*label_cdfs.items())

        display_cdfs(
            cdfs,
            labels,
            title=f"Performance CDFs of various HPs [{space_name}].",
            fig_name=f"{session_name}_{space_name}_setting_cdfs",
        )

        hp_cdfs = space_results["space_hp_returns_cdf"]
        space_hp_cdfs = np.array([
            cdf
            for hp, cdf in hp_cdfs
        ])

        hp_std = space_hp_cdfs.mean(axis=1).std()
        trial_std = space_hp_cdfs.std(axis=1).mean()
        print(f"[{session_name}] {space_name} Trial STD / HP std = {trial_std:0.3} / {hp_std:0.3} = {trial_std/hp_std:0.3}")



compare_hp_spaces_exp = {
    "config": compare_hp_spaces_config,
    "debug_overrides": compare_hp_spaces_debug_overrides,
    "display_func": display_compare_hp_spaces,
    "experiment_func": compare_hp_spaces,
}
