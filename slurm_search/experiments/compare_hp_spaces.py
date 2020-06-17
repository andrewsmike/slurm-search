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
    for hp_space_name in ["loguniform_hp_space"]:
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


compare_hp_spaces_config = {
    "agent": "classic:a2c",
    "env": "classic:CartPole-v1",

    "loguniform_hp_space": {
        "clip_grad": hp.loguniform("clip_grad", log(0.1), log(1)),
        "lr": hp.loguniform("lr", log(1e-4), log(1e-2)),
        "entropy_loss_scaling": hp.uniform("els", 0, 0.1),
        "value_loss_scaling": hp.uniform("vls", 0.2, 1.2),
        "n_envs": hp.qloguniform("n_envs", log(1), log(32), 1),
        "n_steps": hp.qloguniform("n_steps", log(1), log(16), 1),
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
            results["space_hp_returns_mean"],
            setting_dims=["entropy_loss_scaling", "lr"],
            zlabel="Mean return",
            fig_name=f"{session_name}_{space_name}setting_mean_return",
        )

        try:
            display_setting_cdf_surface(
                results["space_hp_returns_cdf"],
                zlabel="Return {space_name}",
                fig_name=f"{session_name}_{space_name}_setting_percentile_return",
            )
        except:
            pass

        for label, mean, std in (
                ("trial best", results["best_hp_results"]["mean"], results["best_hp_results"]["std"]),
                ("entire space", results["space_returns_mean"], results["space_returns_std"]),
        ):
            print(f"{space_name} performance for {label}: {mean} +/- {std}")

        label_cdfs = {
            "space": results["space_returns_cdf"],
            "trial_best": results["best_hp_results"]["cdf"],
        }

        labels, cdfs = zip(*label_cdfs.items())

        display_cdfs(
            cdfs,
            labels,
            title="Performance CDFs of various HPs [{space_name}].",
            fig_name=f"{session_name}_{space_name}_setting_cdfs",
        )

        hp_cdfs = results["space_hp_returns_cdf"]
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
