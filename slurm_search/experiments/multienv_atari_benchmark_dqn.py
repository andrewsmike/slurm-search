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
from slurm_search.experiments.env_baselines import (
    env_min_max_scores,
    env_names,
)
from slurm_search.experiments.display_tools import (
    display_cdfs,
    display_setting_surface,
    display_setting_cdf_surface,
    display_setting_samples,
)


def multienv_atari_benchmark_dqn():
    space_samples = maximizing_sampling(
        "hp",
        random_sampling(
            "env",
            random_sampling(
                "run_seed",
                return_mean("env", "agent", "hp", "run_params", "run_seed"),
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

    """best_hp_samples = use("hp", best_hp)[
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
    ]"""

    return {
        "best_hp": best_hp,

        #"best_hp_results": best_hp_samples,

        "space_results": space_samples,

        "space_returns_mean": space_samples["mean:mean:mean"],
        "space_returns_std": space_samples["std:mean:mean"],
        "space_returns_cdf": space_samples["cdf:mean:mean"],
        "space_hp_returns_mean": space_samples["point_values:mean:mean"],
        "space_hp_returns_cdf": space_samples["point_values:cdf:mean"],
    }


import hyperopt.pyll
from hyperopt.pyll import scope

if not hasattr(scope, "bounded"):
    @scope.define
    def bounded(val, minimum=None, maximum=None):
        if minimum is not None:
            val = max(val, minimum)
        if maximum is not None:
            val = min(val, maximum)
        return val

safe_atari_envs = [
    f"atari:{env}"
    for env in sorted(env_names)
    if env not in {
            # Throw exceptions for whatever reasons.
            "QBert", "KungFuMaster", "Freeway", "WizardofWor",
            "UpandDown", "JamesBond", "RiverRaid", "Skiing",
            "Tutankham", "JourneyEscape", "MsPacman", "Asterix",
            # Too slow to complete 2M frames in <4h.
            "VideoPinball", "DoubleDunk", "CrazyClimber", "Hero",
            "Gravitar", "ElevatorAction",
    }
]

test_atari_envs = sorted(safe_atari_envs)[:8]

multienv_atari_benchmark_dqn_config = {
    "agent": "atari:dqn",

    #"env": "classic:CartPole-v1",
    "env_space": hp.choice(
        "env",
        test_atari_envs,
    ),

    "hp_space": {
        "lr": scope.bounded(hp.lognormal("lr", log(1e-3), (log(1e-3) - log(1e-4))/3), minimum=0, maximum=1),
        "eps": scope.bounded(hp.lognormal("eps", log(1e-3), (log(1e-3) - log(1e-4))/3), minimum=0, maximum=1),

        "minibatch_size": scope.bounded(hp.qlognormal("minibatch_size", log(32), log(32)/8, 1), minimum=4, maximum=64),
        "update_frequency": scope.bounded(hp.qlognormal("update_frequency", log(4), log(4)/8, 1), minimum=2, maximum=16),

        #"clip_grad": scope.bounded(hp.normal("clip_grad", 0.4, 0.1), minimum=0.001, maximum=1),
        #"entropy_loss_scaling": scope.bounded(hp.normal("els", 0.06, 0.01), minimum=0),
        #"value_loss_scaling": hp.uniform("vls", 0.2, 1.2), # Unavailable for classic.
        #"n_envs": scope.bounded(hp.qlognormal("n_envs", log(32)/2, log(32)/8, 1), minimum=2, maximum=32),
        #"n_steps": scope.bounded(hp.qlognormal("n_steps", log(16)/2, log(16)/8, 1), minimum=2, maximum=32),

        #"lr": hp.loguniform("lr", log(0.0001), log(0.01)),
        #"entropy_loss_scaling": hp.uniform("els", 0.0, 0.1),
    },

    "run_seed_space": hp.quniform("run_seed", 0, 2 ** 31, 1),

    "run_params": {
        "train_frames": 200000,
        "train_episodes": np.inf,
        "test_episodes": 200,
    },
    "search": {
        "method": "slurm",
        "threads": 16,

        "setting_samples": 16,
        "env_samples_per_setting": 8,
        "run_samples_per_setting_env": 1,
    },
    "eval": {
        "method": "slurm",
        "threads": 16,

        "env_samples": 16,
        "run_samples_per_env": 1,
    },
}

multienv_atari_benchmark_dqn_debug_overrides = {
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

def display_multienv_atari_benchmark_dqn(session_name, params, results):
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

    best_flattened = np.array([
        result
        for env, env_results in results["best_hp_results"]["point_values"]
        for seed, result in env_results["point_values"]
    ])
    best_mean = best_flattened.mean()
    best_std = best_flattened.std()
    best_cdf = np.sort(best_flattened)

    for label, mean, std in (
            ("trial best", best_mean, best_std),
            ("entire space", results["space_returns_mean"], results["space_returns_std"]),
    ):
        print(f"Performance for {label}: {mean} +/- {std}")

    label_cdfs = {
        "space": results["space_returns_cdf"],
        "trial_best": best_cdf,
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

    setting_names = [
        "clip_grad",
        "lr",
        "entropy_loss_scaling",
        "value_loss_scaling",
        "n_envs",
        "n_steps",
    ]

    setting_point_group = {
        setting_name: [
            (hp[setting_name], run_result)
            for env, env_results in results["space_results"]["point_values"]
            for hp, hp_results in env_results["point_values"]
            for seed, run_result in hp_results["point_values"]
        ]
        for setting_name in setting_names
    }
    for setting_name, point_group in setting_point_group.items():
        display_setting_samples(
            point_groups=[point_group],
            labels=[setting_name],
            title=f"Multienv {setting_name} performance.",
            fig_name=f"{session_name}_{setting_name}_performance",
            show=False,
        )



multienv_atari_benchmark_dqn_exp = {
    "config": multienv_atari_benchmark_dqn_config,
    "debug_overrides": multienv_atari_benchmark_dqn_debug_overrides,
    "display_func": display_multienv_atari_benchmark_dqn,
    "experiment_func": multienv_atari_benchmark_dqn,
}
