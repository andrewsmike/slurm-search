from pprint import pprint
from time import sleep

import numpy as np
from hyperopt import space_eval

from matplotlib import cm, ticker
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as tri

from slurm_search import (
    search_session_progress,
    search_session_results,
    start_slurm_search,
)
    

MINUTE = 60

def wait_for_completion(session_name):
    while True:
        status = search_session_progress(session_name)["status"]
        assert status in ("active", "disabled", "complete"), (
            f"Unknown search status {status} for search {session_name}."
        )

        if status == "disabled":
            raise RuntimeError(f"Someone stopped search session {session_name}.")
        elif status == "complete":
            return

        print(f"Search {session_name} is active. Waiting 1 minute.")
        sleep(1 * MINUTE)

def display_setting_results_summary(setting_results):
    setting_results_returns = (
        lambda setting_results: setting_results[1]["returns_mean"]
    )
    best_settings, best_results = (
        max(setting_results, key=setting_results_returns)
    )
    best_returns = best_results["loss"]
    worst_settings, worst_results = (
        min(setting_results, key=setting_results_returns)
    )
    worst_returns = worst_results["loss"]

    return_means = np.array([
        results["returns_mean"]
        for setting, results in setting_results
    ])
    return_stds = np.array([
        results["returns_std"]
        for setting, results in setting_results
    ])

    print(f"Best settings / returns: {best_settings} => {best_returns}")
    print(f"Worst settings / returns: {worst_settings} => {worst_returns}")

    print(f"Overall return means: {min(return_means):.4} <= {return_means.mean():.4}" +
          f" +/- {return_means.std():.4} <= {max(return_means):.4}")

    print(f"Overall return stds: {min(return_stds):.4} <= {return_stds.mean():.4}" +
          f" +/- {return_stds.std():.4} <= {max(return_stds):.4}")

    return best_settings, worst_settings

def flattened_dict(unflattened_dict, writeback_dict=None, prefix=None, delim="/"):
    prefix = prefix or []
    writeback_dict = writeback_dict if writeback_dict is not None else {}
    for key, value in unflattened_dict.items():
        path = prefix + [str(key)]
        if isinstance(value, dict):
            flattened_dict(value, writeback_dict=writeback_dict, prefix=path, delim=delim)
        else:
            writeback_dict[delim.join(path)] = value

    return writeback_dict

def setting_space_args(space, settings, excluded_keys=None):
    space_dict = flattened_dict(space_eval(space, settings)[0], delim=":")

    return [
        f"{key}={value}"
        for key, value in space_dict.items()
        if not excluded_keys or key not in excluded_keys
    ]

def display_cdf_summaries(searches, title=None):
    for search, session_name in searches:
        results = search_session_results(session_name)

        setting_results = results["setting_results"]

        print(f"{search} results:")
        display_setting_results_summary(setting_results)
        print()

    labels = [
        search
        for search, session_name in searches
    ]

    search_run_return_means = {
        search: np.array([
            mean
            for _, results in (
                    search_session_results(session_name)["setting_results"]
            )
            for mean in results["run_return_means"]
        ])
        for search, session_name in searches
    }

    for search, run_return_means in search_run_return_means.items():
        runs = len(run_return_means)
        search_cdf = np.sort(run_return_means)
        search_cdf_percentiles = np.arange(runs) / (runs - 1)
        plt.plot(search_cdf_percentiles, search_cdf, label=search)

    plt.legend()

    if title:
        plt.title(title)

    fig_name = ",".join(search[1] for search in searches)
    plt.savefig(f"{fig_name}.png")


def display_surface(
        Z,
        X=None,
        Y=None,
        xlabel=None,
        ylabel=None,
        zlabel=None,
        title=None,
        show_points=False,
        view_angle=160,
        fig_name=None,
        show=False,
):
    if X is not None:
        assert Y is not None
        assert X.shape == Y.shape == Z.shape
        assert len(Z.shape) == 1

        xy_mesh = tri.Triangulation(X, Y)
        interpolator = tri.LinearTriInterpolator(xy_mesh, Z)

        X_count = 450
        Y_count = 500
        grid_X_values = np.linspace(min(X), max(X), X_count)
        grid_Y_values = np.linspace(min(Y), max(Y), Y_count)

        orig_X, orig_Y = X, Y
        X, Y = np.meshgrid(grid_X_values, grid_Y_values)

        Z = interpolator(X, Y)

        Z_w_zeros = Z.filled(0)
        Z_w_nans = Z.filled(np.nan)
    else:
        assert X is None and Y is None
        assert len(Z.shape) == 2

        # session:Y, run:X
        Y_count, X_count = Z.shape
        X = np.linspace(0, 1, X_count)
        Y = np.linspace(0, 1, Y_count)

        X, Y = np.meshgrid(X, Y)

        orig_X, orig_Y = X, Y

        Z_w_zeros = Z_w_nans = Z

    assert len(Z.shape) == 2
    assert X.shape == Y.shape == Z.shape == (Y_count, X_count), (
        f"{X.shape} == {Y.shape} == {Z.shape} == {(Y_count, X_count)}"
    )
    
    fig_surface = plt.figure()
    ax_surface = fig_surface.gca(projection="3d")

    ax_surface.plot_surface(
        X=X, Y=Y, Z=Z_w_zeros,
        cmap=cm.coolwarm,
    )

    if xlabel:
        ax_surface.set_xlabel(xlabel)
    if ylabel:
        ax_surface.set_ylabel(ylabel)
    if zlabel:
        ax_surface.set_zlabel(zlabel)

    ax_surface.view_init(elev=30, azim=view_angle)

    if title:
        ax_surface.set_title(title)

    if fig_name:
        fig_surface.savefig(f"{fig_name}_surface.png")

    if show:
        plt.show()

    fig_contour = plt.figure()
    ax_contour = fig_contour.gca()

    ax_contour.contour(
        X, Y, Z_w_nans,
        levels=8,
        colors="k",
    )
    cntr1 = ax_contour.contourf(
        X, Y, Z_w_nans,
        levels=8,
        cmap="RdBu_r",
    )
    fig_contour.colorbar(cntr1, ax=ax_contour)

    if show_points:
        ax_contour.plot(orig_X, orig_Y, "ko", ms=3)

    if xlabel:
        ax_contour.set_xlabel(xlabel)
    if ylabel:
        ax_contour.set_ylabel(ylabel)

    if title:
        ax_contour.set_title(title)

    if fig_name:
        fig_contour.savefig(f"{fig_name}_contour.png")

    if show:
        plt.show()


def display_search_cdf_surface(session_name):
    setting_results = search_session_results(session_name)["setting_results"]

    setting_axis, run_axis = 0, 1
    setting_returns = np.array([
        results["run_return_means"]
        for _, results in sorted(
                setting_results,
                key=lambda setting_result: sum(setting_result[1]["run_return_means"]),
        )
    ])

    setting_returns = np.sort(setting_returns, axis=run_axis)

    display_surface(
        Z=setting_returns, # (Y, X) -> (settings, runs)
        xlabel="Percentile",
        ylabel="Settings",
        zlabel="Mean returns",
        view_angle=-120,
        fig_name=session_name,
        title="PPO performance surface on CartPole-v1:",
    )

    X_label, Y_label = sorted(setting_results[0][0].keys())

    X, Y, Z = np.array([
        [
            settings[X_label],
            settings[Y_label],
            results["run_return_means"].mean(),
        ]
        for settings, results in setting_results
    ]).T

    display_surface(
        X=X,
        Y=Y,
        Z=Z,
        xlabel=X_label,
        ylabel=Y_label,
        zlabel="Mean returns",
        show_points=True,
        view_angle=-140,
        fig_name=f"{session_name}_settings",
        title="PPO performance surface on CartPole-v1 HP settings:",
    )


def find_eval_setting_extremes(hp_search=None, best_eval=None, worst_eval=None):

    args = [
        "type=classic",
        "agent=ppo",
        "env=CartPole-v1",
        "search:algo=rand",

        "frames=50000",
        "episodes=500",
    ]
    base_args = [
        "runs_per_setting=12",
        "search:max_evals=64",
    ]
    arg_keys = {
        "type",
        "agent",
        "env",
        "frames",
        "episodes",
        "runs_per_setting",
        "search:max_evals",
    }
    print("====[Searching for best settings]====")
    print("Command: $ ssearch start ale " + " ".join(args + base_args))

    hp_search = hp_search or (
        start_slurm_search("ale", *args, *base_args)
    )
    wait_for_completion(hp_search)

    print("====[Search results]====")
    hp_search_results = search_session_results(hp_search)

    hp_search_space = hp_search_results["search_args"]["space"]
    hp_search_setting_results = hp_search_results["setting_results"]

    best_settings, worst_settings = (
        display_setting_results_summary(hp_search_setting_results)
    )
    best_settings_args = setting_space_args(
        hp_search_space,
        best_settings,
        excluded_keys=arg_keys,
    )
    worst_settings_args = setting_space_args(
        hp_search_space,
        worst_settings,
        excluded_keys=arg_keys,
    )

    print("====[Individual Re-evaluations]====")
    eval_args = [
        "runs_per_setting=20",
        "search:max_evals=4",
    ]

    print("==Testing best setting:")
    print("Command: $ ssearch start ale " + " ".join(
        best_settings_args + args + eval_args,
    ))
    # best_eval = "free_purple_thanks"
    best_eval = best_eval or (
        start_slurm_search("ale", *best_settings_args, *args, *eval_args)
    )
    wait_for_completion(best_eval)

    print("==Testing worst setting:")
    print("Command: $ ssearch start ale " + " ".join(
        worst_settings_args + args + eval_args,
    ))
    worst_eval = worst_eval or (
        start_slurm_search("ale", *worst_settings_args, *args, *eval_args)
    )
    wait_for_completion(worst_eval)

    return {
        "search_session": hp_search,
        "best_eval_session": best_eval,
        "worst_eval_session": worst_eval,
    }

def main():
    """
    """
    # PPO run:
    hp_search = "spicy_bad_poetry"
    best_eval = "wrong_brawny_bonus"
    worst_eval = "nifty_used_phone"

    # A2C run:
    hp_search = "decent_curved_office"
    best_eval = "curved_brown_army"
    worst_eval = "abrupt_fresh_power"

   
    searches = find_eval_setting_extremes(
        hp_search=hp_search,
        best_eval=best_eval,
        worst_eval=worst_eval,
    )
    
    print("====[Experiment summary]====")
    display_cdf_summaries(
        [
            ("Random settings", searches["search_session"]),
            ("Best settings", searches["best_eval_session"]),
            ("Worst settings", searches["worst_eval_session"]),
        ],
        title="Agent reward CDF for full space, best, and worst settings:",
    )

    display_search_cdf_surface(searches["search_session"])

    display_cdf_summaries(
        [
            ("[PPO] Random settings", "spicy_bad_poetry"),
            ("[PPO] Best settings", "wrong_brawny_bonus"),
            #("[PPO] Worst settings", "nifty_used_phone"),
            ("[A2C] Random settings", "decent_curved_office"),
            ("[A2C] Best settings", "curved_brown_army"),
            #("[A2C] Worst settings", "abrupt_fresh_power"),
        ],
        title="Agent reward CDF for full space, best, and worst settings:",
    )

    

if __name__ == "__main__":
    main()
