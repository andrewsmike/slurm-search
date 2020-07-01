import numpy as np
from matplotlib import cm, ticker
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as tri

def display_cdfs(
        cdfs,
        labels=None,
        xlabel=None,
        ylabel=None,
        title=None,
        fig_name=None,
        show=False,
):

    for i, cdf in enumerate(cdfs):
        cdf_percentiles = np.linspace(0, 1, len(cdf))
        label = None if not labels else labels[i]
        plt.plot(cdf_percentiles, cdf, label=label)

    if labels:
        plt.legend()

    if title:
        plt.title(title)

    xlabel = xlabel or "Percentile"
    plt.xlabel(xlabel)

    if ylabel:
        plt.ylabel(ylabel)

    if show:
        plt.show()

    if fig_name:
        plt.savefig(f"{fig_name}.png")
        plt.clf()


def display_surface(
        Z,
        X=None,
        Y=None,
        xlabel=None,
        ylabel=None,
        zlabel=None,
        title=None,
        view_angle=160,
        show_points=False,
        product_contours=False,
        contour_levels=None,

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

    if show:
        plt.show()

    if fig_name:
        fig_surface.savefig(f"{fig_name}_surface.png")
        plt.clf()

    fig_contour = plt.figure()
    ax_contour = fig_contour.gca()

    contour_levels = contour_levels if contour_levels is not None else 8

    ax_contour.contour(
        X, Y, Z_w_nans,
        levels=contour_levels,
        colors="k",
    )
    cntr1 = ax_contour.contourf(
        X, Y, Z_w_nans,
        levels=contour_levels,
        cmap="RdBu_r",
    )
    fig_contour.colorbar(cntr1, ax=ax_contour)

    if product_contours:
        ax_contour.contour(
            X, Y, X * Y,
            levels=10,
            color="b"
        )

    if show_points:
        ax_contour.plot(orig_X, orig_Y, "ko", ms=3)

    if xlabel:
        ax_contour.set_xlabel(xlabel)
    if ylabel:
        ax_contour.set_ylabel(ylabel)

    if title:
        ax_contour.set_title(title)

    if show:
        plt.show()

    if fig_name:
        fig_contour.savefig(f"{fig_name}_contour.png")
        plt.clf()


def display_setting_surface(
        setting_result,
        setting_dims=None,
        zlabel=None,
        title=None,
        fig_name=None,
        show=False,
        product_contours=False,
        contour_levels=None,
        show_points=True,
):
    X_label, Y_label = setting_dims

    X, Y, Z = np.array([
        [
            settings[X_label],
            settings[Y_label],
            result,
        ]
        for settings, result in setting_result
    ]).T

    display_surface(
        X=X,
        Y=Y,
        Z=Z,
        xlabel=X_label,
        ylabel=Y_label,
        zlabel=zlabel or "Result",
        show_points=show_points,
        view_angle=-140,
        product_contours=product_contours,
        contour_levels=contour_levels,

        fig_name=fig_name,
        title=title,
        show=show,
    )

def display_setting_cdf_surface(
        setting_cdfs,
        zlabel=None,
        title=None,
        fig_name=None,
        show=False,
):
    if isinstance(setting_cdfs[0], tuple):
        setting_cdfs = [
            cdf
            for setting, cdf in setting_cdfs
        ]
    setting_axis, result_axis = 0, 1
    setting_cdfs = np.sort(np.array(setting_cdfs), axis=result_axis)
    setting_cdfs = setting_cdfs[
        np.argsort(setting_cdfs.mean(axis=result_axis), axis=setting_axis)
    ]

    display_surface(
        Z=setting_cdfs, # (Y, X) -> (settings, runs)
        xlabel="Percentile",
        ylabel="Settings",
        zlabel=zlabel or "Result",
        view_angle=-120,

        title=title,
        fig_name=fig_name,
        show=show,
    )

def display_setting_samples(
        point_groups,
        labels=None,
        xlabel=None,
        xscale="linear",
        ylabel=None,
        title=None,
        fig_name=None,
        show=False,
):
    labels = labels or [f"Group {i+1}" for i in range(len(point_groups))]
    for point_group, label in zip(point_groups, labels):
        setting_values, setting_returns = zip(*(point_group))
        plt.scatter(setting_values, setting_returns, marker=".", label=label)

        # Also mark best as a larger circle.
        best_setting, best_setting_returns = max(point_group, key=lambda point: point[1])
        plt.scatter(best_setting, best_setting_returns, marker="^", label=label)

    if labels:
        plt.legend()

    if title:
        plt.title(title)

    plt.xscale(xscale)

    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)

    if show:
        plt.show()

    if fig_name:
        plt.savefig(f"{fig_name}.png")
        plt.clf()
