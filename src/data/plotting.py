from numpy.random import default_rng


def random_rgb() -> int:
    """
    Generate random RGB color code.

    Returns
    -------
    rgb: int
        random 9 digits rgb code. Example: 471369080
    """
    rnd_gen = default_rng()
    rnd_rgb = rnd_gen.random() * 1e9
    rgb = int(rnd_rgb)
    return rgb


def initial_stat_plot(plot_widget) -> None:
    plot_widget.canvas.axes.cla()
    plot_widget.canvas.draw()
