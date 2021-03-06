from typing import Sequence

import matplotlib.pyplot as plt
from config import OUT_PATH

PLOT_PATH = OUT_PATH / "plots"


def save_lineplot_with_best_epoch_marked(
    data: Sequence[float], best_epoch: int, timestamp: str
) -> None:
    """
    Saves win ratio over time during agent training.

    Args:
        data: data points in chronological order.
        best_epoch: index of best point to be additionally marked on plot.
        timestamp: text that differentiates plots from different runs.
    """
    PLOT_PATH.mkdir(parents=True, exist_ok=True)

    plt.plot(data)
    plt.axvline(x=best_epoch, color="g")

    filename = f"win_ratio_over_time_{timestamp}.png"
    plt.savefig(PLOT_PATH / filename)
