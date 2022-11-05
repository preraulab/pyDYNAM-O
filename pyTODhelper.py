import numpy as np


def convertHMS(seconds: float) -> str:
    """Converts seconds to HH:MM:SS string

    :param seconds:
    :return:
    """
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return "%02d:%02d:%02d" % (hour, minutes, seconds)


def arange_inc(start: float, stop: float, step: float) -> np.ndarray:
    """Inclusive numpy arange

    :param start: start value
    :param stop: stop value
    :param step: step value
    :return: range = [start:step:stop]
    """
    stop += (lambda x: step * max(0.1, x) if x < 0.5 else 0)((lambda n: n - int(n))((stop - start) / step + 1))
    return np.arange(start, stop, step)


def create_bins(range_start: float, range_end: float, bin_width: float, bin_step: float, bin_method: str = 'full'):
    """Create bins allowing for various overlap and windowing schemes

    :param bin_range: 1x2 array of start-stop values
    :param bin_width: Bin width
    :param bin_step: Bin step
    :param bin_method: 'full' starts the first bin at bin_range[0]. 'partial' starts the first bin
    with its bin center at bin_range(1) but ignores all values below bin_range[0]. 'full_extend' starts
    the first bin at bin_range[0] - bin_width/2.Note that it is possible to get values outside of bin_range
    with this setting.
    :return: bin_edges, bin_centers
    """

    bin_method = str.lower(bin_method)

    if bin_method == 'full':
        range_start_new = range_start + bin_width / 2
        range_end_new = range_end - bin_width / 2

        bin_centers = np.array(arange_inc(range_start_new, range_end_new, bin_step))
        bin_edges = [bin_centers - bin_width / 2, bin_centers + bin_width / 2]
    elif bin_method == 'partial':
        bin_centers = np.array(arange_inc(range_start, range_end, bin_step))
        bin_edges = np.maximum(np.minimum([bin_centers - bin_width / 2, bin_centers + bin_width / 2],
                                          range_end), range_start)
    elif bin_method == 'extend' or bin_method == 'full extend' or bin_method == 'full_extend':
        range_start_new = range_start - np.floor((bin_width / 2) / bin_step) * bin_step
        range_end_new = range_end + np.floor((bin_width / 2) / bin_step) * bin_step

        bin_centers = np.array(arange_inc(range_start_new + (bin_width / 2), range_end_new - (bin_width / 2), bin_step))
        bin_edges = [bin_centers - bin_width / 2, bin_centers + bin_width / 2]
    else:
        raise ValueError("bin_method should be full, partial, or extend")

    return bin_edges, bin_centers


def outside_colorbar(fig_obj, ax_obj, graphic_obj, gap=0.01, shrink=1, label=""):
    """Creates a colorbar that is outside the axis bounds and does not shrink the axis

    :param fig_obj: Figure object
    :param ax_obj: Axis object
    :param graphic_obj: Graphics object (image, scatterplot, etc.)
    :param gap: Gap between bar and axis
    :param shrink: Colorbar shrink factor
    :param label: Colorbar label
    :return: colorbar object
    """

    ax_pos = ax_obj.get_position().bounds  # Axis position

    # Create new colorbar and get position
    cbar = fig_obj.colorbar(graphic_obj, ax=ax_obj, shrink=shrink, label=label)
    ax_obj.set_position(ax_pos)
    cbar_pos = cbar.ax.get_position().bounds

    # Set new colorbar position
    cbar.ax.set_position([ax_pos[0] + ax_pos[2] + gap, cbar_pos[1], cbar_pos[2], cbar_pos[3]])

    return cbar
