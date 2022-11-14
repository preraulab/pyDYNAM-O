from itertools import groupby
import numpy as np
from matplotlib.patches import Rectangle


def nan_zscore(data):
    """Computes zscore ignoring nan values

    :param data: Input data
    :return: zscored data
    """
    # Compute modified z-score
    mid = np.nanmean(data)
    std = np.nanstd(data)

    return (data - mid) / std


def pow2db(y):
    """
    Converts power to dB
    :param y: values to convert
    :return: val_dB value in dB
    """

    if isinstance(y, int) or isinstance(y, float):
        if y == 0:
            return np.nan
        else:
            ydB = (10 * np.log10(y) + 300) - 300
    else:
        if isinstance(y, list):  # if list, turn into array
            y = np.asarray(y)
        y = y.astype(float)  # make sure it's a float array so we can put nans in it
        y[y == 0] = np.nan
        ydB = (10 * np.log10(y) + 300) - 300

    return ydB


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
        bin_edges = np.vstack([bin_centers - bin_width / 2, bin_centers + bin_width / 2])
    elif bin_method == 'partial':
        bin_centers = np.array(arange_inc(range_start, range_end, bin_step))
        bin_edges = np.maximum(np.minimum([bin_centers - bin_width / 2, bin_centers + bin_width / 2],
                                          range_end), range_start)
    elif bin_method == 'extend' or bin_method == 'full extend' or bin_method == 'full_extend':
        range_start_new = range_start - np.floor((bin_width / 2) / bin_step) * bin_step
        range_end_new = range_end + np.floor((bin_width / 2) / bin_step) * bin_step

        bin_centers = np.array(arange_inc(range_start_new + (bin_width / 2), range_end_new - (bin_width / 2), bin_step))
        bin_edges = np.vstack([bin_centers - bin_width / 2, bin_centers + bin_width / 2])
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


def consecutive(val):
    vals = [v[0] for v in groupby(val)]
    cons = [sum(1 for i in g) for v, g in groupby(val)]

    start_inds = np.cumsum(np.insert(cons, 0, 0))
    end_inds = np.add(start_inds[0:-1], cons)

    return list(zip(vals, start_inds, end_inds))


def find_flat(data, minsize=100):
    inds = np.full((len(data)), False)
    for c in consecutive(data):
        if c[2] - c[1] >= minsize:
            inds[c[1]:c[2]] = True

    return inds


def hypnoplot(time, stage, ax=None, plot_buffer=0.8):
    """Plots the hypnogram

    :param time: Stage times
    :param stage: Stage values 6:art, 5:W, 4:R, 3:N1, 2:N2, 1:N3, 0:Unknown
    :param ax: axis for plotting
    :param plot_buffer: how much space above/below
    :return:
    """
    if ax is None:
        ax = plt.axes()

    ax.step(time, stage, 'k-', where='post')
    ax.set_yticks([0, 1, 2, 3, 4, 5, 6], ['Undef', 'N3', 'N2', 'N1', 'R', 'W', 'Art'])
    ylim = (np.min(stage) - plot_buffer, np.max(stage) + plot_buffer)
    ax.set_ylim(ylim)

    ptime = np.append(time, time[-1])

    for c in consecutive(stage):
        if c[0] == 0:
            color = (.9, .9, .9)
        elif 1 <= c[0] <= 3:
            color = (.8, .8, 1)
        elif c[0] == 4:
            color = (.7, 1, .7)
        elif c[0] == 5:
            color = (1, .7, .7)
        else:
            color = (.6, .6, .6)

        ax.add_patch(Rectangle((ptime[c[1]], ylim[0]), ptime[c[2]] - ptime[c[1]], ylim[1] - ylim[0], facecolor=color))
