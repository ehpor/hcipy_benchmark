import functools
import psutil
import os
import platform
import subprocess
import re
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


@functools.cache
def get_processor_name():
    '''Return processor hardware name, in a cross-platform manner

    With thanks to https://stackoverflow.com/a/20161999
    '''
    if platform.system() == 'Windows':
        return platform.processor()
    elif platform.system() == 'Darwin':
        os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
        command = 'sysctl -n machdep.cpu.brand_string'.split()

        return (subprocess.check_output(command).strip()).decode()
    elif platform.system() == 'Linux':
        command = 'cat /proc/cpuinfo'
        all_info = subprocess.check_output(command, shell=True).strip()

        for line in all_info.split('\n'):
            if 'model name' in line:
                return re.sub('.*model name.*:', '', line, 1)

    return ''


@functools.cache
def get_physical_cores():
    return psutil.cpu_count(logical=False)


@functools.cache
def get_operating_system_name():
    res = platform.system()

    if res == 'Darwin':
        return 'MacOS'

    return res


def errorfill(x, y, y_err_pos, y_err_neg=None, color=None, alpha_fill=0.25, ax=None, **kwargs):
    """
    Plot a line with filled errorbars.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    if y_err_neg is None:
        y_err_neg = y_err_pos

    y_min = np.array(y) - np.array(y_err_neg)
    y_max = np.array(y) + np.array(y_err_pos)
    if color is None:
        line = ax.plot(x, y, **kwargs)
        color = line[0].get_color()
    else:
        line = ax.plot(x, y, color=color, **kwargs)

    facecolor = mpl.colors.colorConverter.to_rgba(color, alpha=alpha_fill)
    edgecolor = (0, 0, 0, 0)
    ax.fill_between(x, y_max, y_min, edgecolor=edgecolor, facecolor=facecolor)

    return line


class BenchmarkIterator:
    def __init__(self, num_burnin=1, num_iterations_min=7, t_min=1):
        self.num_burnin = num_burnin
        self.num_iterations_min = num_iterations_min
        self.t_min = t_min

        self.reset()

    def reset(self):
        self.num_iterations = 0

        self.t_iterations = []
        self.t_setup = None

        self.benchmark_start = time.perf_counter()
        self.t_start = time.perf_counter()

    def __iter__(self):
        return self

    def __next__(self):
        t = time.perf_counter()

        if self.t_setup is None:
            self.t_setup = t - self.t_start
        else:
            self.num_iterations += 1

            if self.num_iterations > self.num_burnin:
                self.t_iterations.append(t - self.t_start)

            if t - self.benchmark_start > self.t_min and self.num_iterations >= (self.num_iterations_min + self.num_burnin):
                self.t_iterations = np.array(self.t_iterations)

                raise StopIteration()

        self.t_start = time.perf_counter()

    @property
    def min(self):
        return np.min(self.t_iterations)

    @property
    def max(self):
        return np.max(self.t_iterations)

    @property
    def mean(self):
        return np.mean(self.t_iterations)

    @property
    def median(self):
        return np.mean(self.t_iterations)

    def quantile(self, q):
        return np.quantile(self.t_iterations, q)


class FixedIterator:
    def __init__(self, num_iterations=1):
        self.num_iterations = num_iterations

    def __iter__(self):
        self._iterations = self.num_iterations

        return self

    def __next__(self):
        if self._iterations == 0:
            raise StopIteration()

        self._iterations -= 1


class Benchmark:
    def __init__(self, title, parameter_name, parameter_values, **kwargs):
        self.title = title
        self.parameter_name = parameter_name
        self.parameter_values = parameter_values
        self.kwargs = kwargs

        self.runs = {}

    def run(self, func, label, **plot_kwargs):
        self.runs[label] = []

        for arg in self.parameter_values:
            # Work out parameters.
            params = {self.parameter_name: arg}
            print(params)

            # Perform the benchmark.
            it = BenchmarkIterator(**self.kwargs)
            func(it, **params)

            self.runs[label].append(it)

    def plot(self, relative_to=None):
        if relative_to is not None:
            ref_run = self.runs[relative_to]
            norm = np.array([res.median for res in ref_run])
        else:
            norm = 1

        for func_name, results in self.runs.items():
            x = self.parameter_values
            y = [res.median for res in results]
            y_min = [res.quantile(0.5 - 0.341) for res in results]
            y_max = [res.quantile(0.5 + 0.341) for res in results]

            x = np.array(x)
            y = np.array(y) / norm
            y_min = np.array(y_min) / norm
            y_max = np.array(y_max) / norm

            errorfill(x, y, y_max - y, y - y_min, label=func_name)

        plt.suptitle(self.title)
        plt.title(f'on {get_operating_system_name()}, {get_processor_name()} [{get_physical_cores()} cores]')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.grid(ls=':', which='major', c='0.4')
        plt.grid(ls=':', which='minor', c='0.7')
        plt.xlabel(self.parameter_name)

        if relative_to is None:
            plt.ylabel('Time [s]')
        else:
            plt.ylabel(f'Time relative to {relative_to}')
