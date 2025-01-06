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
        l = ax.plot(x, y, **kwargs)
        color = l[0].get_color()
    else:
        l = ax.plot(x, y, color=color, **kwargs)

    facecolor = mpl.colors.colorConverter.to_rgba(color, alpha=alpha_fill)
    edgecolor = (0, 0, 0, 0)
    ax.fill_between(x, y_max, y_min, edgecolor=edgecolor, facecolor=facecolor)

    return l


class BenchmarkIterator:
    def __init__(self, num_burnin=1, num_iterations_min=7, t_max=1):
        self.num_burnin = num_burnin
        self.num_iterations_min=num_iterations_min
        self.t_max = t_max

        self.num_iterations = 0

        self.t_iterations = []
        self.t_setup = None

        self.start()

    def start(self):
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

            if t - self.benchmark_start > self.t_max and len(self.t_iterations) >= self.num_iterations_min:
                raise StopIteration()

        self.t_start = time.perf_counter()


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
    def __init__(self, title, parameter_name, parameter_values, t_min=1, num_burnin=5):
        self.title = title
        self.parameter_name = parameter_name
        self.parameter_values = parameter_values
        self.t_min = t_min
        self.num_burnin = num_burnin

        self.runs = {}

    def run(self, func, label, **plot_kwargs):
        self.runs[label] = []

        for arg in self.parameter_values:
            # Work out parameters.
            params = {self.parameter_name: arg}
            print(params)

            # Perform the benchmark.
            it = BenchmarkIterator(self.num_burnin, t_max=self.t_min)
            func(it, **params)

            run_result = {
                'time_setup': it.t_setup,
                'time_iterations' : np.array(it.t_iterations),
                'plot_kwargs': plot_kwargs,
            }
            self.runs[label].append(run_result)

    def plot(self):
        for func_name, result in self.runs.items():
            t = [arg_res['time_iterations'] for arg_res in result]

            x = self.parameter_values
            y = [np.median(tt) for tt in t]
            y_min = [np.quantile(tt, 0.5 - 0.341) for tt in t]
            y_max = [np.quantile(tt, 0.5 + 0.341) for tt in t]

            x = np.array(x)
            y = np.array(y)
            y_min = np.array(y_min)
            y_max = np.array(y_max)

            errorfill(x, y, y_max - y, y - y_min, label=func_name)

        plt.suptitle(self.title)
        plt.title(f'on {get_operating_system_name()}, {get_processor_name()} [{get_physical_cores()} cores]')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.grid(':', c='0.5')
        plt.xlabel(self.parameter_name)
        plt.ylabel('Time [s]')
