"""
Interruptible ssearch-based experimental protocols.
"""
from contextlib import contextmanager
from datetime import datetime
from copy import deepcopy
from functools import wraps
from inspect import signature
from subprocess import Popen
from time import sleep

import numpy as np

from slurm_search.params import (
    unflattened_params,
    updated_params,
)
from slurm_search.slurm_search import (
    launch_slurm_search_workers,
)
from slurm_search.search_session import (
    create_search_session,
    next_search_trial,
    search_session_names,
    search_session_progress,
    search_session_results,
    update_search_results,
)
from slurm_search.session_state import (
    create_session_state,
    session_state,
    session_state_names,
    update_session_state,
)
from slurm_search.random_phrase import random_phrase
from slurm_search.locking import lock


## Experiment persistence.
def create_experiment(func, base_params, override_params):
    existing_session_names = session_state_names()

    session_name = "exp:" + random_phrase()
    while session_name in existing_session_names:
        session_name = "exp:" + random_phrase()

    with lock(session_name):
        create_session_state(
            session_name,
            {
                "func": func,
                "base_params": base_params,
                "override_params": override_params,
                "params": updated_params(
                    base_params,
                    override_params,
                ),
                "start_time": datetime.now(),
            },
        )

    return session_name

def update_experiment_results(session_name, results):
    with lock(session_name):
        experiment_state = session_state(session_name)
        experiment_state["results"] = results
        update_session_state(session_name, experiment_state)


## Experiment runtime management.
_current_experiment = None
_current_params = None

def param(name):
    params = _current_params
    while ":" in name:
        parts = name.split(":")
        head, name = parts[0], ":".join(parts[1:])
        params = params.get(parts[0], {})

    return deepcopy(params.get(name, None))

def params():
    return deepcopy(_current_params)

@contextmanager
def temporary_params(next_params):
    global _current_params

    prev_params = _current_params
    try:
        _current_params = next_params
        yield None
    finally:
        _current_params = prev_params


def accepts_param_names(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return CallNode(
            wrapper,
            args,
            kwargs,
        )

    return wrapper

# Experiment invocations.
def nodes_evaluated(nodes_dict):
    return {
        key: value() if isinstance(value, Node) else value
        for key, value in nodes_dict.items()
    }

def run_experiment(func, defaults, overrides):
    params = updated_params(defaults, overrides)
    session_name = create_experiment(func, defaults, overrides)

    global _current_experiment
    _current_experiment = session_name
    global _current_params
    _current_params = params

    results = nodes_evaluated(func())

    update_experiment_results(session_name, results)

    return session_name, results

def nodes_details(nodes_dict):
    return {
        key: str(value) if isinstance(value, Node) else value
        for key, value in nodes_dict.items()
    }

def experiment_details(func, defaults, overrides):
    params = updated_params(defaults, overrides)
    # session_name = create_experiment(func, defaults, overrides)

    global _current_experiment
    _current_experiment = None
    global _current_params
    _current_params = params

    details = nodes_details(func())

    return details

## I accidentally an AST

class Node(object):
    pass

class CallNode(Node):
    def __init__(self, wrapper, args, kwargs):
        """
        Due to pickling issues, we have to proxy through a reference to the
        wrapping function. It is expected that the underlying function be
        available under the "__wrapped__" attribute.
        """
        self.wrapper = wrapper
        self.args = args
        self.kwargs = kwargs

        self.params = dict(signature(wrapper.__wrapped__).bind(
            *args,
            **kwargs,
        ).arguments)

    def __call__(self):
        # Bind.
        params = dict(self.params)
        params.update({
            param_name: param(param_value)
            for param_name, param_value in params.items()
            if isinstance(param_value, str)
            if param(param_value) is not None
        })
        params.update({
            param_name: param(param_value[:-len("_params")])
            for param_name, param_value in params.items()
            if isinstance(param_value, str)
            if param(param_value[:-len("_params")]) is not None
        })

        return self.wrapper.__wrapped__(**params)

    def __str__(self):
        args_str = ", ".join(
            list(self.args) + [
                f"{key}={val}"
                for key, val in self.kwargs.items()
            ]
        )
        return f"{self.wrapper.__wrapped__.__name__}({args_str})"


class UseNode(Node):
    def __init__(self, param, value, expr=None):
        self.param = param
        self.value = value
        self.expr = expr

    def __getitem__(self, arg):
        if self.expr is None:
            self.expr = arg
            return self
        else:
            return UseNode(
                self.param,
                self.value,
                self.expr[arg],
            )

    def __call__(self):
        param, value = self.param, self.value

        if isinstance(value, Node):
            value = value()

        new_params = updated_params(
            params(),
            unflattened_params({ # Param may be a path.
                param: value,
            }),
        )
        with temporary_params(new_params):
            return self.expr()

    def __str__(self):
        top_padding = " " * (4 + len(self.param) + 3 + 4)
        padded_value = str(self.value).replace("\n", "\n" + top_padding)
        return (
            f"use({self.param} = {padded_value})[\n    " + 
            str(self.expr).replace("\n", "\n    ") +
            "\n]"
        )

def use(param, value):
    return UseNode(param, value)

class SamplingResultsNode(Node):
    def __init__(self, expr, attr_names):
        self.expr = expr
        self.attr_names = attr_names

    def __call__(self):
        return self.expr(*self.attr_names)

    def __str__(self):
        name = "_".join(self.attr_names)
        return f"{name}({self.expr})"

def random_sampling_objective(spec):
    spec, = spec
    func = spec["func"]
    params = spec["params"]
    with temporary_params(params):
        results = func()

    # So you can return interesting data.
    if isinstance(results, (int, float)):
        loss = results
    else:
        loss = 0

    return {
        "result": results,
        "loss": loss,
        "status": "ok",
    }


class RandomSamplingNode(Node):
    """
    Random sampling of a parameter.

    Sampling can happen in three different ways:
    - inline: Run the sampling on this process.
    - cpu: Run the sampling on this system on parallel threads.
    - slurm: Run the sampling on parallel slurm workers.

    Attributes:
    - mean: Result mean.
    - std: Result standard deviation.
    - cdf: Sorted array of results.

    - min: Min of results.
    - max: Max of results.
    - argmin: ArgMin of results.
    - argmax: ArgMax of results.
    """
    def __init__(
            self,
            sampling_var_space,
            func,
            sample_count=None,
            method=None,
            threads=None,
    ):

        if isinstance(sampling_var_space, tuple):
            sampling_var, sampling_space = sampling_var_space
        else:
            sampling_var = sampling_var_space
            sampling_space = sampling_var_space + "_space"

        self.sampling_var = sampling_var
        self.sampling_space = sampling_space
        self.hidden_params = set()

        self.func = func

        self.sample_count = sample_count

        assert method in ("inline", "cpu", "slurm")
        self.method = method
        self.thread_count = threads

        self.launched = False
        self.collected = False

    def launch(self):
        if not self.launched:
            self.bind_params()
            self.register_session()

            if self.method == "cpu":
                self.launch_cpu()
            elif self.method == "slurm":
                self.launch_slurm()

            self.launched = True

    def __getitem__(self, attr_names):
        if not isinstance(attr_names, tuple):
            attr_names = [attr_names]
        return SamplingResultsNode(self, list(attr_names))

    def bind_params(self):
        if isinstance(self.sampling_space, str):
            self.hidden_params.add(self.sampling_space)
            self.sampling_space = param(self.sampling_space)

        if isinstance(self.sample_count, str):
            self.sample_count = param(self.sample_count)

        self.params = params()

    def register_session(self):
        session_type = "slurm:" if self.method == "slurm" else "sampling:"

        existing_session_names = search_session_names(
            filter_inactive=False,
            search_type=session_type,
        )

        session_name =  session_type + random_phrase()
        while session_name in existing_session_names:
            session_name = session_type + random_phrase()

        params = updated_params(
            self.params,
            # Unflatten as sampling var may be a path.
            unflattened_params({
                self.sampling_var: self.sampling_space,
            }),
        )
        for param in self.hidden_params:
            del params[param]

        search_space_spec = {
            "func": self.func,
            "params": params,
        }

        create_search_session(
            session_name,
            objective=random_sampling_objective,
            algo="rand",
            space=[
                search_space_spec
            ], # Wrapped because hyperopt is weird.
            max_evals=self.sample_count,

            # Metadata.
            params=self.params,
            sampling_var=self.sampling_var,
            sampling_space=self.sampling_space,
        )

        self.session_name = session_name

    def run_inline(self):
        try:
            while True:
                self.sample_once(worker_id="cpu0")
        except ValueError as e:
            pass

    def sample_once(self, worker_id):
        trial_id, next_sample = (
            next_search_trial(self.session_name, worker_id)
        )

        results = random_sampling_objective(next_sample)

        update_search_results(
            self.session_name,
            trial_id,
            worker_id,
            next_sample,
            results,
        )

    def launch_cpu(self):
        worker_command = ["ssearch", "work_on", self.session_name]

        self.workers = [
            Popen(worker_command)
            for i in range(self.thread_count)
        ]
        # TODO: Do I want to wait on these in the wait_parallel method?

    def launch_slurm(self):
        launch_slurm_search_workers(
            self.session_name,
            iteration=1,
            thread_count=self.thread_count,
        )

    def wait(self):
        if self.method == "inline":
            self.run_inline()
        elif self.method in ("cpu", "slurm"):
            self.wait_parallel()

    def wait_parallel(self):
        MINUTE = 60
        while True:
            status = search_session_progress(self.session_name)["status"]
            assert status in ("active", "disabled", "complete"), (
                f"Unknown search status {status} for search {self.session_name}."
            )

            if status == "disabled":
                raise RuntimeError(
                    f"Someone stopped search session {self.session_name}."
                )
            elif status == "complete":
                return

            print(f"Search {self.session_name} is active. Waiting 1 minute.")
            sleep(1 * MINUTE)

    def __call__(self, *attrs):
        self.launch()
        results = self.results()

        if not attrs:
            return results
        elif len(attrs) == 1:
            attr, = attrs
            return results[attr]
        else:
            return [
                results[attr]
                for attr in attrs
            ]

    def results(self):
        self.collect_results()

        results = [
            results["result"]
            for setting, results in self.setting_results
        ]
        if isinstance(results[0], (int, float)):
            result_dist = np.array(results)
        else:
            result_dist = np.zeros(len(results))

        sorted_setting_results = sorted([
            setting, results["result"]
            for setting, results in self.setting_results
        ],
            key=lambda setting_results: setting_results[1],
        )

        return {
            "mean": result_dist.mean(),
            "std": result_dist.std(),
            "min": result_dist.min(),
            "max": result_dist.max(),
            "cdf": np.sort(result_dist),
            "argmin": sorted_setting_results[0][0],
            "argmax": sorted_setting_results[-1][0],
            "point_values": sorted_setting_results,
        }

    def collect_results(self):
        if not self.collected:
            self.wait()

            status = search_session_progress(
                self.session_name
            )["status"]
            assert status == "complete", status

            self.setting_results = (
                search_session_results(self.session_name)["setting_results"]
            )

            self.collected = True

    def __str__(self):
        func_str = str(self.func).replace('\n', '\n    ')
        return (
            f"RandomSampling[{self.sampling_var} ~ {self.sampling_space}](\n" +
            f"    {func_str},\n" +
            f"    sample_count={self.sample_count},\n" +
            f"    method={self.method},\n" +
            f"    threads={self.thread_count},\n" +
            "]"
        )

def random_sampling(*args, **kwargs):
    return RandomSamplingNode(*args, **kwargs)
