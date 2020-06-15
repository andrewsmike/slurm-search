"""
Experimental protocol DSL with interruptable distributed execution and
associated utilities. Supports experiment interrupt / resume, parallelization,
pretty printing, and more. Provides tools for clearly describing and analyzing
different experiments.

This level of granularity focuses on sampling details. Equation pretty printing
is partially available, and can be easily extended.

Uses an AST to specify the experiment. Various nodes may create search or
sampling sessions (using slurm search tools) and launch / collect results during
execution. Resume capabilities provided by tracking results by AST path.

Execution and resume relies on the AST to deduplicate effort. This requires an
amount of determinism for resume to work correctly; since nodes can be reached
at multiple locations of an AST, they _must_ start with the same first path.
"""
from contextlib import contextmanager
from datetime import datetime
from copy import deepcopy
from functools import wraps
from inspect import signature
from pprint import pprint
from subprocess import Popen
from time import sleep, time
from os import getpid

import numpy as np
from hyperopt import fmin, hp, space_eval, tpe, trials_from_docs

from slurm_search.params import (
    flattened_params,
    mapped_params,
    unflattened_params,
    unwrapped_settings,
    updated_params,
    params_str,
    params_equal,
)
from slurm_search.slurm_search import (
    launch_slurm_search_workers,
    slurm_iteration_timeout,
    slurm_worker_id,
)
from slurm_search.search_session import (
    create_search_session,
    next_search_trial,
    reset_active_search_trials,
    search_session_names,
    search_session_progress,
    search_session_results,
    unused_session_name,
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

    session_name = unused_session_name("exp")

    print(f"Creating experiment {session_name}...")

    with lock(session_name):
        create_session_state(
            session_name,
            {
                "status": "active",
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

def experiment_params(session_name):
    with lock(session_name):
        return session_state(session_name)["params"]

def experiment_override_params(session_name):
    with lock(session_name):
        return session_state(session_name)["override_params"]

def update_experiment_results(session_name, results):
    with lock(session_name):
        experiment_state = session_state(session_name)
        experiment_state["results"] = results
        experiment_state["status"] = "complete"
        update_session_state(session_name, experiment_state)

def update_experiment_partial_result(session_name, ast_path, partial_result):
    print(f"[{'.'.join(ast_path)}] {partial_result}")
    with lock(session_name):
        experiment_state = session_state(session_name)

        node = experiment_state.setdefault("partial_results", {})
        for part in ast_path:
            node = node.setdefault(part, {})
        node["partial_result"] = partial_result

        update_session_state(session_name, experiment_state)

def experiment_partial_result(session_name, ast_path):
    with lock(session_name):
        experiment_state = session_state(session_name)

        node = experiment_state.get("partial_results", {})
        for part in ast_path:
            node = node.get(part, {})

        return node.get("partial_result", None)


## Experiment runtime management.
_current_experiment = None
_current_params = None

def current_experiment():
    return _current_experiment

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

@contextmanager
def temporary_experiment(next_experiment):
    global _current_experiment

    prev_experiment = _current_experiment
    try:
        _current_experiment = next_experiment
        yield None
    finally:
        _current_experiment = prev_experiment


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
    return mapped_params(
        lambda value, path: (
            value(ast_path=path)
            if isinstance(value, Node) else value
        ),
        nodes_dict,
    )

def run_experiment(name, func, defaults, overrides, resume=None):

    params = updated_params(defaults, overrides)

    if resume:
        session_name = resume
        if ":" not in session_name:
            session_name = "exp:" + session_name

        resume_params = experiment_params(session_name)

        if not params_equal(resume_params, params):
            print(
                "Attempted to reload an experiment with the wrong parameters."
            )
            print("Correct parameters:")
            pprint(resume_params)

            print("Command line:")
            args_str = " ".join([
                f"--{key}={value}"
                for key, value in flattened_params(
                        experiment_override_params(session_name),
                        delim=":"
                ).items()
            ])
            print(f"run_exp {name} {args_str} --resume={session_name}")
            raise ValueError("Bad arguments")
    else:
        session_name = create_experiment(func, defaults, overrides)

    with temporary_experiment(session_name):
        with temporary_params(params):
            results = nodes_evaluated(func())

    update_experiment_results(session_name, results)

    return session_name, results

def nodes_details(nodes_dict, abstract=False):
    str_func = (
        (lambda value, path: value.abstract_expr_str())
        if abstract else
        (lambda value, path: str(value))
    )
    return mapped_params(str_func, nodes_dict)

def experiment_details(func, defaults, overrides, abstract=False):
    params = updated_params(defaults, overrides)

    global _current_experiment
    _current_experiment = None
    global _current_params
    _current_params = params

    details = nodes_details(func(), abstract=abstract)

    return details

## I accidentally an AST

class Node(object):
    def __call__(self, ast_path):
        raise NotImplementedError
    def __str__(self):
        raise NotImplementedError
    def abstract_expr_str(self, measure_parts=None):
        raise NotImplementedError
    def interruptable(self):
        return False

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

    def __call__(self, ast_path=None):
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

    def abstract_expr_str(self, measure_parts=None):
        assert not measure_parts
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

    def __call__(self, ast_path=None):
        param, value = self.param, self.value

        if isinstance(value, Node):
            value = value(
                ast_path=ast_path + ["use_arg"],
            )

        new_params = updated_params(
            params(),
            unflattened_params({ # Param may be a path.
                param: value,
            }),
        )
        with temporary_params(new_params):
            return self.expr(
                ast_path=ast_path + ["use_val"],
            )

    def __str__(self):
        top_padding = " " * (4 + len(self.param) + 3 + 4)
        padded_value = str(self.value).replace("\n", "\n" + top_padding)
        return (
            f"use({self.param} = {padded_value})[\n    " +
            str(self.expr).replace("\n", "\n    ") +
            "\n]"
        )

    def abstract_expr_str(self, measure_parts=None):
        top_padding = " " * (4 + len(self.param) + 3 + 4)
        padded_value = (
            self.value.abstract_expr_str().replace("\n", "\n" + top_padding)
        )
        return (
            f"use({self.param} = {padded_value})[\n    " +
            self.expr.abstract_expr_str(measure_parts).replace("\n", "\n    ") +
            "\n]"
        )

def use(param, value):
    return UseNode(param, value)

def setting_result_tree_measurement(space, setting_results, trials, measure_spec):
    """
    Setting result tree: List[setting, Either[results_dict, str, num]]
    """
    measure_head, measure_rest = measure_spec[0], measure_spec[1:]

    if measure_rest:
        setting_results = [
            (
                setting,
                setting_result_tree_measurement(
                    None,
                    results["point_values"],
                    None,
                    measure_rest,
                ),
            )
            for setting, results in setting_results
        ]

    return setting_result_measurements(
        space,
        setting_results,
        trials,
    )[measure_head]

class SamplingResultsNode(Node):
    def __init__(self, expr, measure_spec_str):
        self.expr = expr
        self.measure_spec = measure_spec_str.split(":")

    def __call__(self, ast_path=None):
        setting_result_tree = self.expr(
            ast_path=ast_path + [":".join(self.measure_spec)],
        )["point_values"]

        return setting_result_tree_measurement(
            self.expr.space,
            setting_result_tree,
            self.expr.trials,
            self.measure_spec,
        )

    def __str__(self):
        name = ":".join(self.measure_spec)
        return f"{name}({self.expr})"

    def abstract_expr_str(self, measure_parts=None):
        assert not measure_parts
        return self.expr.abstract_expr_str(measure_parts=self.measure_spec)


class ParamsWrapper(object):
    """
    Wrapper class to prevent hyperopt from sampling from irrelevant spaces
    in a random_search(). If we just used a dictionary, it would replace them
    with samples.
    """
    def __init__(self, params):
        self.params = params

def random_sampling_objective(spec):
    spec, = spec
    func = spec["func"]

    params = updated_params(
        spec["wrapped_params"].params,
        unflattened_params({
            spec["sampling_var"]: spec["sampling_value"],
        }),
    )

    point_hash = params_str(spec["sampling_value"])

    ast_path = list(spec["ast_path"]) + ["point", point_hash]
    with temporary_experiment(spec["experiment"]):
        with temporary_params(params):
            results = func(
                ast_path=ast_path,
            )

    # TODO: Replace these with func[spec[...]](ast_path=ast_path).
    # For the moment we assume we only optimize on top of dictionary results,
    # using same-layer measures. When we have fully functional resume, revisit
    # this to allow for arbitrarily deep optimizations.
    loss_results = results
    if spec.get("minimize_measure", None):
        loss_results = results[spec["minimize_measure"]]
    elif spec.get("maximize_measure", None):
        loss_results = - results[spec["maximize_measure"]]

    # So you can return interesting data.
    if isinstance(loss_results, (int, float)):
        loss = loss_results
    else:
        loss = 0

    return {
        "result": results,
        "loss": loss,
        "status": "ok",
    }

def tpe_estimated_minimum(space, trials):
    hyperopt_trials = trials_from_docs(trials)

    new_hparams = []
    def capture_hparams(hp):
        new_hparams.append(hp)
        raise ValueError

    try:
        fmin(
            capture_hparams,
            space=[space],
            algo=tpe.suggest,
            trials=hyperopt_trials,
            max_evals=len(hyperopt_trials) + 1,
            show_progressbar=False,
        )
    except ValueError as e:
        if not new_hparams:
            raise e

    return space_eval(
        space,
        unwrapped_settings(hyperopt_trials.trials[-1]["misc"]["vals"]),
    )

def tpe_estimated_setting_result(space, trials, point=None):
    assert point in ("minimum", "maximum")
    if point == "maximum":
        trials = deepcopy(trials)
        for trial in trials:
            trial["result"]["loss"] = - trial["result"]["loss"]

    return tpe_estimated_minimum(space, trials)


def setting_result_measurements(
        space,
        setting_results,
        trials,
        extra_measurements=None,
):
    results = [result for setting, result in setting_results]
    if isinstance(results[0], (int, float)):
        result_dist = np.array(results)

        sorted_setting_results = sorted(
            setting_results,
            key=lambda setting_result: setting_result[1],
        )
    else:
        result_dist = np.zeros(len(results))

        sorted_setting_results = setting_results

    if space is not None and trials is not None:
        model_argmin = tpe_estimated_setting_result(
            space,
            trials,
            point="maximum",
        )
        model_argmax = tpe_estimated_setting_result(
            space,
            trials,
            point="minimum",
        )
    else:
        model_argmax = model_argmin = None

    return dict(**{
        "mean": result_dist.mean(),
        "std": result_dist.std(),
        "min": result_dist.min(),
        "max": result_dist.max(),
        "cdf": np.sort(result_dist),
        "argmin": sorted_setting_results[0][0],
        "argmax": sorted_setting_results[-1][0],
        "point_values": sorted_setting_results,
        "model_argmin": model_argmin,
        "model_argmax": model_argmax,
    }, **(extra_measurements or {})
    )



class SamplingNode(Node):
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
            strategy="randomize",
            sample_count=None,
            method=None,
            threads=None,
            minimize_measure=None,
            maximize_measure=None,
    ):

        if isinstance(sampling_var_space, tuple):
            sampling_var, sampling_space = sampling_var_space
        else:
            sampling_var = sampling_var_space
            sampling_space = sampling_var_space + "_space"

        self.sampling_var = sampling_var
        self.sampling_space = sampling_space

        self.func = func

        self.strategy = strategy
        self.sample_count = sample_count
        if (strategy == "enumerate"
            and isinstance(self.sampling_space, (list, tuple))):
            self.sample_count = len(self.sampling_space)

        assert not (minimize_measure and maximize_measure)
        self.minimize_measure = minimize_measure
        self.maximize_measure = maximize_measure

        self.method = method
        self.thread_count = threads

        self.launched = False
        self.collected = False

    def interruptable(self):
        return True

    def launch(self, ast_path=None):
        if not self.launched:
            self.bind_params()
            self.load_session(ast_path=ast_path)

            # Skip launch on resume if already completed.
            status = search_session_progress(self.session_name)["status"]
            if status != "complete":
                if self.method == "cpu":
                    self.launch_cpu()
                elif self.method == "slurm":
                    self.launch_slurm()

            self.launched = True

    def __getitem__(self, measure_spec):
        if isinstance(measure_spec, tuple):
            measure_spec, = measure_spec
        return SamplingResultsNode(self, measure_spec)

    def bind_params(self):
        if isinstance(self.sampling_space, str):
            self.sampling_space = param(self.sampling_space)
            if self.strategy == "enumerate":
                assert isinstance(self.sampling_space, (list, tuple))
                self.sample_count = len(self.sampling_space)

        if isinstance(self.sample_count, str):
            self.sample_count = int(param(self.sample_count))

        if self.method not in ("inline", "cpu", "slurm"):
            self.method = param(self.method)

        if isinstance(self.thread_count, str):
            self.thread_count = int(param(self.thread_count))

        self.params = params()

        self.space = self.sampling_space
        if self.strategy == "enumerate":
            self.space = hp.choice(self.sampling_var, self.sampling_space)


    def load_session(self, ast_path):
        partial_session_name = experiment_partial_result(
            current_experiment(),
            ast_path,
        )
        if partial_session_name:
            reset_active_search_trials(partial_session_name)
            self.session_name = partial_session_name
        else:
            self.register_session(ast_path)

    def register_session(self, ast_path):
        session_type = "slurm" if self.method == "slurm" else "sampling"

        session_name = unused_session_name(session_type=session_type)

        # Wrapped because hyperopt is weird.
        search_space_spec = [{
            "func": self.func,
            "wrapped_params": ParamsWrapper(self.params),
            "sampling_var": self.sampling_var,
            "sampling_value": self.sampling_space,
            "ast_path": ast_path,
            "experiment": current_experiment(),
            "minimize_measure": self.minimize_measure,
            "maximize_measure": self.maximize_measure,
        }]

        algo = {
            "enumerate": "enumeration",
            "randomize": "rand",
            "minimize": "tpe",
            "maximize": "tpe",
        }[self.strategy]

        create_search_session(
            session_name,
            objective=random_sampling_objective,
            algo=algo,
            space=search_space_spec,
            max_evals=self.sample_count,

            # Metadata.
            params=self.params,
            sampling_var=self.sampling_var,
            sampling_space=self.sampling_space,
            strategy=self.strategy,

            interruptable=self.func.interruptable(),
        )

        # Register so we know how to resume
        update_experiment_partial_result(
            current_experiment(),
            ast_path,
            session_name,
        )

        self.session_name = session_name

    def run_inline(self):
        timeout = slurm_iteration_timeout()
        try:

            durations = []
            while True:
                start_time = time()

                self.sample_once()

                durations.append(time() - start_time)

                if timeout and not self.func.interruptable():
                    remaining_time = timeout - sum(durations)
                    avg_duration = sum(durations) / len(durations)
                    if avg_duration < remaining_time:
                        raise TimeoutError("Cannot complete next trial before timeout.")

        except ValueError as e:
            assert str(e) == "No more trials to run."

    def sample_once(self):
        worker_id = slurm_worker_id()

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

    def __call__(self, ast_path=None):

        self.launch(ast_path=ast_path)

        self.collect_results()

        return setting_result_measurements(
            self.space,
            self.setting_results,
            self.trials,
            extra_measurements={"session_name": self.session_name}
        )

    def collect_results(self):
        if not self.collected:
            self.wait()

            status = search_session_progress(
                self.session_name
            )["status"]
            assert status == "complete", status

            results = search_session_results(self.session_name)

            self.setting_results = [
                (
                    space_eval(self.space, setting),
                    results["result"],
                )
                for setting, results in results["setting_results"]
            ]

            self.trials = results["trials"]

            self.collected = True

    def __str__(self):
        func_str = str(self.func).replace('\n', '\n    ')
        return (
            f"Sampling[{self.sampling_var} ~ {self.sampling_space}](\n" +
            f"    {func_str},\n" +
            f"    sample_count={self.sample_count},\n" +
            f"    method={self.method},\n" +
            f"    threads={self.thread_count},\n" +
            f"    strategy={self.strategy},\n" +
            "]"
        )

    def abstract_expr_str(self, measure_parts=None):
        if measure_parts:
            measure_head, measure_rest = measure_parts[0], measure_parts[1:]
            measure_str = measure_head + "_"
        else:
            measure_head, measure_rest = None, None
            measure_str = ""

        strategy_str = {
            "enumerate": "enumeration_sampling",
            "randomize": "random_sampling",
            "minimize": "minimizing_sampling",
            "maximize": "maximizing_sampling",
        }[self.strategy]

        func_str = self.func.abstract_expr_str(
            measure_parts=measure_rest,
        ).replace('\n', '\n    ')

        func_name_str = f"{measure_str}{strategy_str}"
        sample_sym = "~" if self.strategy != "enumerate" else "in"
        if measure_head == "mean":
            if self.strategy == "enumerate":
                func_name_str = "1/n ∑"
            elif self.strategy == "randomize":
                func_name_str = "E"
        elif measure_head == "std":
            if self.strategy in ("randomize", "enumerate"):
                func_name_str = "√ Var"
        elif measure_head in ("min", "max","cdf", "argmin", "argmax"):
            func_name_str = measure_head

        if measure_head == "point_values":
            return (
                f"λ {self.sampling_var} in {self.sampling_space} -> (\n"
                f"    {func_str}\n"
                f")"
            )

        return (
            f"{func_name_str}" +
            f"({self.sampling_var} {sample_sym} {self.sampling_space}) [\n" +
            f"    {func_str}\n"
            "]"
        )

def random_sampling(*args, **kwargs):
    return SamplingNode(*args, strategy="randomize", **kwargs)

def enumeration_sampling(*args, **kwargs):
    return SamplingNode(*args, strategy="enumerate", **kwargs)

def minimizing_sampling(*args, **kwargs):
    return SamplingNode(*args, strategy="minimize", **kwargs)

def maximizing_sampling(*args, **kwargs):
    return SamplingNode(*args, strategy="maximize", **kwargs)
