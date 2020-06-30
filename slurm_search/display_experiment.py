from code import interact
from math import log
from pprint import pprint
from sys import argv

from hyperopt import hp
import numpy as np
from sys import argv

from slurm_search.experiment import (
    experiment_details,
    experiment_subsessions,
    run_experiment,
)
from slurm_search.experiments import experiment_spec
from slurm_search.params import (
    params_from_args,
    unflattened_params,
    updated_params,
)
from slurm_search.slurm_search import (
    display_slurm_searches,
    stop_slurm_searches,
)

def display_experiment():
    args = argv[1:]

    if not args:
        print("Experiments:")
        print(",".join(experiment_spec.keys()))
        return

    exp_name, args = args[0], args[1:]
    params = unflattened_params(params_from_args(args))

    meta_param_names = {"resume", "display-ast", "debug", "show", "stop"}
    meta_params = {
        param_name: params.get(param_name, None)
        for param_name in meta_param_names
    }

    params = {
        param_name: param_value
        for param_name, param_value in params.items()
        if param_name not in meta_param_names
    }

    exp_spec = experiment_spec[exp_name]

    overrides = params
    if meta_params.get("debug", "true") == "true":
        overrides = updated_params(
            params,
            exp_spec["debug_overrides"],
        )

    stop_session_name = meta_params.get("stop", None)
    if stop_session_name is not None:
        subsession_names = experiment_subsessions(stop_session_name)
        stop_session_names = [stop_session_name] + list(subsession_names)
        stop_slurm_searches(*stop_session_names)
        return

    show_session_name = meta_params.get("show", None)
    if show_session_name:
        subsession_names = experiment_subsessions(show_session_name, ordered=True)
        display_slurm_searches(session_names=subsession_names)
        return

    all_params = updated_params(exp_spec["config"], overrides)

    print("Override params:")
    pprint(overrides)
    print("All params:")
    pprint(all_params)

    if meta_params.get("display-ast", False):
        details = experiment_details(
            exp_spec["experiment_func"],
            defaults=exp_spec["config"],
            overrides=overrides,
            abstract=meta_params["display-ast"] == "abstract",
        )
        if not isinstance(details, dict):
            print(details)
        else:
            for key, value in details.items():
                print(f"{key}:")
                if isinstance(value, (tuple, list)):
                    for subvalue in value:
                        print(subvalue)
                else:
                    print(value)
        return

    session_name, results = run_experiment(
        exp_name,
        exp_spec["experiment_func"],
        exp_spec["config"],
        overrides,
        resume=meta_params.get("resume", None),
    )

    pprint(results)
    print(session_name)
    exp_spec["display_func"](session_name, all_params, results)

    interact(local=locals())
