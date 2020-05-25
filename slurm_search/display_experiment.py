from code import interact
from math import log
from pprint import pprint
from sys import argv

from hyperopt import hp
import numpy as np
from sys import argv

from slurm_search.experiment import (
    experiment_details,
    run_experiment,
)
from slurm_search.experiments import experiment_spec
from slurm_search.params import (
    params_from_args,
    unflattened_params,
    updated_params,
)

def display_experiment():
    args = argv[1:]

    if not args:
        print("Experiments:")
        print(",".join(experiment_spec.keys()))
        return

    exp_name, args = args[0], args[1:]
    params = unflattened_params(params_from_args(args))

    meta_param_names = {"resume", "display-ast", "debug"}
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

    print("Override params:")
    pprint(overrides)
    print("All params:")
    pprint(updated_params(exp_spec["config"], overrides))

    if meta_params.get("display-ast", False):
        details = experiment_details(
            exp_spec["exp_func"],
            defaults=exp_spec["config"],
            overrides=overrides,
        )
        for key, value in details.items():
            print(f"{key}:")
            print(value)

    session_name, results = run_experiment(
        exp_spec["experiment_func"],
        exp_spec["config"],
        overrides,
        resume=meta_params.get("resume", None),
    )

    print(session_name)
    exp_spec["display_func"](session_name, results)

    interact(local=locals())
