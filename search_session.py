"""
Search session interactions.
These functions implement a distributed search session API over the search state
and locking mechanisms. They automatically grab the appropriate locks and are
specialized to hyperopt.
"""
from copy import deepcopy
from time import sleep

from hyperopt import fmin, tpe, rand, Trials, trials_from_docs

from locking import lock
from search_state import *

__all__ = [
    "create_search_session",
    "search_session_exists",
    "search_session_names",
    "search_session_active",
    "next_search_trial",
    "update_search_results",
    "delete_search_session",
]


# State -> hyperopt -> state interactions.
def state_next_trial(state):
    sleep(4)
    prev_trials = state['trials']
    hyperopt_trials = trials_from_docs(prev_trials)

    new_trial = []
    new_hparams = []
    def capture_hparams(hp):
        new_trial.append(deepcopy(hyperopt_trials._trials[-1]))
        new_hparams.append(hp)
        raise ValueError

    algo = {
        "rand": rand.suggest,
        "tpe": tpe.suggest,
    }[state["algo"]]

    try:
        fmin(
            capture_hparams,
            space=state["space"],
            algo=algo,
            trials=hyperopt_trials,
            max_evals=len(prev_trials) + 1,
            show_progressbar=False,
        )
    except:
        pass

    next_state = dict(state)
    next_state['trials'] = prev_trials + [new_trial[0]]

    return (new_trial[0]["tid"], new_hparams[0]), next_state

def state_updated_with_results(state, trial_id, hparams, results):
    next_state = dict(state)
    trials = []
    for trial in state["trials"]:
        if trial["tid"] == trial_id:
            trial = dict(trial)
            trial["result"] = results
        trials.append(trial)

    next_state["trials"] = trials

    return next_state

def trials_exhausted(search_state):
    return (
        len(search_state["trials"]) >= search_state.get("max_evals", 1e18)
        or search_state["status"] != "active"
        # TODO: Timeout.
    )

def all_trials_complete(search_state):
    return all(
        trial["result"]["status"] == "ok"
        for trial in state["trials"]
    )

# Outwards facing functions.
def search_session_exists(session_name):
    with lock(session_name):
        return search_state_exists(session_name)

def search_session_active(session_name):
    """
    Sessions that are marked as active and still have additional trials to assign.
    These exclude sessions that are just waiting on the final round of
    results.
    When we store slurm job info in the search state, we can get more featureful.
    """
    with lock(session_name):
        state = search_state(session_name)
        return (
            state["status"] == "active"
            and not trials_exhausted(state)
        )

def search_session_names():
    return [
        for session_name in search_state_session_names()
        if search_session_active(session_name)
    ]

def create_search_session(session_name, **args):
    with lock(session_name):
        create_search_state(
            session_name,
            dict(**args, **{"trials": []}),
        )

def next_search_trial(session_name):
    """
    Atomically decide the next hparam values and progress the session state.
    """
    with lock(session_name):
        state = search_state(session_name)

        if trials_exhausted(state):
            print("No more trials to run.")
            raise ValueError(
                "No more trials to run."
            )

        (trial_id, trial_hparams), next_state = state_next_trial(state)

        update_search_state(session_name, next_state)

    return trial_id, trial_hparams

def update_search_results(session_name, trial_id, hparams, results):
    """
    Atomically record results into session's state.
    """
    with lock(session_name):
        state = search_state(session_name)

        state = state_updated_with_results(state, trial_id, hparams, results)

        if trials_exhausted(state) and all_trials_complete(state):
            state["status"] = "complete"

        update_search_state(session_name, state)

def disable_search_session(session_name):
    with lock(session_name):
        state = search_state(session_name)
        state["status"] = "disabled"
        update_search_state(session_name, state)

def delete_search_session(session_name):
    raise ValueError("Why are you programmatically deleting files? Stop that.")
    with lock(session_name):
        delete_search_state(session_name)

