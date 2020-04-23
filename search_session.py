"""
Search session interactions.
These functions implement a distributed search session API over the search state
and locking mechanisms. They automatically grab the appropriate locks and are
specialized to hyperopt.

There are two types of functions: Session management (list, create, delete,
check status, etc) and experiment interactions (get next hparam and progress
hyperopt state, record results in state.)
"""
from copy import deepcopy
from time import sleep

from hyperopt import fmin, tpe, rand, Trials, trials_from_docs

from locking import lock
from search_state import *

__all__ = [
    "create_search_session",
    "delete_search_session",
    "delete_active_search_trials",
    "disable_search_session",
    "enable_search_session",
    "next_search_trial",
    "search_session_objective",
    "search_session_active",
    "search_session_exists",
    "search_session_names",
    "search_session_progress",
    "update_search_results",
]


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

    updated = 0

    trials = []
    for trial in state["trials"]:
        if trial["tid"] == trial_id:
            trial = dict(trial)
            trial["result"] = results
            updated += 1
        trials.append(trial)

    if updated == 0:
        print("Trial wasn't in trials object.")
        raise ValueError("Trial record was deleted, failed to record results.")
    elif updated > 1:
        raise ValueError("Multiple trial records for the same trial ID.")

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
        for trial in search_state["trials"]
    )

def trial_active(trial):
    return (trial["result"].get("status", None) != "ok")

def state_without_active_trials(search_state):
    next_state = dict(search_state)
    next_state["trials"] = [
        trial
        for trial in search_state["trials"]
        if not trial_active(trial)
    ]

    return next_state

##############################
# Outwards facing functions. #
##############################

# Search / trial interactions.
def search_session_objective(session_name):
    with lock(session_name):
        state = search_state(session_name)
        return state["objective"]

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

def delete_active_search_trials(session_name):
    with lock(session_name):
        state = search_state(session_name)

        scrubbed_state = state_without_active_trials(state)

        update_search_state(session_name, scrubbed_state)

# Session management.
def search_session_exists(session_name):
    with lock(session_name):
        return search_state_exists(session_name)

def search_session_active(session_name):
    """
    An active search session does not have all its results yet.
    It may have exhausted its trials, and be waiting on workers.
    Its workers may have been killed anomalously.
    """
    with lock(session_name):
        state = search_state(session_name)
        return state["status"] == "active"

def search_session_progress(session_name):
    with lock(session_name):
        state = search_state(session_name)
        trials = state["trials"]
        return {
            "session_name": session_name,
            "status": state["status"],
            "max_trials": state["max_evals"],
            "completed": sum(
                1
                for trial in trials
                if not trial_active(trial)
            ),
            "running": sum(
                1
                for trial in trials
                if trial_active(trial)
            ),
        }


def search_session_names(including_inactive=False):
    return [
        session_name
        for session_name in search_state_session_names()
        if including_inactive or search_session_active(session_name)
    ]

def create_search_session(session_name, **args):
    with lock(session_name):
        create_search_state(
            session_name,
            dict(**args, **{
                "trials": [],
                "status": "active",
            }),
        )

def disable_search_session(session_name):
    with lock(session_name):
        state = search_state(session_name)
        state["status"] = "disabled"
        update_search_state(session_name, state)

def enable_search_session(session_name):
    with lock(session_name):
        state = search_state(session_name)
        state["status"] = "active"
        update_search_state(session_name, state)

def delete_search_session(session_name):
    raise ValueError("Why are you programmatically deleting files? Stop that.")
    with lock(session_name):
        delete_search_state(session_name)

