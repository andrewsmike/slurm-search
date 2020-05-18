"""
Session state handling.

These methods are CRUD-like atomic operations against a shared state object
(using file locks and a directory of pickled states.)

Session states must be formatted as "type:session_name", such as
"search:my_best_search".
"""

from glob import glob
from pickle import load, dump
from os import makedirs, remove
from os.path import expanduser, exists

from slurm_search.locking import lock, needs_lock

__all__ = [
    "create_session_state",
    "session_state_exists",
    "session_state_names",
    "session_state",
    "update_session_state",
    "delete_session_state",
]

def session_state_path(session_name):
    session_type, session_name = session_name.split(":")
    return expanduser(
        f"~/hyperparameters/{session_type}/{session_name}/" +
        "session_state.pickle"
    )

def session_state_dir(session_name):
    session_type, session_name = session_name.split(":")
    return expanduser(f"~/hyperparameters/{session_type}/{session_name}")

def path_session_state_name(path):
    path_parts = path.split("/")
    return path_parts[-3] + ":" + path_parts[-2]

def session_state_names():
    session_state_pattern = expanduser(f"~/hyperparameters/*/*/session_state.pickle")
    session_state_paths = glob(session_state_pattern)
    return [
        path_session_state_name(session_state_path)
        for session_state_path in session_state_paths
    ]

@needs_lock
def session_state_exists(session_name):
    return exists(session_state_path(session_name))

@needs_lock
def create_session_state(session_name, state):
    assert not session_state_exists(session_name), (
        f"Session state '{session_name}' already exists. " +
        "Please remove or kill_session_state() it to continue."
    )

    makedirs(session_state_dir(session_name), exist_ok=False)

    update_session_state(
        session_name,
        state,
    )

@needs_lock
def session_state(session_name):
    with open(session_state_path(session_name), "rb") as f:
        content = load(f)

    return content

@needs_lock
def update_session_state(session_name, session_state):
    with open(session_state_path(session_name), "wb") as f:
        dump(session_state, f)

@needs_lock
def delete_session_state(session_name):
    raise ValueError("Why are you programmatically deleting files? Stop that.")
    remove(session_state_path(session_name))

