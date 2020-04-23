"""
Search state logic.
These methods are CRUD-like actions that exist inside of a mutex lock.
These allow above methods to manipulate the search state as a pickle-able CRUD
object.
"""

from glob import glob
from pickle import load, dump
from os import makedirs, remove
from os.path import expanduser, exists

from locking import lock, needs_lock

__all__ = [
    "create_search_state",
    "search_state_exists",
    "search_state_session_names",
    "search_state",
    "update_search_state",
    "delete_search_state",
]

def search_state_path(session_name):
    return expanduser(f"~/hyperparameters/search/{session_name}_search_state.pickle")

def path_search_session_name(path):
    return path.split("/")[-1].split("_search_state.pickle")[0]

def search_state_session_names():
    search_state_pattern = expanduser(f"~/hyperparameters/search/*_search_state.pickle")
    search_state_paths = glob(search_state_pattern)
    return [
        path_search_session_name(search_state_path)
        for search_state_path in search_state_paths
    ]


@needs_lock
def search_state_exists(session_name):
    return exists(search_state_path(session_name))

@needs_lock
def create_search_state(session_name, state):
    assert not search_state_exists(session_name), (
        f"Search state '{session_name}' already exists. " +
        "Please remove or kill_search_state() it to continue."
    )

    makedirs(search_state_dir(session_name), exist_ok=False)

    update_search_state(
        session_name,
        state,
    )

@needs_lock
def search_state(session_name):
    with open(search_state_path(session_name), "rb") as f:
        content = load(f)

    return content

@needs_lock
def update_search_state(session_name, search_state):
    with open(search_state_path(session_name), "wb") as f:
        dump(search_state, f)

@needs_lock
def delete_search_state(session_name):
    raise ValueError("Why are you programmatically deleting files? Stop that.")
    remove(search_state_path(session_name))

