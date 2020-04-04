"""
Search state logic.
These methods are CRUD-like actions that exist inside of a mutex lock.
These allow above methods to manipulate the search state as a pickle-able CRUD
object.
"""

from locking import lock, needs_lock

from pickle import load, dump
from os import remove
from os.path import expanduser, exists

__all__ = [
    "create_search_state",
    "search_state_exists",
    "search_state",
    "update_search_state",
    "delete_search_state",
]

def search_state_path(session_name):
    return expanduser(f"~/tmp/{session_name}_search_state.pickle")

@needs_lock
def search_state_exists(session_name):
    return exists(search_state_path(session_name))

@needs_lock
def create_search_state(session_name, state):
    assert not search_state_exists(session_name), (
        f"Search state '{session_name}' already exists. " +
        "Please remove or kill_search_state() it to continue."
    )
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

