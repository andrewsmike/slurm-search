"""
Filesystem based mutex locking.

Uses module-level state for tracking this processs' locked paths.
Does no path resolution, deduplication, or counting: These may be implemented
as necessary.
"""
from contextlib import contextmanager
from fcntl import flock, LOCK_EX, LOCK_UN
from functools import wraps
from os.path import expanduser

__all__ = [
    "have_lock",
    "lock",
    "needs_lock",
]

def lock_file_path(lock_name):
    return expanduser(f"~/tmp/{lock_name}.lock")

_file_lock_paths = set()
def have_lock(lock_name):
    return lock_file_path(lock_name) in _file_lock_paths

@contextmanager
def lock(lock_name):
    assert not have_lock(lock_name), (
        "Attempted to obtain a lock that this process already owns."
    )
    path = lock_file_path(lock_name)
    lock_file = open(path, "w")
    flock(lock_file.fileno(), LOCK_EX)
    _file_lock_paths.add(path)

    try:
        yield None
    finally:
        _file_lock_paths.remove(path)
        flock(lock_file.fileno(), LOCK_UN)
        lock_file.close()

def needs_lock(func):
    """
    Asserts ownership of lock named in first positional argument.

    :Example:
    >>> @needs_lock
    ... def my_func(session_name, a, b, c=None):
    ...     return a + b if c is None else a + b + c

    >>> my_func("my_session", 1, 2, c=3)
    Traceback (most recent call last):
      ...
    AssertionError: Must have exclusive lock on 'my_session' to call my_func.
    Please wrap call in `with lock(lock_name):` context (from locking module.)

    >>> with lock("my_session"):
    ...     my_func("my_session", 1, 2, c=3)
    6
    """
    @wraps(func)
    def wrapped(*args, **kwargs):
        session_name = args[0]
        assert have_lock(session_name), (
            f"Must have exclusive lock on '{session_name}' to call {func.__name__}.\n" +
            "Please wrap call in `with lock(lock_name):` context (from locking module.)"
        )

        return func(*args, **kwargs)

    return wrapped
