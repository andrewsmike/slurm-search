
from time import sleep

from hyperopt import hp

from search_session import (
    create_search_session,
    search_session_exists,
    next_search_trial,
    update_search_results,
)

def slow_objective(spec):
    x, y = spec
    sleep(4)
    return x ** 2 + y ** 2

space = [
    hp.uniform("x", -2, 2),
    hp.uniform("y", -2, 2),
]

# Use at, cron, or what?
def run_trials(session_name):
    if not search_session_exists(session_name):
        create_search_session(
            session_name,
            space=space,
            algo="tpe",
        )

    while True:
        print(f"[{session_name}] Getting next trial.")
        trial_id, trial_hparams = next_search_trial(session_name)
        print(f"[{session_name}] Next trial: {trial_id}:{trial_hparams}.")
        loss = slow_objective(trial_hparams)
        results = {"loss": loss}
        print(f"[{session_name}] Writing back results: {trial_id}:{trial_hparams} => {loss}.")
        update_search_results(session_name, trial_id, trial_hparams, results)
        print(f"[{session_name}] Results recorded.")

def display_stats(session_name):
    pass

def main():
    session_name = "alpha"

    print(f"[{session_name}] Initializing search.")
    try:
        run_trials(session_name)
    except:
        display_stats(session_name)


if __name__ == "__main__":
    main()
