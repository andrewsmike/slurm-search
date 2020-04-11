from pprint import pprint
from sys import argv, exit

from search_session import search_session_names

def start_slurm_search():
    raise NotImplementedError

def restart_slurm_search(session_name):
    raise NotImplementedError

def stop_slurm_search(session_name):
    disable_search_session(session_name)
    print("Disabled search. No more trials will start.")
    print("TODO: You may want to kill the remaining slurm workers.")
    # TODO: Kill the slurm jobs.

def list_slurm_searches():
    pprint(search_session_names())
    # TODO: Print actual slurm job names.

def demo_objective(spec):
    x, y = spec
    sleep(4)
    return {
        "loss": x ** 2 + y ** 2,
        "status": "ok",
    }


def work_on_slurm_search(session_name, *args):
    objective = demo_objective

    try:
        while True:
            print(f"[{session_name}] Getting next trial.")
            trial_id, trial_hparams = next_search_trial(session_name)
            print(f"[{session_name}] Next trial: {trial_id}:{trial_hparams}.")
            results = objective(trial_hparams)
            print(f"[{session_name}] Writing back results: {trial_id}:{trial_hparams} => {loss}.")
            update_search_results(session_name, trial_id, trial_hparams, results)
            print(f"[{session_name}] Results recorded.")
    except ValueError e:
        print("Hit exception: {e}")



def main(args):
    if not args:
        print_usage()
        return -1

    command = args[1]

    command_func = {
        "list": list_slurm_searches,
        "launch": launch_slurm_search,
        "relaunch": relaunch_slurm_search,
        "kill": kill_slurm_search,
        "work_on": work_on_slurm_search,
    }.get(command, None)

    if not command_func:
        print(f"Invalid command '{command}'.")
        print("Valid commands: " + ",".join(command_func.keys()))

    return command_func(*args[1:])


if __name__ == "__main__":
    exit(main(argv[1:]))
