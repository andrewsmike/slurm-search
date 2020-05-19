#!/usr/bin/env python3
from code import interact
from glob import glob
import logging
from os import getenv, getpid, makedirs
from os.path import expanduser, join
from pprint import pprint
from subprocess import Popen, run, TimeoutExpired
from sys import argv, exit
from time import sleep

from slurm_search.locking import lock
from slurm_search.objectives import (
    ale_objective,
    demo_objective,
    search_session_args,
)
from slurm_search.random_phrase import random_phrase
from slurm_search.search_session import (
    create_search_session,
    delete_active_search_trials,
    disable_search_session,
    enable_search_session,
    search_session_names,
    search_session_objective,
    search_session_progress,
    search_session_results,
    update_search_results,
    next_search_trial,
)
from slurm_search.session_state import session_state

from hyperopt import hp, space_eval
import numpy as np

# for use in timing code.
MINUTE = 60
HOUR = 60 * MINUTE

def display_table(columns=None, rows=None):
    row_strs = [columns] + [
        [
            str(row[column])
            for column in columns
        ]
        for row in rows
    ]
    column_max_len = {
        column: max(len(row[column_index])
                    for row in row_strs)
        for column_index, column in enumerate(columns)
    }

    print("\n".join(
        " ".join(row[column_index].ljust(column_max_len[column])
                 for column_index, column in enumerate(columns))
        for row in row_strs
    ))

def launch_parallel_search_workers(session_name):
    """
    Launch multiple worker processes on the current host.
    Wait for them to terminate.
    """
    worker_command = ["ssearch", "work_on", session_name]
    PARALLELISM = 8
    workers = [
        Popen(worker_command)
        for i in range(PARALLELISM)
    ]
    print(f"[{session_name}] Workers launched. Waiting on them to return.")
    for worker in workers:
        worker.wait()
    print(f"[{session_name}] Workers finished. Exiting...")

def slurm_job_name(session_name, iteration=0):
    return f"all_auto_{session_name}_{iteration}"

def write_slurm_script(path, slurm_args, command):
    arg_comment_block_str = "\n".join(
        f"#SBATCH --{key}={value}"
        for key, value in slurm_args.items()
    )
    # TODO: Escaping things properly.
    command_str = " ".join(command)

    slurm_script_str = (
        "#!/bin/sh\n\n" +
        f"{arg_comment_block_str}\n" +
        f"{command_str}\n"
    )

    with open(path, "w") as f:
        f.write(slurm_script_str)

def launch_slurm_script(script_path):
    run(['sbatch', script_path])

def launch_slurm_search_workers(session_name, iteration, thread_count=None):
    session_name = session_name.split(":")[1]
    SESSION_DIR = expanduser(f"~/hyperparameters/search/{session_name}/")

    OUTPUT_DIR = join(SESSION_DIR, "logs")
    makedirs(OUTPUT_DIR, exist_ok=True)

    dev_thread_count = thread_count or 4

    slurm_args = {
        "job-name": slurm_job_name(session_name, iteration),
        "output": join(OUTPUT_DIR, f"worker_{iteration}_%a.out"),
        "error": join(OUTPUT_DIR, f"worker_{iteration}_%a.err"),
        "array": f"0-{dev_thread_count - 1}",
        "ntasks": 1,
        "mem-per-cpu": 10,
        "exclude": "node121",
    }

    dev_environment = getenv("HOSTNAME", None) == "ernie"
    if not dev_environment:
        thread_count = thread_count or 8
        slurm_args.update({
            #"mem-per-cpu": 300,
            #"time": "05:00",
            "time": "02:10:00",
            "mem-per-cpu": 2000,
            "gres": "gpu:1",
            "partition": "1080ti-short",
            "array": f"0-{thread_count - 1}",
        })

    script_path = join(SESSION_DIR, f"launch_slurm_{session_name}_{iteration}.sh")

    write_slurm_script(script_path, slurm_args, [
        "ssearch", "generational_work_on", session_name,
    ])

    launch_slurm_script(script_path)

def start_slurm_search(search_type, *args):
    session_name = "search:" + random_phrase()
    print(f"[{session_name}] Creating a new {search_type} search session.")

    search_args = search_session_args(search_type, *args)

    create_search_session(
        session_name,
        **search_args
    )

    print(f"[{session_name}] Launching workers...")
    launch_slurm_search_workers(session_name, iteration=1)

    return session_name


def restart_slurm_search(session_name):
    delete_active_search_trials(session_name)
    enable_search_session(session_name)
    launch_slurm_search_workers(session_name, iteration=1)

def stop_slurm_searches(*session_names):
    for session_name in session_names:
        disable_search_session(session_name)
        print(f"[{session_name}] Disabled search. No more trials will start.")
    print("TODO: You may want to kill the remaining slurm workers. They could take a while to die off.")

def display_slurm_searches(show_inactive=False):
    print("Active sessions (may not have workers):")
    rows =[
        search_session_progress(session_name, hide_type=True)
        for session_name in search_session_names(
                including_inactive=bool(show_inactive),
        )
    ]
    display_table(
        columns=["session_name", "status", "running", "completed", "max_trials"],
        rows=rows,
    )

def unwrapped_settings(settings):
    return {
        key: (value
              if not isinstance(value, list)
                  and len(value) == 0
              else value[0])
        for key, value in settings.items()
    }

def display_results_summary(session_name):
    search_results = search_session_results(session_name)

    space = search_results["search_args"]["space"]

    setting_losses = [
        (settings, results["loss"])
        for settings, results in search_results["setting_results"]
        if "loss" in results
    ]
    if not setting_losses:
        return

    losses = np.array([
        loss
        for setting, loss in setting_losses
    ])
    print(f"[{session_name}] Loss dist: {min(losses):.4} <= {losses.mean():.4} " +
          f"+/- {losses.std():.4} <= {max(losses):.4}")


    best_setting, best_loss = min(setting_losses, key=lambda trial: trial[1])
    print(f"[{session_name}] Best setting: {best_setting} => " +
          f"Loss={best_loss:.4}. Spec:")
    pprint(space_eval(space, best_setting))

    worst_setting, worst_loss = max(setting_losses, key=lambda trial: trial[1])
    print(f"[{session_name}] Worst setting: {worst_setting} => " +
          f"Loss={worst_loss:.4}. Spec:")
    pprint(space_eval(space, worst_setting))

def display_slurm_search_state(session_name):
    with lock(session_name):
        state = session_state(session_name)
        pprint(state)
    if len(state.get("trials", [])) > 0:
        display_results_summary(
            session_name,
        )

def inspect_slurm_search_state(session_name):
    with lock(session_name):
        state = session_state(session_name)

    interact(
        local={"state": state, "pprint": pprint},
        banner=(
            "Dropping into interactive session.\n"
            "Try 'pprint(state)'."
        ),
    )

def display_search_session_logs(session_name):
    session_name = session_name.split(":")[1]
    logs_dir = expanduser(f"~/hyperparameters/search/{session_name}/logs")
    log_files = glob(join(logs_dir, "*"))
    run(["tail", "-n", "+1"] + log_files)


def work_on_slurm_search(session_name):
    """
    Work on a particular search.
    """
    # Objective is a pickled reference to a function.
    # It must be available to pickle during load()ing.
    objective = search_session_objective(session_name)

    worker_id = slurm_worker_id()

    try:
        while True:
            print(f"[{session_name}] Getting next trial.", flush=True)
            trial_id, trial_hparams = next_search_trial(session_name, worker_id)
            print(f"[{session_name}] Next trial: {trial_id}:{trial_hparams}.", flush=True)
            results = objective(trial_hparams)
            print(f"[{session_name}] Writing back results: " +
                  f"{trial_id}:{trial_hparams} => {results['loss']}.", flush=True)
            update_search_results(
                session_name,
                trial_id,
                worker_id,
                trial_hparams,
                results,
            )
            print(f"[{session_name}] Results recorded.", flush=True)
    except ValueError as e:
        print(f"Hit exception: {e}")

def slurm_iteration():
    slurm_job_name = getenv("SLURM_JOB_NAME", None)
    if slurm_job_name:
        return int(slurm_job_name.split("_")[-1])
    else:
        return None

def slurm_array_task_index():
    return int(getenv("SLURM_ARRAY_TASK_ID")) or None

def slurm_worker_id():
    job_id = getenv("SLURM_JOB_ID", "UNKNOWN")
    return f"{job_id}_{slurm_array_task_index() or 0}_{getpid()}"

def launch_next_generation(session_name, iteration):
    print("Waiting for other workers to quit...")
    # TODO: Wait properly
    sleep(2 * MINUTE)

    print("Clearing incomplete trials...")
    delete_active_search_trials(session_name)

    print("Spawning the next generation...")
    launch_slurm_search_workers(
        session_name,
        iteration=(slurm_iteration() + 1),
    )


def generational_work_on_slurm_search(
        session_name,
        timeout=2 * HOUR,
        max_iters=5,
):
    iteration = slurm_iteration()
    assert iteration, "This command must be run from within a slurm worker."

    search_complete = True
    try:
        run(
            ["ssearch", "work_on", session_name],
            timeout=timeout,
        )
    except TimeoutExpired as e:
        search_complete = False
        print("Search timed out.")
    except Exception as e:
        print(f"Search hit {type(e)} exception: '{e}'")
        raise

    if search_complete:
        print("Search completed. Exiting.")
        return 0

    am_not_primary = (slurm_array_task_index() != 0)
    if am_not_primary:
        print("Search incomplete, but I am not the primary. Exiting.")
        return 0

    if max_iters and iteration >= max_iters:
        print(f"Search incomplete, but iteration {iteration} >= " +
              f"max_iters ({max_iters}). Exiting.")
        return 0

    print("Search didn't finish and I am the primary. Spawning next generation.")

    launch_next_generation(session_name, iteration + 1)

def main():
    args = argv[1:]

    # TODO: Do proper logging.
    logging.disable(logging.CRITICAL)

    command_funcs = {
        "list": display_slurm_searches,
        "start": start_slurm_search,
        "restart": restart_slurm_search,
        "stop": stop_slurm_searches,
        "show": display_slurm_search_state,
        "inspect_state": inspect_slurm_search_state,
        "logs": display_search_session_logs,
        "work_on": work_on_slurm_search,
        "generational_work_on": generational_work_on_slurm_search,
    }

    no_session_commands = {"list", "start"}

    if not args:
        print(f"Commands: {', '.join(command_funcs.keys())}")
        return -1

    command = args[0]

    command_func = command_funcs.get(command, None)

    if not command_func:
        print(f"Invalid command '{command}'.")
        print("Commands: " + ", ".join(command_funcs.keys()))
        return -1

    if command not in no_session_commands:
        session_name = args[1]
        if not ":" in session_name:
            session_name = "search:" + args[1]

        return command_func(session_name, *args[2:])
    else:
        return command_func(*args[1:])

