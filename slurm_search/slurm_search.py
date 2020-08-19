#!/usr/bin/env python3
from code import interact
from copy import deepcopy
from glob import glob
import logging
from os import getenv, getpid, makedirs
from os.path import expanduser, join
from pprint import pprint
from subprocess import check_output, Popen, run, TimeoutExpired
from sys import argv, exit
from threading import get_ident
from time import sleep, time

from slurm_search.locking import lock
from slurm_search.objectives import (
    ale_objective,
    demo_objective,
    search_session_args,
)
from slurm_search.random_phrase import random_phrase
from slurm_search.search_session import (
    create_search_session,
    disable_search_session,
    enable_search_session,
    reset_active_search_trials,
    search_session_interruptable,
    search_session_names,
    search_session_objective,
    search_session_progress,
    search_session_results,
    update_session_state,
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
    if session_name.startswith("slurm:"):
        session_name = session_name[len("slurm:"):]

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
        # 2020-07-23: 136 hit a cuda error.
        "exclude": "node114,node116,node121,node136,node149", # 114 is running really slow. 121 hits CUDA exceptions. 
    }

    dev_environment = getenv("HOSTNAME", None) == "ernie"
    if not dev_environment:
        thread_count = thread_count or 8
        slurm_args.update({
            #"mem-per-cpu": 300,
            #"time": "05:00",
            "time": "03:55:00",
            "mem-per-cpu": 3000,
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
    session_name = "slurm:" + random_phrase()
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
    reset_active_search_trials(session_name)
    enable_search_session(session_name)
    launch_slurm_search_workers(session_name, iteration=1)

def stop_slurm_searches(*session_names):
    for session_name in session_names:
        try:
            disable_search_session(session_name)
            print(f"[{session_name}] Disabled search. No more trials will start.")
        except Exception as e:
            print(f"[{session_name}] Failed to stop search, state probably corrupted.")
            print(e)

    print("TODO: You may want to kill the remaining slurm workers. They could take a while to die off.")

def stop_all_slurm_searches(session_type="slurm"):
    for session_name in search_session_names(
            search_type=session_type,
            filter_inactive=True,
    ):
        try:
            disable_search_session(session_name)
            print(f"[{session_name}] Disabled search. No more trials will start.")
        except Exception as e:
            print(f"Encountered exception trying to stop {session_name}: {e}")

    print("TODO: You may want to kill the remaining slurm workers. They could take a while to die off.")

def display_slurm_searches(search_type="slurm", show_inactive=False, session_names=None):
    if not session_names:
        print("Active sessions (may not have workers):")
        session_names = search_session_names(
            search_type=search_type,
            filter_inactive=not show_inactive,
        )
    else:
        print("Sessions:")

    rows =[
        search_session_progress(
            session_name,
        )
        for session_name in session_names
    ]
    display_table(
        columns=[
            "session_name",
            "session_type",
            "status",
            "running",
            "completed",
            "max_trials",
        ],
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
    print(f"[{session_name}] Loss dist: {float(min(losses)):.4} <= {float(losses.mean()):.4} " +
          f"+/- {float(losses.std()):.4} <= {float(max(losses)):.4}")


    best_setting, best_loss = min(setting_losses, key=lambda trial: trial[1])
    print(f"[{session_name}] Best setting: {best_setting} => " +
          f"Loss={float(best_loss):.4}. Spec:")
    pprint(space_eval(space, best_setting))

    worst_setting, worst_loss = max(setting_losses, key=lambda trial: trial[1])
    print(f"[{session_name}] Worst setting: {worst_setting} => " +
          f"Loss={float(worst_loss):.4}. Spec:")
    pprint(space_eval(space, worst_setting))

def display_slurm_search_state(session_name, key=None):
    with lock(session_name):
        state = session_state(session_name)

    if key:
        pprint(state[key])
    else:
        pprint(state)

    if len(state.get("trials", [])) > 0:
        display_results_summary(
            session_name,
        )

def inspect_slurm_search_state(session_name):
    with lock(session_name):
        state = session_state(session_name)

    interact(
        local={
            "deepcopy": deepcopy,
            "lock": lock,
            "pprint": pprint,
            "session_name": session_name,
            "state": state,
            "update_session_state": update_session_state,
        },
        banner=(
            "Dropping into interactive session.\n"
            "Try 'pprint(state)'."
        ),
    )

def del_slurm_search_state_trial(session_name, trial_index):
    print(f"[{session_name}] Deleting trial #[{trial_index}].")
    with lock(session_name):
        state = session_state(session_name)
        del state["trials"][int(trial_index)]
        update_session_state(session_name, state)

def display_search_session_logs(session_name, worker_number=None):
    session_name = session_name.split(":")[1]
    logs_dir = expanduser(f"~/hyperparameters/search/{session_name}/logs")
    if worker_number is None:
        log_files = glob(join(logs_dir, "worker_*_*.*"))
    else:
        log_files = glob(join(logs_dir, f"worker_*_{worker_number}.*"))

    run(["tail", "-n", "+1"] + log_files)


def work_on_slurm_search(session_name, timeout=None):
    """
    Work on a particular search.
    """
    timeout = float(timeout) if timeout is not None else None
    start_time = time()

    # Let children searches check when the deadline is so they don't start
    # anything uninterruptable that they can't finish.
    if timeout:
        deadline = start_time + float(timeout)
        print(f"Setting deadline: {start_time} + {timeout} = {deadline}.")
        set_slurm_deadline(deadline)

    # Objective is a pickled reference to a function.
    # It must be available to pickle during load()ing.
    objective = search_session_objective(session_name)

    interruptable = search_session_interruptable(session_name)

    worker_id = slurm_worker_id()

    try:
        durations = []
        while True:

            start_time = time()

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

            durations.append(time() - start_time)

            print("Durations: " + str(durations))
            if timeout and not interruptable:
                remaining_time = deadline - time()
                avg_duration = sum(durations) / len(durations)
                print(f"Avg duration, remaining time: {avg_duration}, {remaining_time}")
                if avg_duration > remaining_time:
                    return -1

    except TimeoutError as e:
        print("Got timeout error. Letting search continue.")
        return -1
    except ValueError as e:
        print(f"Hit exception: {e}")
        return 0

def user():
    return getenv("USER")

def slurm_iteration():
    slurm_job_name = getenv("SLURM_JOB_NAME")
    if slurm_job_name:
        return int(slurm_job_name.split("_")[-1])
    else:
        return None

def slurm_array_task_index():
    task_index = getenv("SLURM_ARRAY_TASK_ID", None)
    if task_index is None:
        return task_index
    else:
        return int(task_index)

def slurm_array_task_count():
    return (
        int(getenv("SLURM_ARRAY_TASK_MAX", 0))
        - int(getenv("SLURM_ARRAY_TASK_MIN", 0))
    ) + 1

def slurm_worker_id():
    job_id = getenv("SLURM_JOB_ID", "UNKNOWN")
    task_index = slurm_array_task_index() or 0
    pid, tid = getpid(), get_ident()
    return f"{job_id}_{task_index}_{pid}_{tid}"

def launch_next_generation(session_name, iteration, thread_count=None):
    print("Resetting incomplete trials...")
    reset_active_search_trials(session_name)

    print("Spawning the next generation...")
    launch_slurm_search_workers(
        session_name,
        iteration=(slurm_iteration() + 1),
        thread_count=thread_count,
    )


def update_slurm_iteration_timeout(session_name, iteration, timeout):
    start_time = time()

    with lock(session_name):
        state = dict(session_state(session_name))
        state["slurm_iteration"] = iteration
        state["slurm_iteration_start_time"] = start_time
        state["slurm_iteration_end_time"] = start_time + timeout

        update_session_state(session_name, state)

def slurm_iteration_remaining_time(session_name, iteration):
    with lock(session_name):
        state = session_state(session_name)

    assert state["slurm_iteration"] == iteration, "Worker started way too late."

    current_time = time()
    return state["slurm_iteration_end_time"] - current_time

_slurm_deadline = None

def set_slurm_deadline(deadline):
    global _slurm_deadline
    _slurm_deadline = deadline

def slurm_iteration_deadline():
    global _slurm_deadline
    if _slurm_deadline is None:
        return None

    return _slurm_deadline

def slurm_iteration_workers(session_name, iteration):
    job_name = slurm_job_name(session_name, iteration=iteration)
    job_id_strs = check_output(
        ["squeue", "-n", job_name, "-u", user(), "-o", "%A", "-h"],
    ).decode().split("\n")

    results = {int(job_id_str) for job_id_str in job_id_strs if job_id_str}

    print(results)
    return results

def wait_on_slurm_search(session_name, iteration):
    search_active = (search_session_progress(session_name)["status"] == "active")
    remaining_workers = len(slurm_iteration_workers(session_name, iteration)) > 1
    remaining_time = slurm_iteration_remaining_time(session_name, iteration) # Unnecessary

    while search_active and remaining_workers and (remaining_time > 0):
        search_active = (search_session_progress(session_name)["status"] == "active")
        remaining_workers = len(slurm_iteration_workers(session_name, iteration)) > 1
        remaining_time = slurm_iteration_remaining_time(session_name, iteration)
        print(f"Workers still running and there is remaining time ({remaining_time}). Waiting 1m.", flush=True)
        sleep(1 * MINUTE)

    return not search_active

def generational_work_on_slurm_search(
        session_name,
        timeout=3 * HOUR + 40 * MINUTE,
        max_iters=24,
):
    """
    Meta-process for launching workers, cleaning up after them, and launching the
    next generation.

    Not primary (slurm array task ID isn't 0):
    - Give primary 20s to get prepared.
    - Run until timeout, deadline, exception, or completion.

    Primary:
    - Set the session iteration deadline.
    - Run until timeout, deadline, exception, or completion.
    - Wait for other workers.
    - If exception, exit.
    - Reset interrupted trials.
    - Launch next generation.
    """
    iteration = slurm_iteration()
    assert iteration, "This command must be run from within a slurm worker."

    am_primary = (slurm_array_task_index() == 0)

    if am_primary:
        print("I am the primary.")
        update_slurm_iteration_timeout(session_name, iteration, timeout)
    else:
        print("I am not the primary.")
        # Allow primary to update timeout.
        sleep(20)

    timeout = slurm_iteration_remaining_time(session_name, iteration)
    start_time = time()

    # Work_on doesn't interrupt on timeout, so use timeout command.
    timeout_work_on_command = [
        "timeout", str(max(int(timeout), 1)),
        "ssearch", "work_on", session_name, str(timeout),
    ]

    work_on_thread_count = 2
    workers = [
        Popen(timeout_work_on_command)
        for work_on_thread in range(work_on_thread_count)
    ]

    print("Waiting on work_on threads.", flush=True)
    for worker in workers:
        worker.wait()

    if not am_primary:
        print("Workers finished and I am not the primary. Exiting.")
        return 0

    print("Work_on threads finished and I am the primary.")
    print("Waiting on other slurm workers.", flush=True)
    search_completed = wait_on_slurm_search(
        session_name,
        iteration,
    )

    if search_completed:
        print("Search is no longer active. Exiting.")
        return 0

    if time() - start_time < 40 * MINUTE:
        print("Workers returned way too quickly, probably hit error. Exiting.")
        return 0

    if max_iters and iteration >= max_iters:
        print(f"Search incomplete, but iteration {iteration} >= " +
              f"max_iters ({max_iters}). Exiting.")
        return 0

    print("Waiting a bit, then launching next generation.", flush=True)
    sleep(20)

    launch_next_generation(
        session_name,
        iteration + 1,
        thread_count=slurm_array_task_count(),
    )

def main():
    args = argv[1:]

    # TODO: Do proper logging.
    logging.disable(logging.CRITICAL)

    command_funcs = {
        "list": display_slurm_searches,
        "start": start_slurm_search,
        "restart": restart_slurm_search,
        "stop": stop_slurm_searches,
        "stopall": stop_all_slurm_searches,
        "show": display_slurm_search_state,
        "inspect_state": inspect_slurm_search_state,
        "del_trial": del_slurm_search_state_trial,
        "logs": display_search_session_logs,
        "work_on": work_on_slurm_search,
        "generational_work_on": generational_work_on_slurm_search,
    }

    no_session_commands = {"list", "start", "stopall"}
    multi_session_commands = {"stop"}

    if not args:
        print(f"Commands: {', '.join(command_funcs.keys())}")
        return -1

    command = args[0]

    command_func = command_funcs.get(command, None)

    if not command_func:
        print(f"Invalid command '{command}'.")
        print("Commands: " + ", ".join(command_funcs.keys()))
        return -1

    if command in multi_session_commands:
        args = [
            ("slurm:" + arg) if ":" not in arg else arg
            for arg in args[1:]
        ]
        return command_func(*args)
        
    elif command not in no_session_commands:
        session_name = args[1]
        if not ":" in session_name:
            session_name = "slurm:" + args[1]

        return command_func(session_name, *args[2:])
    else:
        return command_func(*args[1:])

