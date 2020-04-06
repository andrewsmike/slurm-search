from sys import argv, exit

def active_slurm_searches():
    """
    For later.
    """
    raise NotImplementedError

def launch_slurm_search():
    pass

def work_on_slurm_search():
    pass

def kill_slurm_search():
    pass

def resume_slurm_search():
    """
    For later.
    """
    raise NotImplementedError


def main():
    command = "list"

    action_func = {
        "list": active_slurm_searches,
        "launch": launch_slurm_search,
        "relaunch": relaunch_slurm_search,
        "kill": kill_slurm_search,
        "work_on": work_on_slurm_search,
    }[command]

    return action_func(*args)


if __name__ == "__main__":
    exit(main())
