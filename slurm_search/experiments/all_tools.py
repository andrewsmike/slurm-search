from all.environments import AtariEnvironment, GymEnvironment
from all.experiments import ParallelEnvExperiment
from all.presets import atari, classic_control, continuous

from slurm_search.experiment import accepts_param_names


## ALL target function
def run_results(env, agent, hp, run_params, run_seed):

    # Shortcut everything for fast debugging.
    return {"return_mean": run_seed}

    train_frames = run_params["train_frames"]
    train_episodes = run_params["train_episodes"]
    test_episodes = run_params["test_episodes"]

    agent_type, agent_name = agent.split(":")
    env_type, env_name = env.split(":")

    env_func = {
        "classic": GymEnvironment,
        "continuous": GymEnvironment,
        "atari": AtariEnvironment,
    }[env_type]

    env = env_func(env_name, device="cuda")

    agent_mod = {
        "classic": classic_control,
        "continuous": continuous,
        "atari": atari,
    }[agent_type]
    agent_func = getattr(agent_mod, agent_name)

    agent = agent_func(
        device="cuda",
        **hp,
    )

    experiment = ParallelEnvExperiment(
        agent,
        env,
        render=False,
        quiet=True,
        write_loss=False,
    )

    experiment.train(
        frames=train_frames,
        episodes=train_episodes or np.inf,
    )
    returns = experiment.test(
        episodes=test_episodes,
    )[:spec["test_episodes"]] # ALL BUG: May return >test_episodes episodes.
    del experiment

    returns = np.array(returns)

    return {
        "return_mean": returns.mean(),
        "return_var": returns.std(),
        "returns_cdf": np.sorted(returns),
    }

@accepts_param_names
def return_mean(env, agent, hp, run_params, run_seed):
    return run_results(env, agent, hp, run_params, run_seed)["return_mean"]

@accepts_param_names
def return_var(env, agent, hp, run_params, run_seed):
    return run_results(env, agent, hp, run_params, run_seed)["return_var"]

@accepts_param_names
def return_cdf(env, agent, hp, run_params, run_seed):
    return run_results(env, agent, hp, run_params, run_seed)["return_cdf"]

