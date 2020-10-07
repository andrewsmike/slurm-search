"""
Provide env / agent performance baseline data from benchmarks, ALE paper, and experiments.
Sources include https://arxiv.org/pdf/1207.4708.pdf, Appendix D.1, Table 4,
manual entry, and simulations.
"""
from csv import reader
from functools import lru_cache
from os.path import expanduser
from pickle import load

agent_names = {
    "Basic",
    "BASS",
    "DISCO",
    "LSH",
    "RAM",
    "Random",
    "Const",
    "Perturb",
}

env_names = {
    "Asterix", "BeamRider", "Freeway", "Seaquest",
    "SpaceInvaders", "Alien", "Amidar", "Assault",
    "Asteroids", "Atlantis", "BankHeist", "BattleZone",
    "Berzerk", "Bowling", "Boxing", "Breakout",
    "Carnival", "Centipede", "ChopperCommand", "CrazyClimber",
    "DemonAttack", "DoubleDunk", "ElevatorAction", "Enduro",
    "FishingDerby", "Frostbite", "Gopher", "Gravitar",
    "Hero", "IceHockey", "Jamesbond", "JourneyEscape",
    "Kangaroo", "Krull", "KungFuMaster", "MontezumaRevenge",
    "MsPacman", "NameThisGame", "Pooyan", "Pong",
    "PrivateEye", "Qbert", "Riverraid", "RoadRunner",
    "Robotank", "Skiing", "StarGunner", "Tennis",
    "TimePilot", "Tutankham", "UpNDown", "Venture",
    "VideoPinball", "WizardOfWor", "Zaxxon",
}

DEFAULT_ENV_PATH = (
    "~/src/slurm_search/slurm_search/experiments/env_baselines.csv"
)

@lru_cache(maxsize=None)
def env_agent_data(path=None):
    path = path or expanduser(DEFAULT_ENV_PATH)

    with open(path, "r") as f:
        lines = list(reader(f))

    header = lines[0]

    return [
        dict(zip(header, line))
        for line in lines[1:]
    ]

def env_agent_score(env, agent, path=None):
    rows = env_agent_data(path)

    env_row, = [row for row in rows if row["ALE Game Name"] == env]

    return env_row["agent"]

@lru_cache(maxsize=None)
def env_min_max_scores(path=None):
    rows = env_agent_data(path)

    return {
        row["ALE Game Name"]: (
            min(float(row[agent_name]) for agent_name in agent_names),
            max(float(row[agent_name]) for agent_name in agent_names),
        )
        for row in rows
    }


@lru_cache(maxsize=None)
def benchmark_data(benchmark_name):
    with open(expanduser(f"~/benchmark_{benchmark_name}.pkl"), "rb") as f:
        return load(f)

def benchmark_env_cdf(benchmark_name, env):
    return benchmark_data(benchmark_name)["env_cdfs"][env]

if __name__ == "__main__":
    from pprint import pprint
    pprint(env_agent_data())
    pprint(env_min_max_scores())
    exit(0)
