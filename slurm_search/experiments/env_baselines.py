"""
Provide env / agent performance baseline data from ALE paper and experiments.
Sources include https://arxiv.org/pdf/1207.4708.pdf, Appendix D.1, Table 4, and
manual entry.
"""
from csv import reader
from functools import lru_cache
from os.path import expanduser

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
    "Asterix", "Beam Rider", "Freeway", "Seaquest",
    "Space Invaders", "Alien", "Amidar", "Assault",
    "Asteroids", "Atlantis", "Bank Heist", "Battle Zone",
    "Berzerk", "Bowling", "Boxing", "Breakout",
    "Carnival", "Centipede", "Chopper Command", "Crazy Climber",
    "Demon Attack", "Double Dunk", "Elevator Action", "Enduro",
    "Fishing Derby", "Frostbite", "Gopher", "Gravitar",
    "H.E.R.O.", "Ice Hockey", "James Bond", "Journey Escape",
    "Kangaroo", "Krull", "Kung-Fu Master", "Montezuma’s Revenge",
    "Ms. Pac-Man", "Name This Game", "Pooyan", "Pong",
    "Private Eye", "Q*Bert", "River Raid", "Road Runner",
    "Robotank", "Skiing", "Star Gunner", "Tennis",
    "Time Pilot", "Tutankham", "Up and Down", "Venture",
    "Video Pinball", "Wizard of Wor", "Zaxxon",
}

DEFAULT_ENV_PATH = (
    "~/src/slurm_search/slurm_search/experiments/env_baselines.csv"
)

@lru_cache
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

    env_row, = [row for row in rows if row["Game"] == env]

    return env_row["agent"]

@lru_cache
def env_min_max_scores(path=None):
    rows = env_agent_data(path)

    return {
        row["Game"]: (
            min(float(row[agent_name]) for agent_name in agent_names),
            max(float(row[agent_name]) for agent_name in agent_names),
        )
        for row in rows
    }

if __name__ == "__main__":
    from pprint import pprint
    pprint(env_agent_data())
    pprint(env_min_max_scores())
    exit(0)
