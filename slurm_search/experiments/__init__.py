from .all_tools import *
from slurm_search.experiments.compare_agents import compare_agents_exp
from slurm_search.experiments.compare_hp_spaces import compare_hp_spaces_exp
from slurm_search.experiments.hp_tuning_effects import hp_tuning_effects_exp
from slurm_search.experiments.hp_tuning_effects_rand import hp_tuning_effects_rand_exp
from slurm_search.experiments.hp_tuning_model_best import hp_tuning_model_best_exp

from slurm_search.experiments.multienv_tuning_beta import multienv_tuning_beta_exp
from slurm_search.experiments.multienv_benchmark import multienv_benchmark_exp
from slurm_search.experiments.multienv_atari_benchmark import multienv_atari_benchmark_exp
from slurm_search.experiments.multienv_atari_benchmark_dqn import multienv_atari_benchmark_dqn_exp
from slurm_search.experiments.multienv_atari_benchmark_vsarsa import multienv_atari_benchmark_vsarsa_exp

from slurm_search.experiments.gen_benchmark import gen_benchmark_exp
from slurm_search.experiments.multienv_benchmark_tuning import (
    multienv_benchmark_tuning_exp,
    multienv_a2c_benchmark_tuning_exp,
    multienv_ppo_benchmark_tuning_exp,
    multienv_vsarsa_benchmark_tuning_exp,
    multienv_dqn_benchmark_tuning_exp,
)


from slurm_search.experiments.multienv_hp_tuning_effects import multienv_hp_tuning_effects_exp
from slurm_search.experiments.tuning_config_effects import tuning_config_effects_exp

experiment_spec = {
    "compare_agents": compare_agents_exp,
    "compare_hp_spaces": compare_hp_spaces_exp,
    "multienv_hp_tuning_effects": multienv_hp_tuning_effects_exp,

    "multienv_tuning_beta": multienv_tuning_beta_exp,
    "multienv_benchmark": multienv_benchmark_exp,

    "multienv_atari_benchmark": multienv_atari_benchmark_exp,
    "multienv_atari_benchmark_dqn": multienv_atari_benchmark_dqn_exp,
    "multienv_atari_benchmark_vsarsa": multienv_atari_benchmark_vsarsa_exp,

    "gen_benchmark": gen_benchmark_exp,
    "multienv_benchmark_tuning": multienv_benchmark_tuning_exp,
    "multienv_a2c_benchmark_tuning": multienv_a2c_benchmark_tuning_exp,
    "multienv_ppo_benchmark_tuning": multienv_ppo_benchmark_tuning_exp,
    "multienv_vsarsa_benchmark_tuning": multienv_vsarsa_benchmark_tuning_exp,
    "multienv_dqn_benchmark_tuning": multienv_dqn_benchmark_tuning_exp,

    "hp_tuning_effects": hp_tuning_effects_exp,
    "hp_tuning_effects_rand": hp_tuning_effects_rand_exp,
    "hp_tuning_model_best": hp_tuning_model_best_exp,
    "tuning_config_effects": tuning_config_effects_exp,
}
