from .all_tools import *
from slurm_search.experiments.hp_tuning_effects import hp_tuning_effects_exp
from slurm_search.experiments.hp_tuning_effects_rand import hp_tuning_effects_rand_exp
from slurm_search.experiments.hp_tuning_model_best import hp_tuning_model_best_exp
from slurm_search.experiments.tuning_config_effects import tuning_config_effects_exp
from slurm_search.experiments.compare_agents import compare_agents_exp

experiment_spec = {
    "hp_tuning_effects": hp_tuning_effects_exp,
    "hp_tuning_effects_rand": hp_tuning_effects_rand_exp,
    "hp_tuning_model_best": hp_tuning_model_best_exp,
    "tuning_config_effects": tuning_config_effects_exp,
    "compare_agents": compare_agents_exp,
}
