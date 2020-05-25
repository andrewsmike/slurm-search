from .all_tools import *
from slurm_search.experiments.hp_tuning_effects import hp_tuning_effects_exp
from slurm_search.experiments.tuning_config_effects import tuning_config_effects_exp

experiment_spec = {
    "hp_tuning_effects": hp_tuning_effects_exp,
    "tuning_config_effects": tuning_config_effects_exp,
}
