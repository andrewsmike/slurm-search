Slurm Search
============
Slurm search is a library for expressing, running, and managing distributed
experiments on a cluster. It has a number of parts, including a storage system,
a job management tool, an experiment management system, hyperparameter tuning
algorithms, and integrations with a reinforcement learning library.

Installation
------------
Install autonomous-learning-library and pytorch before installing this package.
Run the following commands:
```bash
pip install .
mkdir -p ~/hyperparameters/locks # Locks assume their directory already exists.
```

Usage
-----
Experiments are specified in slurm_search/experiments/. Using the run_exp command, you may:
```bash
# Launch an experiment with a series of override parameters.
run_exp hp_tuning_effects --agent=atari:a2c --env=atari:Breakout --search:threads=12

# Relaunch an interrupted experiment, including to repeat an updated result analysis.
run_exp hp_tuning_effects --agent=atari:a2c --env=atari:Breakout --search:threads=12 --resume=exp:my_exp_name

# Display the experiment AST visually
run_exp hp_tuning_effects --display-ast=true # Regular
run_exp hp_tuning_effects --display-ast=abstract # Abstract equations
```


Components
----------
This package has the following parts:
- A pickle-based object database that supports safe concurrent access across a
    cluster.
- A session management system that integrates with the slurm job management
    system for parallel execution of tasks across a cluster.
- [Hyperopt](https://github.com/hyperopt/hyperopt) integrations for parallel
    hyperparameter tuning searches.
- An python experiment DSL and tools for configuring, running, and resuming
    experiments and tools for analyzing the results.
- An integration with the [Autonomoun Learning Library](https://github.com/cpnota/autonomous-learning-library)
    for running reinforcement learning experiments.

Files
-----
Core:
- experiments/: Experiment specifications, domain-of-interest integrations, and
    analysis tools.
- display_experiment.py: Experiment management CLI tool.
- experiment.py: Experiment DSL.
- slurm_search.py: Slurm integration, session and state management CLI tool 'ssearch'.
- search_session.py: Parallelized search session interface.
- session_state.py: Transactional string-identified state store (uses locking and pickle).
- locking.py: String-identified mutex (uses file locks).

Utilities:
- params.py: Tools for handling parameter dict trees.
- random_phrase.py: Generate random phrases / names for object identifiers.
