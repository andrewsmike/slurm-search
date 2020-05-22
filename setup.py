from setuptools import find_packages, setup

setup(
    name='slurm_search',
    version='0.1.0',

    description="Perform hyperparameter searches using SLURM.",

    packages=find_packages(),

    # TBD
    install_requires=[
    ],

    entry_points={
        "console_scripts": [
            "ssearch=slurm_search:main",
            "exp_hp_imp=slurm_search.experiments:demo_hp_improvement",
            "exp_tuning_curve=slurm_search.experiments:demo_tuning_curve",
        ],
    }
)
