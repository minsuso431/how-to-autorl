# AutoRL Hydra Sweepers
This folder contain all necessary codes to execute MODEHB with Hydra Sweeper.

## Prerequisite
To use the MODEHB Hydra Sweeper you first need to have access to the original MODEHB code from Awad et al. (https://arxiv.org/abs/2305.04502) (https://github.com/ayushi-3536/MODEHB)

Please rename the dehb folder of the above git repo to modehb and put it manually inside your desired venv/conda environment.


## Examples
In ['examples'](examples) you can see example configurations and setups for all sweepers on Stable Baselines 3 agents.
To run an example with the sweeper, you need to set the '--multirun' flag:
```bash
python examples/modehb_for_pendulum_ppo.py -m
```

## Additional Parameters 
MODEHB has several parameters that are not present in DEHB. These include
- mo_strategy [Type String]: Either "EPSNET" or "NSGA-II"
- num_objectives [Type Integer] 
- stability_objective [Type String]: This parameter is relevant only when optimizing over multiple training seeds.
 One of the following must be chosen
    - "mean_stds": Takes the mean standard deviation of episodic returns across configurations for each training seed.
    - "std_returns": Takes the standard deviation of the episodic mean returns for the seeded configuration.
    - "max_min_return": Takes the maximum of the episodic mean returns for the seeded configuration if maximize = false is selected (This is equivalent to minimizing the maximum of the means). Takes the minium otherwise.
    - "max_min_return_quantile": Takes the third quartil of the episodic mean returns for the seeded configuration if maximize = false is selected (This is equivalent to minimizing the third quartile of the means). Takes the first quartil otherwise.


## Usage in your own project
In your yaml-configuration file, set `hydra/sweeper` to the sweeper name, e.g. `MODEHB`:
```yaml
defaults:
  - override hydra/sweeper: MODEHB
```