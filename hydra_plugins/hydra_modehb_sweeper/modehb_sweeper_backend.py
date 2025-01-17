"""
This is an adaptation on MODEHB of original file written by Eimer et al.: https://github.com/facebookresearch/how-to-autorl/tree/main/hydra_plugins/hydra_dehb_sweeper
"""
from __future__ import annotations

from typing import List

import logging
import operator
import os
import copy
from functools import reduce
import pickle

from hydra.core.plugins import Plugins
from hydra.plugins.sweeper import Sweeper
from hydra.types import HydraContext, TaskFunction
from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf, open_dict
from rich import print as printr

from hydra_plugins.hydra_modehb_sweeper.hydra_modehb import HydraMODEHB
from hydra_plugins.utils.search_space_encoding import search_space_to_config_space

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver("get_class", get_class, replace=True)

class MODEHBSweeperBackend(Sweeper):
    def __init__(
        self,
        search_space: DictConfig,
        budget_variable: str | None = None,
        modehb_kwargs: DictConfig | dict = {},
        resume: str | None = None,
        n_jobs: int = 8,
        slurm: bool = False,
        slurm_timeout: int = 10,
        total_function_evaluations: int | None = None,
        total_brackets: int | None = None,
        total_cost: int | None = None,
        total_wallclock_cost: float | None = None,
    ) -> None:
        """
        Backend for the MODEHB sweeper. Instantiate and launch MODEHB's optimization.

        Parameters
        ----------
        search_space: DictConfig
            The search space, either a DictConfig from a hydra yaml config file, or a path to a json configuration space
            file in the format required of ConfigSpace, or already a ConfigurationSpace config space.
        budget_variable: str | None
            Name of the variable controlling the budget, e.g. max_epochs. Only relevant for multi-fidelity methods.
        modehb_kwargs: DictConfig | None
            Keywords for MODEHB
        total_function_evaluations: int | None
            Maximum number of function evaluations for the optimization. One of total_function_evaluations,
            total_brackets, total_cost and total_wallclock_cost must be given.
        total_brackets: int | None
            Maximum number of brackets (Successive Halving brackets run under Hyperband ) for the optimization.
            One of total_function_evaluations, total_brackets, total_cost and total_wallclock_cost must be given.
        total_cost: int | None
            Total amount of seconds for the optimization (i.e. runtimes of all jobs will be summed up for this!).
            (Total computational cost (in seconds) returned by the objective function. Might be simulated costs.)
            One of total_function_evaluations, total_brackets, total_cost and total_wallclock_cost must be given.
        total_wallclock_cost: int | None
            Total computational cost (in seconds) aggregated by all function evaluations (total_wallclock_cost).
            One of total_function_evaluations, total_brackets, total_cost and total_wallclock_cost must be given.

        Returns
        -------
        None

        """
        self.search_space = search_space
        self.modehb_kwargs = modehb_kwargs
        self.budget_variable = budget_variable

        self.fevals = total_function_evaluations
        self.brackets = total_brackets
        self.cost = total_cost
        self.time_cost = total_wallclock_cost
        self.n_jobs = n_jobs
        self.slurm = slurm
        self.slurm_timeout = slurm_timeout
        self.resume = resume

        self.task_function: TaskFunction | None = None
        self.sweep_dir: str | None = None

    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        """
        Setup launcher.

        Parameters
        ----------
        hydra_context: HydraContext
        task_function: TaskFunction
        config: DictConfig

        Returns
        -------
        None

        """
        self.config = config
        self.hydra_context = hydra_context
        self.launcher = Plugins.instance().instantiate_launcher(
            config=config, hydra_context=hydra_context, task_function=task_function
        )
        self.task_function = task_function
        self.sweep_dir = config.hydra.sweep.dir

    def sweep(self, arguments: List[str]) -> List | None:
        """
        Run optimization with MODEHB.

        Parameters
        ----------
        arguments: List[str]
            Hydra overrides for the sweep.

        Returns
        -------
        Configuration | None
            Incumbent (best) configuration.

        """
        assert self.config is not None
        assert self.launcher is not None
        assert self.hydra_context is not None

        printr("Config", self.config)
        printr("Hydra context", self.hydra_context)

        self.launcher.global_overrides = arguments
        # if len(arguments) == 0:
        #     log.info("Sweep doesn't override default config.")
        # else:
        #     log.info(f"Sweep overrides: {' '.join(arguments)}")

        configspace = search_space_to_config_space(search_space=self.search_space)
        dimensions = len(configspace.get_hyperparameters())
        modehb = HydraMODEHB(
            global_config=self.config,
            global_overrides=arguments,
            launcher=self.launcher,
            budget_variable=self.budget_variable,
            n_jobs=self.n_jobs,
            base_dir=self.sweep_dir,
            cs=configspace,
            objective_function=self.task_function,
            dimensions=dimensions,
            **self.modehb_kwargs,
        )

        # if self.resume is not None:
        #     dehb.load_dehb(self.resume)

        incumbent_list = modehb.run(
            total_cost=self.cost,
            fevals=self.fevals,
            brackets=self.brackets,
            total_wallclock_cost=self.time_cost,
            single_node_with_gpus=False,
            verbose=True,
            debug=False,
            save_intermediate=True,
            save_history=True
        )

        pareto_fit = modehb.pareto_fit
        final_configs = [copy.deepcopy(self.config) for _ in range(len(incumbent_list))]

        for i, final_config in enumerate(final_configs):
            with open_dict(final_config):
                del final_config["hydra"]
            log.info(f"# {i} final configuration keys: {final_config.keys()}")

        # for a in arguments:
        #     # This code has not been adapted for modehb. Assumption is that we do not work with workters anymore
        #     # n, v = a.split("=")
        #     # key_parts = n.split(".")
        #     # reduce(operator.getitem, key_parts[:-1], final_config)[key_parts[-1]] = v
        #     raise Exception(
        #         f"Not Implemented Error - arguments array not expected in local or slurm implmentation of modehb")
        
        for i, incumbent in enumerate(incumbent_list):
            for k, v in incumbent.items():
                key_parts = k.split(".")
                reduce(operator.getitem, key_parts[:-1], final_configs[i])[key_parts[-1]] = v
            with open(os.path.join(modehb.output_path, f"final_config_{i+1}.yml"), "w+") as fp:
                OmegaConf.save(config=final_configs[i], f=fp)

        # Save all candidate configurations generated
        os.makedirs(os.path.join(modehb.output_path, "candidate_configs"), exist_ok=True)
        for i, candidate in enumerate(modehb.saved_candidates):
            with open(os.path.join(modehb.output_path, "candidate_configs", f"candidates_detailed_config_{i+1}.txt"), "w+") as fp:
                fp.write(str(candidate))
        for i, candidate in enumerate(modehb.saved_candidates):
            # if 'cs_config' in candidate:
            #     del candidate['cs_config']
            with open(str(os.path.join(modehb.output_path, "candidate_configs", f"candidates_config_{i+1}.pkl")), "wb") as fp:
                pickle.dump(candidate, fp)

        # Save all configurations with full budget training seperately
        highest_budget = max(config["budget"] for config in modehb.saved_candidates)
        full_trained_configs = [config for config in modehb.saved_candidates if config["budget"] == highest_budget]
        os.makedirs(os.path.join(modehb.output_path, "full_budget_configs"), exist_ok=True)
        for i, candidate in enumerate(full_trained_configs):
            with open(str(os.path.join(modehb.output_path, "full_budget_configs", f"full_budget_config_{i+1}.pkl")), "wb") as fp:
                pickle.dump(candidate, fp)

        # Save all pareto optimal solutions
        with open(os.path.join(modehb.output_path, f"final_pareto_fits.pkl"), "wb") as fp:
            pickle.dump(pareto_fit, fp,)
        with open(os.path.join(modehb.output_path, f"final_pareto_configs.pkl"), "wb") as fp:
            pickle.dump(final_configs, fp)

        return incumbent_list