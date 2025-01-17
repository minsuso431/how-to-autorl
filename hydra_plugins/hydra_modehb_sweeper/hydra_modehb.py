"""
This is an adaptation on Hydra of original MODEHB code written by Awad et al. and Hydra DEHB code from Eimer et al.
"""
import json
import logging
import os
import pickle
import sys
import time
from copy import deepcopy
import numpy as np
import wandb
from deepcave import Objective, Recorder
from modehb.optimizers import modehb
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

log = logging.getLogger(__name__)

# Most of this is very similar to the original MODEHB
class HydraMODEHB(modehb.MODEHB):
    def __init__(
        self,
        global_config,
        global_overrides,
        launcher,
        budget_variable,
        n_jobs,
        base_dir,
        # Necessary MODEHB configs
        cs,
        objective_function,
        dimensions=None,
        mutation_factor=0.5,
        crossover_prob=0.5,
        mutation_strategy="rand1_bin",
        mo_strategy='NSGA-II',
        min_budget=None,
        max_budget=None,
        eta=3,
        min_clip=None,
        max_clip=None,
        configspace=True,
        boundary_fix_type="random",
        max_age=np.inf,
        num_objectives=2,
        log_interval=100,
        # hydra and additional configs
        seeds=False,
        slurm=False,
        slurm_timeout=10,
        wandb_project=False,
        wandb_entity=False,
        wandb_tags=["modehb"],
        deepcave=False,
        maximize=False,
        # Stability Objective
        stability_objective="std_returns",
        time_limit=None,
        **kwargs,
    ):
        output_path = to_absolute_path(base_dir)
        kwargs["output_path"] = output_path
        assert min_budget is not None, "Please set a minimum budget per run"
        assert max_budget is not None, "Please set a maximum budget per run"
        super().__init__(
            cs=cs,
            objective_function=objective_function,
            dimensions=dimensions,
            mutation_factor=mutation_factor,
            crossover_prob=crossover_prob,
            mutation_strategy=mutation_strategy,
            mo_strategy=mo_strategy,
            min_budget=min_budget,
            max_budget=max_budget,
            eta=eta,
            min_clip=min_clip,
            max_clip=max_clip,
            configspace=configspace,
            boundary_fix_type=boundary_fix_type,
            n_workers=1,
            max_age=max_age,
            num_objectives=num_objectives,
            log_interval=log_interval,
            **kwargs,
        )
        self.global_overrides = global_overrides
        self.launcher = launcher
        self.budget_variable = budget_variable
        self.n_jobs = n_jobs
        self.seeds = seeds
        self.slurm = slurm
        self.slurm_timeout = slurm_timeout
        if seeds and len(self.global_overrides) > 0:
            for i in range(len(self.global_overrides)):
                if self.global_overrides[i].split("=")[0] == "seed":
                    self.global_overrides = self.global_overrides[:i] + self.global_overrides[i + 1 :]
                    break
        self.maximize = maximize

        self.deepcave = deepcave
        if self.deepcave:
            log.warning("Deepcave will only record the first objective value!")
            reward_objective = Objective("reward", optimize="lower")
            deepcave_path = os.path.join(self.output_path, "deepcave_logs")
            self.deepcave_recorder = Recorder(self.cs, objectives=[reward_objective], save_path=deepcave_path)

        self.wandb_project = wandb_project
        if self.wandb_project:
            wandb_config = OmegaConf.to_container(global_config, resolve=False, throw_on_missing=False)
            assert wandb_entity, "Please provide an entity to log to W&B."
            wandb.init(
                project=self.wandb_project,
                entity=wandb_entity,
                tags=wandb_tags,
                config=wandb_config,
            )

        self.trial_id = 0
        self.cumulated_costs = 0
        self.opt_time = 0
        self.saved_candidates = []

        self.stability_objective = stability_objective
        self.time_limit = time_limit

    def save_results(self):
        with open(os.path.join(self.output_path, "pareto_fit_{}.txt".format(time.time())), 'w') as f:
            np.savetxt(f, self.pareto_fit)
        with open(os.path.join(self.output_path, "pareto_pop_{}.txt".format(time.time())), 'w') as f:
            pareto_configs = [self.vector_to_configspace(inc_config) for inc_config in self.pareto_pop]
            for item in pareto_configs:
                f.write(str(item))

    def _save_incumbent(self, name=None):
        if name is None:
            name = "incumbents.json"

        res = dict()
        if self.configspace:
            pareto_configs = [self.vector_to_configspace(inc_config).get_dictionary() for inc_config in self.pareto_pop]
            res["pareto_configs"] = pareto_configs
        else:
            res["pareto_configs"] = [inc_config.tolist() for inc_config in self.pareto_pop]
        res["pareto_scores"] = self.pareto_fit.tolist()
        # res["info"] = self.inc_info
        res["total_cumulated_costs"] = float(self.cumulated_costs)
        res["total_wallclock_time"] = self.start - time.time()
        res["total_optimization_time"] = self.opt_time
        with open(os.path.join(self.output_path, name), "a+") as f:
            json.dump(res, f)
            f.write("\n")

    def checkpoint_modehb(self):
        d = deepcopy(self.__getstate__())
        del d["logger"]
        del d["launcher"]
        del d["f"]
        del d["client"]
        for k in d["de"].keys():
            d["de"][k].f = None
        if "f" in d["de_params"].keys():
            del d["de_params"]["f"]
        try:
            with open(os.path.join(self.output_path, "modehb_state.pkl"), "wb") as f:
                pickle.dump(d, f)
        except Exception as e:
            log.warning("Checkpointing failed: {}".format(repr(e)))

        if self.wandb_project:
            stats = {}
            stats["optimization_time"] = time.time() - self.start
            stats["pareto_inc_scores"] = self.pareto_fit
            stats["cumulated_costs"] = self.cumulated_costs
            stats["pareto_inc_configs"] = self.pareto_pop
            wandb.log(stats)

    def load_modehb(self, path):
        with open(path, "rb") as f:
            past_state = pickle.load(f)
        self.__dict__.update(**past_state)
        func = list(self.de.values())[0].f
        for k in self.de:
            self.de[k].f = func

    # This is very similar to the original MODEHB, only that we replace the submit function
    # with the launcher and run one bracket at a time
    def run(self, total_cost=None, fevals=None, brackets=None, total_wallclock_cost=None,
            single_node_with_gpus=False, verbose=True, debug=False,
            save_intermediate=True, save_history=True, name=None,  **kwargs):
        """ Main interface to run optimization by MODEHB. This is exactly same as the DEHB version

        This function waits on workers and if a worker is free, asks for a configuration and a
        budget to evaluate on and submits it to the worker. In each loop, it checks if a job
        is complete, fetches the results, carries the necessary processing of it asynchronously
        to the worker computations.

        The duration of the DEHB run can be controlled by specifying one of 3 parameters. If more
        than one are specified, DEHB selects only one in the priority order (high to low):
        1) Number of function evaluations (fevals)
        2) Number of Successive Halving brackets run under Hyperband (brackets)
        3) Total computational cost (in seconds) aggregated by all function evaluations (total_wallclock_cost)
        4) Total computational cost (in seconds) returned by the objective function. Might be simulated costs. (total_cost)
        """
        self._init_subpop()

        num_brackets = brackets
        self.start = time.time()
        if verbose:
                print("\nLogging at {} for optimization starting at {}\n".format(
                    os.path.join(os.getcwd(), self.log_filename),
                    time.strftime("%x %X %Z", time.localtime(self.start))
                ))
        while True:
            if self._is_run_budget_exhausted(fevals, brackets, total_cost, total_wallclock_cost):
                break
            bracket = None
            opt_time_start = time.time()
            overrides = []
            bracket_jobs = []
            # Fill up job queue as far as possible
            jobs_left = self.n_jobs
            reset_job_count = False
            if fevals:
                jobs_left = min(jobs_left, fevals - len(self.traj))
            elif total_cost:
                reset_job_count
            while len(overrides) < jobs_left:
                num_queued = len(overrides)
                if len(self.active_brackets) == 0 or np.all(
                    [bracket.is_bracket_done() for bracket in self.active_brackets]
                ):
                    # start new bracket when no pending jobs from existing brackets or empty bracket list
                    bracket = self._start_new_bracket()
                    # self.logger.debug(f"New starting Bracket:/n{bracket}")
                else:
                    for _bracket in self.active_brackets:
                        # check if _bracket is not waiting for previous rung results of same bracket
                        # _bracket is not waiting on the last rung results
                        # these 2 checks allow DEHB to have a "synchronous" Successive Halving
                        if not _bracket.previous_rung_waits() and _bracket.is_pending():
                            # bracket eligible for job scheduling
                            bracket = _bracket
                            break
                    if bracket is None:
                        # start new bracket when existing list has all waiting brackets
                        bracket = self._start_new_bracket()
                # budget that the SH bracket allots
                new_budget = bracket.get_next_job_budget()
                if new_budget is None:
                    break
                budget = new_budget
                if reset_job_count:
                    budget_left = (total_cost - self.cumulated_costs) // budget
                    jobs_left = min(jobs_left, budget_left)
                    if jobs_left == 0:
                        break
                
                # Add more jobs if current bracket isn't waiting on results and also has jobs left
                space_in_bracket = (
                    bracket.sh_bracket[budget] > 0 and not bracket.previous_rung_waits() and bracket.is_pending()
                )
                while space_in_bracket and len(overrides) < jobs_left:
                    vconfig, parent_id = self._acquire_config(bracket, budget)
                    config = self.vector_to_configspace(vconfig)
                    global_parent_id = self.de[budget].global_parent_id[parent_id]
                    bracket_jobs.append(
                        {
                            "cs_config": config,
                            "config": vconfig,
                            "budget": budget,
                            "parent_id": parent_id,
                            "global_parent_id": global_parent_id,
                            "bracket_id": bracket.bracket_id,
                            "info": {},
                        }
                    )
                    # log.info(f'Got config {config}, parent id {parent_id} and budget {int(budget)+1}')
                    names = list(config.keys()) + [self.budget_variable]
                    values = list(config.values()) + [int(budget) + 1]
                    if self.slurm:
                        names += ["hydra.launcher.timeout_min"]
                        optimized_timeout = (
                            self.slurm_timeout * 1 / (self.max_budget // budget) + 0.1 * self.slurm_timeout
                        )
                        values += [int(optimized_timeout)]
                    if self.seeds:
                        for s in self.seeds:
                            job_overrides = tuple(self.global_overrides) + tuple(
                                f"{name}={val}" for name, val in zip(names + ["seed"], values + [s])
                            )
                            overrides.append(job_overrides)
                    else:
                        job_overrides = tuple(self.global_overrides) + tuple(
                            f"{name}={val}" for name, val in zip(names, values)
                        )
                        overrides.append(job_overrides)
                    self.cumulated_costs += budget
                    bracket.register_job(budget)
                    space_in_bracket = bracket.sh_bracket[budget] > 0
                if len(overrides) == num_queued:
                    break
            
            # Run jobs
            # if verbose:
            #     self.logger.debug(f"Current bracket before launch:\n{str(bracket)}")

            # Saved overrides (store all sampled config cadidates) and bracket_jobs (job_info for each jobs)
            while len(overrides) > 0:
                index = min(self.n_jobs, len(overrides))
                # Make sure that all seeds of a config are launched at the same time so we can aggregate
                if self.seeds:
                    index = (index // len(self.seeds)) * len(self.seeds) // len(self.seeds)
                    launching_jobs = bracket_jobs[:index]
                    bracket_jobs = bracket_jobs[index:]
                    to_launch = overrides[: index * len(self.seeds)]
                    overrides = overrides[index * len(self.seeds) :]
                else:
                    launching_jobs = bracket_jobs[:index]
                    bracket_jobs = bracket_jobs[index:]
                    to_launch = overrides[:index]
                    overrides = overrides[index:]
                
                if len(to_launch) == 0:
                    break
                if verbose:
                    if self.seeds:
                        bgt = [job_info['budget'] for job_info in launching_jobs for _ in range(len(self.seeds))]
                        bk_id = [job_info['bracket_id'] for job_info in launching_jobs for _ in range(len(self.seeds))]
                        log.info(
                            f"Begin Launching {len(to_launch)} configurations/jobs with {bgt} budgets in {bk_id} brackets with seeds {self.seeds}"
                        )
                    else:
                        bgt = [job_info['budget'] for job_info in launching_jobs]
                        bk_id = [job_info['bracket_id'] for job_info in launching_jobs]
                        log.info(
                            f"Begin Launching {len(to_launch)} configurations/jobs with {bgt} budgets in {bk_id} brackets"
                        )
                self.opt_time += time.time() - opt_time_start
                res = self.launcher.launch(to_launch, initial_job_idx=self.trial_id)
                self.trial_id += len(to_launch)

                for i in range(len(launching_jobs)):
                    launching_jobs[i]["cost"] = launching_jobs[i]["budget"]
                done = False
                while not done:
                    for i in range(len(to_launch)):
                        if res[i].status.name == "COMPLETED":
                            res[i].return_value
                            done = True
                        else:
                            done = False

                if self.seeds:
                    for i in range(0, len(launching_jobs)):
                        # Resulted means and stds of each training seeds
                        means = [res[i * len(self.seeds) + j].return_value[0] for j in range(len(self.seeds))]
                        stds = [res[i * len(self.seeds) + j].return_value[1] for j in range(len(self.seeds))]

                        ## You can add your own second objective to optimize. This version will always consider the mean performance as the first objective ##
                        if self.stability_objective == "mean_stds":
                            launching_jobs[i]["fitness"] = [
                                np.mean(means),
                                np.mean(stds)
                            ]
                        elif self.stability_objective == "std_returns":
                            launching_jobs[i]["fitness"] = [
                                np.mean(means),
                                np.std(means)
                            ]
                        elif self.stability_objective == "max_min_return":
                            launching_jobs[i]["fitness"] = [
                                np.mean(means),
                                max(means) # Maximize the min(mean) -> Minimize the max(-mean)
                            ]
                        elif self.stability_objective == "max_min_return_quantile":
                            launching_jobs[i]["fitness"] = [
                                np.mean(means),
                                np.quantile(means, 0.75)
                            ]
                        else:
                            raise ValueError(f'Unknown stability objective, {self.stability_objective}')

                        if self.maximize:
                           launching_jobs[i]["fitness"] = -launching_jobs[i]["fitness"]

                        self.futures.append(launching_jobs[i])
                        log.info(
                            f'Finished job across {len(self.seeds)} seeds with performance value {round(launching_jobs[i]["fitness"][0], 2)} and with stability value {round(launching_jobs[i]["fitness"][1], 2)} and mean cost {round(launching_jobs[i]["cost"], 2)}. Chosen stability metric was {self.stability_objective}'
                        )
                else:
                    for i in range(len(to_launch)):
                        launching_jobs[i]["fitness"] = res[i].return_value
                        if self.maximize:
                            launching_jobs[i]["fitness"] = -launching_jobs[i]["fitness"]
                        self.futures.append(launching_jobs[i])

                        # if verbose:
                        #     # self._verbosity_runtime(fevals, brackets, total_cost, total_wallclock_cost)
                        #     self.logger.debug(
                        #         "Evaluating a configuration with budget {} under "
                        #         "bracket ID {}".format(launching_jobs[i]["budget"], launching_jobs[i]['bracket_id'])
                        #     )
                        log.info(
                            f'Finished job in bracket {launching_jobs[i]["bracket_id"]} with budget {launching_jobs[i]["budget"]}, fitness {launching_jobs[i]["fitness"]} and cost {round(launching_jobs[i]["cost"], 2)}'
                        )

                if self.deepcave:
                    for job in launching_jobs:
                        self.deepcave_recorder.start(config=job["cs_config"], budget=float(job["budget"]))
                        self.deepcave_recorder.end(costs=float(job["fitness"][0]), config=job["cs_config"], budget=float(job["budget"]))

                ## Save all generated candidates and its resulted information
                for launching_job in launching_jobs:
                    candicate = {"cs_config": launching_job["cs_config"],
                                 "budget": launching_job["budget"],
                                 "parent_id": launching_job["parent_id"],
                                 "global_parent_id": launching_job["global_parent_id"],
                                 "bracket_id": launching_job["bracket_id"],
                                 "fitness": launching_job["fitness"],
                                 "seed": self.seeds}
                    self.saved_candidates.append(candicate)

                opt_time_start = time.time()
                """
                - update bracket information
                - carry out DE selection according (check_fitness) according to NDS sorting
                - update pareto front
                - book-keeping
                """
                self._fetch_results_from_workers()
                if verbose:
                    # self._verbosity_debug()
                    self.logger.debug(f"Current bracket after launch and fetch:\n{str(bracket)}")

                if verbose:
                    log.info(f"Scores of best Pareto Front seen so far: {self.pareto_fit}")
                if self.inc_config is not None:
                    self._save_incumbent(name)
                if save_history and self.history is not None:
                    self._save_history(name)
                self.checkpoint_modehb()
                self.clean_inactive_brackets()
                self.opt_time += time.time() - opt_time_start
            log.info("Bracket finished")
            # end of while

        if verbose and len(self.futures) > 0:
            log.info("MODEHB optimisation over! Waiting to collect results from workers running...")
        while len(self.futures) > 0:
            self._fetch_results_from_workers()
            if save_intermediate and self.inc_config is not None:
                self._save_incumbent(name)
            if save_history and self.history is not None:
                self._save_history(name)
            time.sleep(0.05)  # waiting 50ms

        if verbose:
            time_taken = time.time() - self.start
            log.info(
                "End of optimisation! Total duration: {}s; Optimization overhead: {}s; Total fevals: {}\n".format(
                    np.round(time_taken, decimals=2), np.round(self.opt_time, decimals=2), len(self.traj)
                )
            )
            log.info(f"Pareto front scores: {self.pareto_fit}")
            log.info("Pareto front configs: ")
            if self.configspace:
                log.info(self.pareto_pop)
                configs = [self.vector_to_configspace(inc_config) for inc_config in self.pareto_pop]
                for i, config in enumerate(configs):
                    log.info(f"Config {i} from {len(self.pareto_pop)}")
                    for k, v in config.get_dictionary().items():
                        log.info("{}: {}".format(k, v))
            else:
                log.info(self.pareto_pop)
        self.save_results()
        self._save_incumbent(name)
        self._save_history(name)
        return configs  # np.array(self.traj), np.array(self.runtime), np.array(self.history, dtype=object)