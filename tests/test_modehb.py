import os
import re
import shutil
import time

import ConfigSpace as CS
from omegaconf import OmegaConf
from hydra.core.plugins import Plugins
from hydra.plugins.sweeper import Sweeper
from pytest import mark

from dummy_modehb_function import run_dummy
from hydra_plugins.hydra_modehb_sweeper.modehb_sweeper import MODEHBSweeper
from hydra_plugins.hydra_modehb_sweeper.hydra_modehb import HydraMODEHB
from utils import run_short_modehb


def test_sweeper_found() -> None:
    assert MODEHBSweeper.__name__ in [x.__name__ for x in Plugins.instance().discover(Sweeper)]


@mark.parametrize("cutoff", [(10), (30), (60)])
def test_termination_time(cutoff):
    outdir = "./tests/modehb/time_termination_test"
    cs = CS.ConfigurationSpace(
        space={
            "a": CS.Float("a", bounds=(0.1, 1.5), log=True),
        }
    )
    sweeper = HydraMODEHB(
        global_overrides=[],
        global_config=None,
        launcher=None,
        budget_variable="b",
        n_jobs=3,
        base_dir=outdir,
        objective_function=run_dummy,
        cs=cs,
        min_budget=2.0,
        max_budget=3.0,
        dimensions=0,
        mo_strategy='NSGA-II',
        stability_objective="std_returns",
        num_objectives=2,
        seeds=False
    )
    run_short_modehb(outdir, termination=[f"+hydra.sweeper.total_wallclock_cost={cutoff}"])
    end_time = time.time()
    buffer = 30
    sweeper.load_modehb(os.path.join(outdir, "modehb_state.pkl"))
    assert end_time - sweeper.start < cutoff + buffer
    shutil.rmtree(outdir, ignore_errors=True)


def test_termination_steps():
    outdir = "./tests/modehb/step_termination_test"
    cs = CS.ConfigurationSpace(
        space={
            "a": CS.Float("a", bounds=(0.1, 1.5), log=True),
        }
    )
    sweeper = HydraMODEHB(
        global_overrides=[],
        global_config=None,
        launcher=None,
        budget_variable="b",
        n_jobs=3,
        base_dir=outdir,
        objective_function=run_dummy,
        cs=cs,
        min_budget=2.0,
        max_budget=3.0,
        dimensions=0,
    )
    run_short_modehb(outdir, termination=["+hydra.sweeper.total_cost=5"])
    sweeper.load_modehb(os.path.join(outdir, "modehb_state.pkl"))

    # In _fetch_results_from_workers() function of the new MODEHB Repository, we see that the cumulated_costs is further added with the seen budgets so far. Therefore we adapt this in this test as well.
    assert sweeper.cumulated_costs - sum(cand["budget"] for cand in sweeper.saved_candidates) < 6
    shutil.rmtree(outdir, ignore_errors=True)


def test_termination_brackets():
    outdir = "./tests/modehb/bracket_termination_test"
    cs = CS.ConfigurationSpace(
        space={
            "a": CS.Float("a", bounds=(0.1, 1.5), log=True),
        }
    )
    sweeper = HydraMODEHB(
        global_overrides=[],
        global_config=None,
        launcher=None,
        budget_variable="b",
        n_jobs=3,
        base_dir=outdir,
        objective_function=run_dummy,
        cs=cs,
        min_budget=2.0,
        max_budget=3.0,
        dimensions=0,
    )
    run_short_modehb(outdir, termination=["+hydra.sweeper.total_brackets=1"])
    sweeper.load_modehb(os.path.join(outdir, "modehb_state.pkl"))
    assert len(sweeper.active_brackets) == 1
    assert all([bracket.is_bracket_done() for bracket in sweeper.active_brackets])
    shutil.rmtree(outdir, ignore_errors=True)


def test_termination_fevals():
    raise NotImplementedError("The 'traj' attribute is no longer supported in the MODEHB implementation. It is not saved in the '_update_trackers()' function.")
