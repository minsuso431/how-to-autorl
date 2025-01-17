"""
This code is from Eimer et al.: https://github.com/facebookresearch/how-to-autorl/tree/main/hydra_plugins/hydra_dehb_sweeper/config.py with a different target directory
"""

from typing import Any, Dict, Optional

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


@dataclass
class MODEHBSweeperConfig:
    _target_: str = "hydra_plugins.hydra_modehb_sweeper.modehb_sweeper.MODEHBSweeper"
    search_space: Dict[str, Any] = field(default_factory=dict)
    modehb_kwargs: Optional[Dict] = field(default_factory=dict)
    budget_variable: Optional[str] = None
    resume: Optional[str] = None
    n_jobs: Optional[int] = 8
    slurm: Optional[bool] = False
    slurm_timeout: Optional[int] = 10
    total_function_evaluations: Optional[int] = None
    total_brackets: Optional[int] = None
    total_cost: Optional[int] = None
    total_wallclock_cost: Optional[int] = None


ConfigStore.instance().store(group="hydra/sweeper", name="MODEHB", node=MODEHBSweeperConfig, provider="hydra_modehb_sweeper")