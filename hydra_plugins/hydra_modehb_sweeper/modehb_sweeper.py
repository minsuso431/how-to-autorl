"""
This is an adaptation on MODEHB of original file written by Eimer et al.: https://github.com/facebookresearch/how-to-autorl/tree/main/hydra_plugins/hydra_dehb_sweeper 
"""
from typing import Any, Dict, List

from hydra.plugins.sweeper import Sweeper
from hydra.types import HydraContext, TaskFunction
from omegaconf import DictConfig

class MODEHBSweeper(Sweeper):
    """Class to interface with the MODEHB run"""
    def __init__(self, *args: Any, **kwargs: Dict[Any, Any]) -> None:
        from .modehb_sweeper_backend import MODEHBSweeperBackend
        self.sweeper = MODEHBSweeperBackend(*args, **kwargs)

    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        self.sweeper.setup(hydra_context=hydra_context, task_function=task_function, config=config)

    def sweep(self, arguments: List[str]) -> None:
        return self.sweeper.sweep(arguments)