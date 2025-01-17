import hydra
import logging
from typing import List

import gymnasium as gym
import hydra
import stable_baselines3
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


log = logging.getLogger(__name__)

def mo_train_sb3(cfg, log) -> List:
    log.info(OmegaConf.to_yaml(cfg))

    log.info(f"Training {cfg.algorithm.agent_class} Agent on {cfg.env_name} for {cfg.algorithm.total_timesteps} steps")
    env = gym.make(cfg.env_name)
    if cfg.reward_curves:
        env = Monitor(env, ".")

    agent_class = getattr(stable_baselines3, cfg.algorithm.agent_class)

    if cfg.load:
        model = agent_class.load(cfg.load, env=env, **cfg.algorithm.model_kwargs)
    else:
        model = agent_class(cfg.algorithm.policy_model, env, **cfg.algorithm.model_kwargs)

    model.learn(total_timesteps=cfg.algorithm.total_timesteps, reset_num_timesteps=False)

    if cfg.save:
        model.save(cfg.save)

    mean_reward, std_reward = evaluate_policy(
        model,
        model.get_env(),
        n_eval_episodes=cfg.algorithm.n_eval_episodes,
    )
    log.info(
        f"Mean evaluation reward at the end of training across {cfg.algorithm.n_eval_episodes} episodes was {mean_reward}"
    )

    return [-mean_reward, std_reward]


@hydra.main(version_base="1.3", config_path="configs", config_name="ppo_pendulum_modehb")
def run_ppo_modehb(cfg):
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)
    return mo_train_sb3(cfg, log)


if __name__ == "__main__":
    run_ppo_modehb()