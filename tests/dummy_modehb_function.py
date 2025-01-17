import hydra


@hydra.main(config_path=".", config_name="dummy_modehb_config", version_base="1.3")
def run_dummy(cfg):
    return int(cfg.x > 1), int(cfg.x < 1.4)


if __name__ == "__main__":
    run_dummy()