import hydra
from src.eventgen.processes import (
    ttbarExperiment,
    zmumuExperiment,
)

@hydra.main(config_path="config_quick", config_name="zmumu", version_base=None)
def main(cfg):
    if cfg.exp_type == "ttbar":
        exp = ttbarExperiment(cfg)
    elif cfg.exp_type == "zmumu":
        exp = zmumuExperiment(cfg)
    else:
        raise ValueError(f"exp_type {cfg.exp_type} not implemented")

    exp()


if __name__ == "__main__":
    main()
