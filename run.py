import hydra
from src.experiments.zmumu import zmumuExperiment
from src.experiments.ttbar import ttbarExperiment

@hydra.main(config_path="config", config_name="zmumu", version_base=None)
def main(cfg):
    if cfg.exp_type == "zmumu":
        exp = zmumuExperiment(cfg)
    elif cfg.exp_type == "ttbar":
        exp = ttbarExperiment(cfg)
    
    exp()

if __name__ == "__main__":
    main()
