import numpy as np
import torch

import os, time
from datetime import datetime
import zipfile
import logging
from pathlib import Path
from omegaconf import OmegaConf, open_dict, errors
import mlflow

from src.utils.misc import get_device, flatten_dict
import src.utils.logger
from src.utils.logger import LOGGER, MEMORY_HANDLER, FORMATTER
from src.utils.mlflow import log_mlflow


class BaseExperiment:
    """
    Boilerplate code that could be reused for anything
    - create experiment folder
    - connect to mlflow
    - logging
    - backend (device, dtype, attention type)
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self):
        # pass all exceptions to the logger
        try:
            self.run_mlflow()
        except errors.ConfigAttributeError:
            LOGGER.exception(
                "Tried to access key that is not specified in the config files"
            )
        except:
            LOGGER.exception("Exiting with error")

        # print buffered logger messages if failed
        if not src.utils.logger.LOGGING_INITIALIZED:
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging.DEBUG)
            MEMORY_HANDLER.setTarget(stream_handler)
            MEMORY_HANDLER.close()

    def run_mlflow(self):
        experiment_id, run_name = self._init()
        LOGGER.info(
            f"### Starting experiment {self.cfg.exp_name}/{run_name} (id={experiment_id}) ###"
        )
        if self.cfg.use_mlflow:
            with mlflow.start_run(
                experiment_id=experiment_id,
                run_name=self.cfg.mlflow.runname
                if self.cfg.mlflow.runname is not None
                else run_name,
            ):
                self.full_run()
        else:
            # dont use mlflow
            self.full_run()

    def _init(self):
        run_name = self._init_experiment()
        self._init_directory()

        if self.cfg.use_mlflow:
            experiment_id = self._init_mlflow()
        else:
            experiment_id = None

        # initialize environment
        self._init_logger()
        self._init_backend()

        return experiment_id, run_name

    def _init_experiment(self):
        self.warm_start = False if self.cfg.warm_start_idx is None else True

        if not self.warm_start:
            now = datetime.now()
            modelname = self.cfg.generator.name
            run_name = now.strftime("%Y%m%d_%H%M%S") + "-" + modelname
            if self.cfg.run_name is None:
                pass
            else:
                run_name += "-" + self.cfg.run_name

            run_dir = os.path.join(
                self.cfg.base_dir, "runs", self.cfg.exp_name, run_name
            )
            run_idx = 0
            LOGGER.info(f"Creating new experiment {self.cfg.exp_name}/{run_name}")

        else:
            run_name = self.cfg.run_name
            run_idx = self.cfg.run_idx + 1
            LOGGER.info(
                f"Warm-starting from existing experiment {self.cfg.exp_name}/{run_name} for run {run_idx}"
            )

        with open_dict(self.cfg):
            self.cfg.run_idx = run_idx
            if not self.warm_start:
                self.cfg.warm_start_idx = 0
                self.cfg.run_name = run_name
                self.cfg.run_dir = run_dir

        # set seed
        if self.cfg.seed is not None:
            LOGGER.info(f"Using seed {self.cfg.seed}")
            torch.random.manual_seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)

        return run_name

    def _init_mlflow(self):
        # mlflow tracking location
        Path(self.cfg.mlflow.db).parent.mkdir(exist_ok=True)
        mlflow.set_tracking_uri(f"sqlite:///{Path(self.cfg.mlflow.db).resolve()}")

        Path(self.cfg.mlflow.artifacts).mkdir(exist_ok=True)
        try:
            # artifacts not supported
            # mlflow call triggers alembic.runtime.migration logger to shout -> shut it down (happy for suggestions on how to do this nicer)
            logging.disable(logging.WARNING)
            experiment_id = mlflow.create_experiment(
                self.cfg.exp_name,
                artifact_location=f"file:{Path(self.cfg.mlflow.artifacts).resolve()}",
            )
            logging.disable(logging.DEBUG)
            LOGGER.info(
                f"Created mlflow experiment {self.cfg.exp_name} with id {experiment_id}"
            )
        except mlflow.exceptions.MlflowException:
            LOGGER.info(f"Using existing mlflow experiment {self.cfg.exp_name}")
            logging.disable(logging.DEBUG)

        experiment = mlflow.set_experiment(self.cfg.exp_name)
        experiment_id = experiment.experiment_id

        LOGGER.info(f"Set experiment {self.cfg.exp_name} with id {experiment_id}")
        return experiment_id

    def _init_directory(self):

        # create experiment directory
        run_dir = Path(self.cfg.run_dir).resolve()
        if run_dir.exists() and not self.warm_start:
            raise ValueError(
                f"Experiment in directory {self.cfg.run_dir} alredy exists. Aborting."
            )
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, "models"), exist_ok=True)

        # save source
        if self.cfg.save_source:
            zip_name = os.path.join(self.cfg.run_dir, "source.zip")
            LOGGER.debug(f"Saving source to {zip_name}")
            zipf = zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED)
            path_gatr = os.path.join(self.cfg.base_dir, "gatr")
            path_experiment = os.path.join(self.cfg.base_dir, "experiments")
            for path in [path_gatr, path_experiment]:
                for root, dirs, files in os.walk(path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, os.path.relpath(file_path, path))
            zipf.close()

    def _init_logger(self):
        # silence other loggers
        # (every app has a logger, eg hydra, torch, mlflow, matplotlib, fontTools...)
        for name, other_logger in logging.root.manager.loggerDict.items():
            if not "lorentz-gatr" in name:
                other_logger.level = logging.WARNING

        if src.utils.logger.LOGGING_INITIALIZED:
            LOGGER.info("Logger already initialized")
            return

        LOGGER.setLevel(logging.DEBUG if self.cfg.debug else logging.INFO)

        # init file_handler
        file_handler = logging.FileHandler(
            Path(self.cfg.run_dir) / f"out_{self.cfg.run_idx}.log"
        )
        file_handler.setFormatter(FORMATTER)
        file_handler.setLevel(logging.DEBUG)
        LOGGER.addHandler(file_handler)

        # init stream_handler
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(LOGGER.level)
        stream_handler.setFormatter(FORMATTER)
        LOGGER.addHandler(stream_handler)

        # flush memory to stream_handler
        # this allows to catch logs that were created before the logger was initialized
        MEMORY_HANDLER.setTarget(
            stream_handler
        )  # can only flush to one handler, choose stream_handler
        MEMORY_HANDLER.close()
        LOGGER.removeHandler(MEMORY_HANDLER)

        # add new handlers to logger
        LOGGER.propagate = False  # avoid duplicate log outputs

        src.utils.logger.LOGGING_INITIALIZED = True
        LOGGER.debug("Logger initialized")

    def _init_backend(self):
        self.device = get_device()
        LOGGER.info(f"Using device {self.device}")

        # could implement more dtypes if needed
        if (
            self.cfg.backend.float16
            and self.device == "cuda"
            and torch.cuda.is_bf16_supported()
        ):
            self.dtype = torch.bfloat16
            LOGGER.debug("Using dtype bfloat16")
        elif self.cfg.backend.float16:
            self.dtype = torch.float16
            LOGGER.debug(
                "Using dtype float16 (bfloat16 is not supported by environment)"
            )
        else:
            self.dtype = torch.float32
            LOGGER.debug("Using dtype float32")

        torch.backends.cuda.enable_flash_sdp(self.cfg.backend.enable_flash_sdp)
        torch.backends.cuda.enable_math_sdp(self.cfg.backend.enable_math_sdp)
        torch.backends.cuda.enable_mem_efficient_sdp(
            self.cfg.backend.enable_mem_efficient_sdp
        )

    def _save_config(self, filename="amplitudes.yaml", to_mlflow=False):
        config_filename = Path(self.cfg.run_dir) / filename
        LOGGER.debug(f"Saving config at {config_filename}")
        with open(config_filename, "w", encoding="utf-8") as file:
            file.write(OmegaConf.to_yaml(self.cfg))

        if to_mlflow and self.cfg.use_mlflow:
            for key, value in flatten_dict(self.cfg).items():
                log_mlflow(key, value, kind="param")
