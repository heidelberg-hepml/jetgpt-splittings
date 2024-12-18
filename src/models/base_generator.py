import torch
from torch import nn
import numpy as np

import os, time
from hydra.utils import instantiate

from src.utils.mlflow import log_mlflow
from src.utils.logger import LOGGER

from src.utils.preprocessing_gen import preprocess, undo_preprocess
from src.utils.data import JointGeneratorDataset, JointGeneratorDataLoader

import matplotlib.pyplot as plt  # debugging


class BaseGenerator(nn.Module):
    def __init__(self, cfg, warm_start, device, dtype):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.prep_params = None

        # init model
        self.init_model()
        num_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        if self.cfg.use_mlflow:
            log_mlflow("num_parameters", float(num_parameters), step=0)
        LOGGER.info(
            f"Instantiated generator {type(self).__name__} with {num_parameters} learnable parameters"
        )

        # load existing model if specified
        self.warm_start = warm_start
        if self.warm_start:
            model_path = os.path.join(
                self.cfg.run_dir,
                "models",
                f"{self.cfg.warm_start_stage}_run{self.cfg.warm_start_idx}.pt",
            )
            try:
                state_dict = torch.load(model_path, map_location="cpu")["model"]
            except FileNotFoundError:
                raise ValueError(f"Cannot load model from {model_path}")
            LOGGER.info(f"Loading model from {model_path}")
            self.model.load_state_dict(state_dict)

        self.model.to(self.device, dtype=self.dtype)

    def init_dataloaders(
        self, raw, channels, log_probs=None, weights=None, store_truth=False
    ):
        # use this function to create dataloaders for both usual generator training and discformer training

        # organize data
        data_raw = []
        data_prepd = []
        if self.prep_params is None:
            self.prep_params = []
        if log_probs is not None or weights is not None:
            log_probs_split = []
            weights_split = []
        for ijet, channels_single in enumerate(channels):
            # prepare dataset splits
            assert len(self.cfg.data.train_test_val) == 3
            assert (np.cumsum(self.cfg.data.train_test_val) <= 1).all()
            splits = np.round(
                np.cumsum(self.cfg.data.train_test_val) * len(raw[ijet])
            ).astype("int")

            # split and save raw data
            raw_trn, raw_tst, raw_val, _ = np.split(raw[ijet], splits, axis=0)
            raw_splits = {"trn": raw_trn, "tst": raw_tst, "val": raw_val}
            data_raw.append(raw_splits)

            # preprocessing
            cfg = self.cfg.copy()
            cfg.data.channels = channels_single
            prep_params = (
                None if len(self.prep_params) <= ijet else self.prep_params[ijet]
            )
            data_prepd_single, prep_params = preprocess(raw[ijet], cfg, prep_params)
            if len(self.prep_params) <= ijet:
                self.prep_params.append(prep_params)

            # split and save prepd data
            prepd_trn, prepd_tst, prepd_val, _ = np.split(
                data_prepd_single, splits, axis=0
            )
            data_prepd.append({"trn": prepd_trn, "tst": prepd_tst, "val": prepd_val})
            if log_probs is not None or weights is not None:
                log_probs_trn, log_probs_tst, log_probs_val, _ = np.split(
                    log_probs[ijet], splits, axis=0
                )
                weights_trn, weights_tst, weights_val, _ = np.split(
                    weights[ijet], splits, axis=0
                )
                log_probs_split.append(
                    {"trn": log_probs_trn, "tst": log_probs_tst, "val": log_probs_val}
                )
                weights_split.append(
                    {"trn": weights_trn, "tst": weights_tst, "val": weights_val}
                )
        if store_truth:
            self.truth_raw = data_raw
            self.truth_prepd = data_prepd

        # create datasets + dataloaders
        loaders = []
        batchsizes = [
            self.cfg.training_gen.batchsize,
            self.cfg.training_gen.batchsize_eval,
            self.cfg.training_gen.batchsize_eval,
        ]
        for label, shuffle, batchsize in zip(
            ["trn", "tst", "val"], [True, False, False], batchsizes
        ):
            if log_probs is None and weights is None:
                args = [[data[label] for data in data_prepd]]
            else:
                args = [
                    [data[label] for data in data_prepd],
                    [log_prob[label] for log_prob in log_probs_split],
                    [weight[label] for weight in weights_split],
                ]
            dataset = JointGeneratorDataset(*args)
            loader = JointGeneratorDataLoader(
                dataset,
                batch_size=batchsize,
                shuffle=shuffle,
                drop_last=shuffle,
            )
            loaders.append(loader)
        self.trnloader, self.tstloader, self.valloader = loaders

    def init_optimizer(self):
        if self.cfg.training_gen.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.cfg.training_gen.lr,
                betas=self.cfg.training_gen.betas,
                eps=self.cfg.training_gen.eps,
                weight_decay=self.cfg.training_gen.weight_decay
                if hasattr(self.cfg.training_gen, "weight_decay")
                else 0,
            )
        else:
            raise ValueError(
                f"Optimizer {self.cfg.training_gen.optimizer} not implemented"
            )
        LOGGER.debug(
            f"Using optimizer {self.cfg.training_gen.optimizer} with lr={self.cfg.training_gen.lr} for generator"
        )

        if self.warm_start:
            model_path = os.path.join(
                self.cfg.run_dir, "models", f"gen_run{self.cfg.warm_start_idx}.pt"
            )
            try:
                state_dict = torch.load(model_path, map_location="cpu")["optimizer"]
            except FileNotFoundError:
                raise ValueError(f"Cannot load optimizer from {model_path}")
            LOGGER.info(f"Loading optimizer from {model_path}")
            self.optimizer.load_state_dict(state_dict)

    def init_scheduler(self, epochs=None):
        if self.cfg.training_gen.scheduler is None:
            self.scheduler = None  # constant lr
        elif self.cfg.training_gen.scheduler == "OneCycleLR":
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                self.cfg.training_gen.lr * 10,
                epochs=epochs,
                steps_per_epoch=len(self.trnloader),
            )
        elif self.cfg.training_gen.scheduler == "CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=epochs * len(self.trnloader),
                eta_min=self.cfg.training_gen.lr_eta_min,
            )
        else:
            raise ValueError(
                f"Learning rate scheduler {self.cfg.training_gen.scheduler} not implemented"
            )

        LOGGER.debug(
            f"Using learning rate scheduler {self.cfg.training_gen.scheduler} for generator"
        )

    def sample(self, num_samples, contexts):
        prepd, raw = [], []
        self.model.eval()
        for ijet in range(len(self.cfg.data.n_jets_list)):
            t0 = time.time()

            LOGGER.info(
                f"Starting to generate {num_samples} {self.cfg.data.n_jets_list[ijet]}j events"
            )
            samples_prepd = self.sample_single(num_samples, contexts[ijet])

            cfg = self.cfg.copy()
            cfg.data.channels = contexts[ijet]["channels"]
            samples_raw = undo_preprocess(samples_prepd, cfg, self.prep_params[ijet])
            num_particles = (
                self.cfg.data.n_hard_particles + self.cfg.data.n_jets_list[ijet]
            )
            samples_raw = samples_raw[:, :num_particles, :]  # remove zero-padding
            prepd.append(samples_prepd)
            raw.append(samples_raw)

            dt = time.time() - t0
            LOGGER.info(
                f"Created {self.cfg.data.n_jets_list[ijet]}j events with shape {samples_raw.shape} in {dt:.2f}s = {dt/60:.2f}min"
            )
        return raw, prepd

    def create_evaluation_dataloader(self, data_raw, contexts):
        loaders = []
        for ijet in range(len(self.cfg.data.n_jets_list)):
            cfg = self.cfg.copy()
            cfg.data.channels = contexts[ijet]["channels"]
            prepd, _ = preprocess(data_raw[ijet], cfg, self.prep_params[ijet])
            prepd = torch.tensor(prepd)
            dataset = torch.utils.data.TensorDataset(prepd)
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=self.cfg.training_gen.batchsize_eval, shuffle=False
            )
            loaders.append(loader)
        return loaders

    def evaluate_log_prob(self, samples, contexts):
        LOGGER.info(
            f"Starting to evaluate likelihoods for samples of shapes {[sample.shape for sample in samples]}"
        )
        loaders = self.create_evaluation_dataloader(samples, contexts)

        log_probs = []
        self.model.eval()
        with torch.no_grad():
            for ijet, loader in enumerate(loaders):
                log_probs_fixedmult = []
                for (data,) in loader:
                    log_prob = self.log_prob(data, contexts[ijet])
                    log_probs_fixedmult.append(log_prob.cpu())
                log_probs.append(np.concatenate(log_probs_fixedmult, axis=0))
        return log_probs

    def save_model(self, path, filename):
        model_path = os.path.join(path, filename)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            model_path,
        )

    def init_model(self):
        raise NotImplementedError

    def log_prob(self, x, context):
        raise NotImplementedError
