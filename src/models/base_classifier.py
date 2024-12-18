import torch
from torch import nn
import numpy as np
import os, time

from src.utils.mlflow import log_mlflow
from src.utils.logger import LOGGER
from src.utils.data import JointClassifierDataset, JointClassifierDataLoader

import matplotlib.pyplot as plt


class BaseClassifier(nn.Module):
    def __init__(self, label, cfg, device, dtype):
        # we do not implement warm_start for classifiers for now,
        # because they should train fast
        super().__init__()
        self.label = label
        self.label_short = {"Reweighter": "rew", "Discifier": "dfc"}[self.label]
        self.cfg = cfg
        self.device = device
        self.dtype = dtype

        self.prep_params = None
        self.results_eval = {}

    def init_dataloaders(self, data_true, data_fake):
        self.prepd_true, self.prepd_fake = [], []
        if self.prep_params is None:
            self.prep_params = []
        for ijet in range(len(self.cfg.data.n_jets_list)):
            # prepare dataset splits
            # Note: No test set for the classifier
            assert len(self.cfg.training_cls.train_val) == 2
            assert (np.cumsum(self.cfg.training_cls.train_val) <= 1).all()
            splits_true = np.round(
                np.cumsum(self.cfg.training_cls.train_val) * len(data_true[ijet])
            ).astype("int")
            splits_fake = np.round(
                np.cumsum(self.cfg.training_cls.train_val) * len(data_fake[ijet])
            ).astype("int")

            # preprocessing
            prep_params = (
                None if len(self.prep_params) <= ijet else self.prep_params[ijet]
            )
            prepd_true, prep_params = self._preprocess(
                data_true[ijet], prep_params=prep_params
            )
            prepd_fake, _ = self._preprocess(data_fake[ijet], prep_params=prep_params)
            LOGGER.info(
                f"Preprocessed {self.cfg.data.n_jets_list[ijet]}j data has shapes "
                f"true={prepd_true.shape}, fake={prepd_fake.shape}"
            )
            if len(self.prep_params) <= ijet:
                self.prep_params.append(prep_params)

            # split and save prepd data
            prepd_true_trn, prepd_true_val, _ = np.split(
                prepd_true, splits_true, axis=0
            )
            prepd_fake_trn, prepd_fake_val, _ = np.split(
                prepd_fake, splits_fake, axis=0
            )
            prepd_true = {
                "trn": prepd_true_trn,
                "val": prepd_true_val,
            }
            prepd_fake = {
                "trn": prepd_fake_trn,
                "val": prepd_fake_val,
            }
            self.prepd_true.append(prepd_true)
            self.prepd_fake.append(prepd_fake)

        # create datasets + dataloaders
        trnset = JointClassifierDataset(
            [data["trn"] for data in self.prepd_true],
            [data["trn"] for data in self.prepd_fake],
        )
        valset = JointClassifierDataset(
            [data["val"] for data in self.prepd_true],
            [data["val"] for data in self.prepd_fake],
        )
        # note: take half batchsize here, because later we stack true and fake data to double it again
        self.trnloader = JointClassifierDataLoader(
            trnset,
            batch_size=self.cfg.training_cls.batchsize // 2,
            shuffle=True,
            drop_last=True,
        )
        self.valloader = JointClassifierDataLoader(
            valset,
            batch_size=self.cfg.training_cls.batchsize_eval // 2,
            shuffle=False,
            drop_last=True,
        )

        # initialize model in the very end
        # (might use the data shapes to dynamically set the input size)
        self.init_model()

    def create_evaluation_dataloaders(self, data_raw):
        loaders = []
        for ijet, raw in enumerate(data_raw):
            prepd, _ = self._preprocess(raw, prep_params=self.prep_params[ijet])
            prepd = torch.tensor(prepd)
            dataset = torch.utils.data.TensorDataset(prepd)
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=self.cfg.training_cls.batchsize_eval, shuffle=False
            )
            loaders.append(loader)
        return loaders

    def init_optimizer(self):
        if self.cfg.training_cls.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.cfg.training_cls.lr,
                betas=self.cfg.training_cls.betas,
                eps=self.cfg.training_cls.eps,
            )
        else:
            raise ValueError(
                f"Optimizer {self.cfg.training_cls.optimizer} not implemented"
            )
        LOGGER.debug(
            f"Using optimizer {self.cfg.training_cls.optimizer} with lr={self.cfg.training_cls.lr} for {self.label}"
        )

    def init_scheduler(self):
        if self.cfg.training_cls.scheduler is None:
            self.scheduler = None  # constant lr
        elif self.cfg.training_cls.scheduler == "OneCycleLR":
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                self.cfg.training_cls.lr * 10,
                total_steps=self.cfg.training_cls.iterations,
            )
        elif self.cfg.training_cls.scheduler == "CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.cfg.training_cls.iterations,
                eta_min=self.cfg.training_cls.lr_eta_min,
            )
        elif self.cfg.training_cls.scheduler == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                factor=self.cfg.training_cls.lr_factor,
                patience=self.cfg.training_cls.lr_patience,
            )
        else:
            raise ValueError(
                f"Learning rate scheduler {self.cfg.training_cls.scheduler} not implemented"
            )

        LOGGER.debug(
            f"Using learning rate scheduler {self.cfg.training_cls.scheduler} for {self.label}"
        )

    def init_loss(self):
        self.loss = nn.BCEWithLogitsLoss()
        self.get_LR = lambda x: torch.exp(x.clamp(max=10))

    def run_training(self):
        # initialize everything
        if not hasattr(self, "optimizer"):
            # only initialize these objects once
            self.init_loss()
        self.init_optimizer()
        self.init_scheduler()

        # initialize metrics
        self.train_lr, self.train_loss, self.val_loss = [], [], []
        self.train_metrics, self.val_metrics = (
            self._init_metrics(),
            self._init_metrics(),
        )
        smallest_val_loss, smallest_val_loss_epoch = 1e10, 0
        es_patience = 0

        # train loop
        LOGGER.info(
            f"Starting to train {self.label} for {self.cfg.training_cls.nepochs} epochs"
        )
        self.training_start_time = time.time()
        for epoch in range(self.cfg.training_cls.nepochs):
            # training
            for step, data in enumerate(self.trnloader):
                self._step(data, step=epoch * len(self.trnloader) + step)

            # validation
            val_loss = self._validate(epoch * len(self.trnloader))
            if self.cfg.training_cls.scheduler == "ReduceLROnPlateau":
                self.scheduler.step(val_loss)
            if val_loss < smallest_val_loss:
                smallest_val_loss = val_loss
                smallest_val_loss_epoch = epoch
                es_patience = 0

                if self.cfg.training_cls.save_best_model:
                    self.save_model(
                        os.path.join(self.cfg.run_dir, "models"),
                        f"{self.label_short}_run{self.cfg.run_idx}_ep{epoch}.pt",
                    )
            else:
                es_patience += 1
                if (
                    es_patience > self.cfg.training_cls.es_patience
                    and self.cfg.training_cls.early_stopping
                ):
                    LOGGER.info(f"Early stopping in epoch {epoch}")
                    break  # early stopping

            # output
            dt = time.time() - self.training_start_time
            if epoch == 0:
                dt_estimate = dt * self.cfg.training_cls.nepochs
                LOGGER.info(
                    f"Finished epoch 1 after {dt:.2f}s, "
                    f"training time estimate: {dt_estimate/60:.2f}min "
                    f"= {dt_estimate/60**2:.2f}h"
                )

        dt = time.time() - self.training_start_time
        LOGGER.info(
            f"Finished training for {self.cfg.training_cls.nepochs} epochs "
            f"after {dt/60:.2f}min = {dt/60**2:.2f}h"
        )
        if self.cfg.use_mlflow:
            log_mlflow(f"traintime_{self.label_short}", dt / 3600)

        # load best model
        if self.cfg.training_cls.es_load_best_model:
            model_path = os.path.join(
                self.cfg.run_dir,
                "models",
                f"{self.label_short}_run{self.cfg.run_idx}_ep{smallest_val_loss_epoch}.pt",
            )
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                LOGGER.info(f"Loading model from {model_path}")
                self.model.load_state_dict(state_dict)
            except FileNotFoundError:
                LOGGER.warning(
                    f"Cannot load best model (epoch {smallest_val_loss_epoch}) from {model_path}"
                )

        # could save best model in the end for future use
        # but have to be careful then because we have several classifiers (give them names?)

    def _step(self, data, step):
        self.model.train()

        # actual update step
        loss, metrics = self._batch_loss(data)
        self.optimizer.zero_grad()
        assert torch.isfinite(loss).all(), f"Loss is not finite"
        loss.backward()
        grad_norm = (
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.cfg.training_cls.clip_grad_norm,
                error_if_nonfinite=True,
            )
            .cpu()
            .item()
        )
        self.optimizer.step()

        if self.cfg.training_cls.scheduler in ["OneCycleLR", "CosineAnnealingLR"]:
            self.scheduler.step()

        # collect metrics
        self.train_loss.append(loss.item())
        self.train_lr.append(self.optimizer.param_groups[0]["lr"])
        for key, value in metrics.items():
            self.train_metrics[key].append(value)

        # log to mlflow
        if (
            self.cfg.use_mlflow
            and self.cfg.training_cls.log_every_n_steps != 0
            and step % self.cfg.training_cls.log_every_n_steps == 0
        ):
            log_dict = {
                "loss": loss.item(),
                "lr": self.train_lr[-1],
                "time_per_step": (time.time() - self.training_start_time) / (step + 1),
                "grad_norm": grad_norm,
            }
            for key, values in log_dict.items():
                log_mlflow(f"{self.label_short}.train.{key}", values, step=step)
            for key, values in metrics.items():
                log_mlflow(f"{self.label_short}.train.{key}", values, step=step)

    def _validate(self, step):
        losses = []
        metrics = self._init_metrics()

        self.model.eval()
        with torch.no_grad():
            for data in self.valloader:
                loss, metric = self._batch_loss(data)
                losses.append(loss.cpu().item())
                for key, value in metric.items():
                    metrics[key].append(value)
        val_loss = np.mean(losses)
        self.val_loss.append(val_loss)
        for key, values in metrics.items():
            self.val_metrics[key].append(np.mean(values))
        if self.cfg.use_mlflow:
            log_mlflow(f"{self.label_short}.val.loss", val_loss, step=step)
            for key, values in metrics.items():
                log_mlflow(f"{self.label_short}.val.{key}", np.mean(values), step=step)
        return val_loss

    def evaluate(self, events, label=None):
        LOGGER.info(
            f"Starting to evaluate {self.label} on {label} data of shapes {[event.shape for event in events]}"
        )
        loaders = self.create_evaluation_dataloaders(events)

        for event in events:
            assert np.isfinite(event).all(), f"Event is not finite"
        LOGGER.info(f"Created evaluation dataloaders for {self.label}")
        scores, weights = [], []
        self.model.eval()
        with torch.no_grad():
            # create a seperate dataloader for each multiplicity
            for ijet, loader in enumerate(loaders):
                scores_fixedmult, weights_fixedmult = [], []
                for (data,) in loader:
                    assert torch.isfinite(data).all(), f"Data is not finite"
                    logit = self._evaluate(data, ijet)
                    assert torch.isfinite(logit).all(), f"Logits are not finite"
                    # print the 20 largest logits
                    # LOGGER.info(f"20 largest logits: {logit.flatten().cpu().sort(descending=True).values[:20]}")
                    weight = self.get_LR(logit)
                    score = nn.functional.sigmoid(logit)
                    assert torch.isfinite(score).all(), f"Scores are not finite"
                    # assert torch.isfinite(weight).all(), f"Weights are not finite"
                    scores_fixedmult.append(score.flatten().cpu())
                    weights_fixedmult.append(weight.flatten().cpu())
                scores.append(np.concatenate(scores_fixedmult))
                weights.append(np.concatenate(weights_fixedmult))

        # save everything (for plotter)
        if label is not None:
            self.results_eval[label] = {"weights": weights, "scores": scores}
        return weights

    def save_model(self, path, filename):
        model_path = os.path.join(path, filename)
        torch.save(self.model.state_dict(), model_path)

    def _init_metrics(self):
        return {f"bce.{n_jets}j": [] for n_jets in self.cfg.data.n_jets_list}

    def _preprocess(self, data):
        raise NotImplementedError

    def _batch_loss(self, data):
        raise NotImplementedError

    def _evaluate(self, data):
        raise NotImplementedError
