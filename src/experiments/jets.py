import numpy as np
import torch

import os, time
from omegaconf import OmegaConf, open_dict

from src.utils.logger import LOGGER
from src.utils.mlflow import log_mlflow

from src.experiments.base_experiment import BaseExperiment
import src.models
import src.utils.plotter as plotter
from src.utils.physics import fourmomenta_to_jetmomenta

torch.autograd.set_detect_anomaly(False)


class JetExperiment(BaseExperiment):
    def __init__(self, cfg):
        super().__init__(cfg)
        # some code that is similar for all types of datasets
        self.plot_titles = [
            r"${%s}+{%s}j$" % (self.plot_title, n_jets)
            for n_jets in self.cfg.data.n_jets_list
        ]
        # in plots, use logy=True for pt and logy=False otherwise
        self.obs_logy = np.concatenate(
            [[True, False, False, False] for _ in range(len(self.obs_names_index))]
        ).tolist()
        self.virtual_logy = np.concatenate(
            [[True, False, False, False] for _ in range(len(self.virtual_names) // 4)]
        ).tolist()

    def full_run(self):
        t0 = time.time()

        # save config
        LOGGER.debug(OmegaConf.to_yaml(self.cfg))
        self._save_config("config.yaml", to_mlflow=True)
        self._save_config(f"config_{self.cfg.run_idx}.yaml")

        self.init_data()
        self.extend_cfg()
        self.init_model()
        self.generator.init_dataloaders(self.truth_raw, self.channels, store_truth=True)

        if self.cfg.train_gen:
            self.generator.init_optimizer()
            self.generator.init_scheduler(epochs=self.cfg.training_gen.nepochs)

            self.run_training(run="gen", nepochs=self.cfg.training_gen.nepochs)
            self.generator.save_model(
                os.path.join(self.cfg.run_dir, "models"),
                f"gen_run{self.cfg.run_idx}.pt",
            )
            self._plot_loss_gen(
                os.path.join(self.cfg.run_dir, f"plots{self.cfg.run_idx}"), title="gen"
            )

        if self.cfg.discformer.iterations > 0:
            self.discform()

        self._test_generator()

        if self.cfg.sample:
            self.sample_generator()

        if self.cfg.reweight:
            self.reweight()

        if self.cfg.plot:
            self.plot()

        # GPU RAM information
        if self.device == torch.device("cuda"):
            max_used = torch.cuda.max_memory_allocated()
            max_total = torch.cuda.mem_get_info()[1]
            LOGGER.info(
                f"GPU RAM information: max_used = {max_used/1e9:.3} GB, max_total = {max_total/1e9:.3} GB"
            )

        # save config once again (in case things have been overwritten)
        self._save_config("config.yaml", to_mlflow=True)
        self._save_config(f"config_{self.cfg.run_idx}.yaml")
        dt = time.time() - t0
        LOGGER.info(
            f"Finished experiment {self.cfg.exp_name}/{self.cfg.run_name} after {dt/60:.2f}min = {dt/60**2:.2f}h"
        )

    ### initialize model and data

    def init_data(self):
        assert (
            np.diff(self.cfg.data.n_jets_list) > 0
        ).all(), f"n_jets_list={self.cfg.data.n_jets_list} is not ordered"
        LOGGER.info(f"Working on jet multiplicities {self.cfg.data.n_jets_list}")

        # load data
        self.truth_raw = []
        for n_jets in self.cfg.data.n_jets_list:
            data_path = os.path.join(
                self.cfg.data.data_path, eval(f"self.cfg.data.file_{n_jets}j")
            )
            assert os.path.exists(data_path)
            data_raw = np.load(data_path)
            data_raw = data_raw.reshape(data_raw.shape[0], data_raw.shape[1] // 4, 4)
            data_raw = fourmomenta_to_jetmomenta(data_raw)
            assert (
                data_raw.shape[1] == self.n_hard_particles + n_jets
            ), f"Need {self.n_hard_particles} particles, but loaded data has shape {data_raw.shape}"
            self.truth_raw.append(data_raw)
            LOGGER.info(
                f"Loaded {n_jets}j data of shape {data_raw.shape} from {data_path}"
            )

        # create channels list
        self.channels = []
        for n_jets in self.cfg.data.n_jets_list:
            mask = np.array(self.cfg.data.channels) < 4 * (
                self.n_hard_particles + n_jets
            )
            self.channels.append(np.array(self.cfg.data.channels)[mask].tolist())
        LOGGER.info(f"Working on channels {self.channels}")

    def extend_cfg(self):
        # write parameters to cfg to access them more easily
        with open_dict(self.cfg):
            self.cfg.data.n_hard_particles = self.n_hard_particles
            self.cfg.data.n_jets_max = self.n_jets_max
            self.cfg.data.virtual_components = self.virtual_components

    def init_model(self):
        if self.cfg.generator.name == "JetGPT":
            self.generator = src.models.JetGPT(
                self.cfg, self.warm_start, self.device, self.dtype
            )
        elif self.cfg.generator.name == "JetGPT2":
            raise NotImplementedError
        else:
            raise ValueError(f"generator {self.cfg.generator.name} not implemented")

        # update cfg (generator sets its architecture parameters dynamically)
        self.cfg = self.generator.cfg

    ### generator training code

    def run_training(self, run, nepochs, iteration=0):

        # performance metrics
        self.train_lr, self.train_loss, self.val_loss, self.grad_norm = [], [], [], []
        self.train_metrics = self._init_metrics()
        self.val_metrics = self._init_metrics()

        smallest_val_loss, smallest_val_loss_epoch = 1e10, 0
        es_patience = 0

        # main train loop
        LOGGER.info(f"Starting to train generator ({run}) for {nepochs} epochs")
        self.training_start_time = time.time()
        for epoch in range(nepochs):

            # training
            for step, data in enumerate(self.generator.trnloader):
                self._step_generator(
                    data,
                    iteration=iteration,
                    step=epoch * len(self.generator.trnloader) + step,
                    run=run,
                )

            # validation
            val_loss = self._validate_generator(
                iteration=iteration,
                step=epoch * len(self.generator.trnloader),
                run=run,
            )
            if val_loss < smallest_val_loss:
                smallest_val_loss = val_loss
                smallest_val_loss_epoch = epoch
                es_patience = 0

                if self.cfg.training_gen.save_best_model:
                    self.generator.save_model(
                        os.path.join(self.cfg.run_dir, "models"),
                        f"{run}_run{self.cfg.run_idx}_ep{epoch}.pt",
                    )
            else:
                es_patience += 1
                if (
                    es_patience > self.cfg.training_gen.es_patience
                    and self.cfg.training_gen.early_stopping
                ):
                    LOGGER.info(f"Early stopping in epoch {epoch}")
                    break  # early stopping

            # output
            dt = time.time() - self.training_start_time
            if epoch == 0:
                dt_estimate = dt * nepochs
                LOGGER.info(
                    f"Finished epoch 1 after {dt:.2f}s, "
                    f"training time estimate: {dt_estimate/60:.2f}min "
                    f"= {dt_estimate/60**2:.2f}h"
                )

        dt = time.time() - self.training_start_time
        LOGGER.info(
            f"Finished training for {nepochs} epochs "
            f"after {dt/60:.2f}min = {dt/60**2:.2f}h"
        )
        if self.cfg.use_mlflow:
            log_mlflow(f"traintime_{run}", dt / 3600)

        # load best model
        if self.cfg.training_gen.es_load_best_model:
            LOGGER.warning(
                f"Loading best-validation generator after training, this might decrease performance. "
                "Use training_gen.es_load_best_model=False instead."
            )
            model_path = os.path.join(
                self.cfg.run_dir,
                "models",
                f"{run}_run{self.cfg.run_idx}_ep{smallest_val_loss_epoch}.pt",
            )
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                LOGGER.info(f"Loading model and optimizer from {model_path}")
                self.generator.model.load_state_dict(state_dict["model"])
                self.generator.optimizer.load_state_dict(state_dict["optimizer"])
            except FileNotFoundError:
                LOGGER.warning(
                    f"Cannot load best model and optimizer (epoch {smallest_val_loss_epoch}) from {model_path}"
                )

    def _step_generator(self, data, iteration, step, run):
        self.generator.train()

        # actual update step
        loss, metrics = self._batch_loss_generator(data, iteration, step, train=True)
        self.generator.optimizer.zero_grad()
        loss.backward()
        grad_norm = (
            torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(),
                self.cfg.training_gen.clip_grad_norm,
                error_if_nonfinite=False,
            )
            .cpu()
            .item()
        )
        self.generator.optimizer.step()

        if self.cfg.training_gen.scheduler in ["OneCycleLR", "CosineAnnealingLR"]:
            self.generator.scheduler.step()

        # collect metrics
        self.train_loss.append(loss.item())
        self.train_lr.append(self.generator.optimizer.param_groups[0]["lr"])
        self.grad_norm.append(grad_norm)
        for key, value in metrics.items():
            self.train_metrics[key].append(value)

        # log to mlflow
        if (
            self.cfg.use_mlflow
            and self.cfg.training_gen.log_every_n_steps != 0
            and step % self.cfg.training_gen.log_every_n_steps == 0
        ):
            log_dict = {
                "loss": loss.item(),
                "lr": self.train_lr[-1],
                "time_per_step": (time.time() - self.training_start_time) / (step + 1),
                "grad_norm": grad_norm,
            }
            for key, values in log_dict.items():
                log_mlflow(f"{run}.train.{key}", values, step=step)

            for key, values in metrics.items():
                log_mlflow(f"{run}.train.{key}", values, step=step)

    def _validate_generator(self, iteration, step, run):
        losses = []
        metrics = self._init_metrics()

        self.generator.eval()
        with torch.no_grad():
            for data in self.generator.valloader:
                loss, metric = self._batch_loss_generator(
                    data, iteration, step=step, train=False
                )
                losses.append(loss.cpu().item())
                for key, value in metric.items():
                    metrics[key].append(value)
        val_loss = np.mean(losses)
        self.val_loss.append(val_loss)
        for key, values in metrics.items():
            self.val_metrics[key].append(np.mean(values))
        if self.cfg.use_mlflow:
            log_mlflow(f"{run}.val.loss", val_loss, step=step)
            for key, values in metrics.items():
                log_mlflow(f"{run}.val.{key}", np.mean(values), step=step)
        return val_loss

    def _batch_loss_generator(self, data, iteration, step, train):
        # Note: step is required for discformer alpha scheduler update

        loss = 0.0
        log_probs = []
        for ijet, batch in enumerate(data):
            context = {"ijet": ijet, "channels": self.channels[ijet]}
            if len(batch) == 1:  # normal training
                (x,) = batch
            else:  # discformer training
                x, log_prob0, weight = batch
                log_prob0 = log_prob0.to(device=self.device, dtype=self.dtype)
                weight = weight.to(device=self.device, dtype=self.dtype)
            x = x.to(device=self.device, dtype=self.dtype)
            log_prob = self.generator.log_prob(x, context)
            log_probs.append(log_prob.mean())
            if len(batch) == 1:  # normal training
                loss += -log_prob.mean() / len(data)
            else:  # discformer training
                discformer_factor = self.discformer.loss_factor(
                    log_prob, log_prob0, weight, ijet, iteration, step=step, train=train
                )
                loss += -torch.mean(log_prob * discformer_factor) / len(data)
        assert torch.isfinite(loss)

        metrics = {
            f"neg_log_prob.{n_jets}j": -log_probs[i].detach().cpu()
            for i, n_jets in enumerate(self.cfg.data.n_jets_list)
        }
        return loss, metrics

    def _init_metrics(self):
        return {f"neg_log_prob.{n_jets}j": [] for n_jets in self.cfg.data.n_jets_list}

    def sample_generator(self):
        contexts = [
            {"ijet": ijet, "channels": channels}
            for ijet, channels in enumerate(self.channels)
        ]
        (
            self.generator.gen_raw,
            self.generator.gen_prepd,
        ) = self.generator.sample(self.cfg.sampling.nsamples, contexts)

    def _test_generator(self):
        contexts = [
            {"ijet": ijet, "channels": channels}
            for ijet, channels in enumerate(self.channels)
        ]
        self.generator.test_log_probs = self.generator.evaluate_log_prob(
            [data["tst"] for data in self.generator.truth_raw], contexts
        )
        LOGGER.info(
            f"Test log probs: {[log_prob.mean() for log_prob in self.generator.test_log_probs]}"
        )

    def discform(self):
        # init discifier = discformer classifier
        # use same settings as for the reweighter
        label = "Discifier"
        if self.cfg.classifier.name == "MLP":
            self.discifier = src.models.MLPClassifier(
                label, self.cfg, self.device, self.dtype
            )
        elif self.cfg.classifier.name == "Transformer":
            self.discifier = src.models.TransformerClassifier(
                label, self.cfg, self.device, self.dtype
            )
        else:
            raise ValueError(f"Classifier {self.cfg.classifier.name} not implemented")

        self.discformer = src.models.DiscFormer(self.cfg)
        for iteration in range(self.cfg.discformer.iterations):
            self.discformer_iteration(iteration + 1)

    def discformer_iteration(self, iteration):
        LOGGER.info(f"Starting DiscFormer iteration {iteration}")
        path = os.path.join(
            self.cfg.run_dir, f"plots{self.cfg.run_idx}", f"discformer_it{iteration}"
        )
        os.makedirs(path, exist_ok=True)

        # create discifier dataloader
        contexts = [
            {"ijet": ijet, "channels": channels}
            for ijet, channels in enumerate(self.channels)
        ]
        (
            self.generator.gen_raw,
            self.generator.gen_prepd,
        ) = self.generator.sample(self.cfg.discformer.nsamples_discifier, contexts)

        # warm start discifier from previous run
        if self.cfg.discformer.discifier.warm_start and iteration == 1:
            LOGGER.info("Warm-starting discifier from existing discifier")
            self.discifier.init_dataloaders(
                [data["trn"] for data in self.generator.truth_raw],
                self.generator.gen_raw,
            )
            discifier_path = os.path.join(
                self.cfg.run_dir,
                "models",
                f"disc{iteration}_run{self.cfg.discformer.discifier.warm_start_idx}.pt",
            )
            try:
                state_dict = torch.load(discifier_path, map_location=self.device)

            except:
                raise ValueError(f"Cannot load discifier from {discifier_path}")
            LOGGER.info(f"Loading discifier from {discifier_path}")
            state_dict = {"model." + key: value for key, value in state_dict.items()}
            self.discifier.load_state_dict(state_dict)
            self.discifier.init_loss()  # defines the get_LR method
        else:
            self.discifier.init_dataloaders(
                [data["trn"] for data in self.generator.truth_raw],
                self.generator.gen_raw,
            )
            self.discifier.run_training()
            self.discifier.save_model(
                os.path.join(self.cfg.run_dir, "models"),
                f"disc{iteration}_run{self.cfg.run_idx}.pt",
            )

        # evaluate discifier and create plots (careful: did not re-sample for this -> might have overfitting)
        self.discifier.evaluate(self.generator.gen_raw, "generator")
        self.discifier.evaluate(
            [data["tst"] for data in self.generator.truth_raw], "truth"
        )

        # plots classifier performance (if the classifier is trained)
        if not (self.cfg.discformer.discifier.warm_start and iteration == 1):
            filename = os.path.join(path, f"loss_dfc.pdf")
            plotter.plot_cls_metrics(self, self.discifier, filename)
            if self.cfg.plotting.classifier:
                filename = os.path.join(path, "classifier.pdf")
                plotter.plot_classifier(self, self.discifier, filename)

        # prepare generator dataloader and discformer training
        if self.cfg.discformer.discformation.sample_model:
            (
                self.generator.gen_raw,
                self.generator.gen_prepd,
            ) = self.generator.sample(self.cfg.discformer.nsamples_discformer, contexts)
            samples = self.generator.gen_raw
        else:
            samples = self.truth_raw
        log_probs = self.generator.evaluate_log_prob(samples, contexts)
        weights = self.discifier.evaluate(samples, "discformer")
        self.generator.init_dataloaders(
            samples, self.channels, log_probs=log_probs, weights=weights
        )
        self.discformer.prepare_training(
            max_steps=self.cfg.discformer.nepochs * len(self.generator.trnloader)
        )

        # plots to check the reweighting and discformation
        self._plot_discformed_distributions(path, weights)
        if self.cfg.discformer.plot_only:
            # stop here, if only interested in discformed distributions
            LOGGER.info(
                "Stopping discformer after creating discformer-reweighted plots"
            )
            return

        # generator train loop
        if not hasattr(self.generator, "optimizer"):
            # only called when warm-starting; optimizer is then warm-started as well
            self.generator.init_optimizer()
        self.generator.init_scheduler(
            epochs=self.cfg.discformer.nepochs
        )  # re-initialise scheduler
        self.run_training(
            run=f"df{iteration}",
            nepochs=self.cfg.discformer.nepochs,
            iteration=iteration,
        )
        self.generator.save_model(
            os.path.join(self.cfg.run_dir, "models"),
            f"df{iteration}_run{self.cfg.run_idx}.pt",
        )
        self._test_generator()

        # plots
        self._plot_loss_gen(path, "gen")
        if self.cfg.plotting.discformer.metrics:
            filename = os.path.join(path, "metrics.pdf")
            plotter.plot_discformer(self, filename)
        if self.cfg.plotting.discformer.resample:
            (
                self.generator.gen_raw,
                self.generator.gen_prepd,
            ) = self.generator.sample(self.cfg.sampling.nsamples, contexts)
            self._plot_distributions(title="distributions", path=path)

    def reweight(self):
        # init reweighter
        label = "Reweighter"
        if self.cfg.classifier.name == "MLP":
            self.reweighter = src.models.MLPClassifier(
                label, self.cfg, self.device, self.dtype
            )
        elif self.cfg.classifier.name == "Transformer":
            self.reweighter = src.models.TransformerClassifier(
                label, self.cfg, self.device, self.dtype
            )
        else:
            raise ValueError(f"Classifier {self.cfg.classifier.name} not implemented")

        # create generator events and use them to initialize dataloader
        contexts = [
            {"ijet": ijet, "channels": channels}
            for ijet, channels in enumerate(self.channels)
        ]
        gen_samples, _ = self.generator.sample(
            self.cfg.training_cls.gen_samples, contexts
        )
        self.reweighter.init_dataloaders(self.truth_raw, gen_samples)

        # train classifier
        self.reweighter.run_training()

        # compute weights of original events
        self.generator.weights = self.reweighter.evaluate(
            self.generator.gen_raw, "generator"
        )
        self.reweighter.evaluate(
            [data["tst"] for data in self.generator.truth_raw], "truth"
        )

        # plot results
        path = os.path.join(self.cfg.run_dir, f"plots{self.cfg.run_idx}")
        os.makedirs(path, exist_ok=True)
        filename = os.path.join(path, "loss_rew.pdf")
        plotter.plot_cls_metrics(self, self.reweighter, filename)
        if self.cfg.plotting.classifier:
            filename = os.path.join(path, "classifier.pdf")
            plotter.plot_classifier(self, self.reweighter, filename)

    def plot(self):
        # prepare folder
        path = os.path.join(self.cfg.run_dir, f"plots{self.cfg.run_idx}")
        os.makedirs(path, exist_ok=True)

        # Final distributions
        if self.cfg.reweight:
            weights_clipped = [
                w.clip(max=self.cfg.plotting.w_max) for w in self.generator.weights
            ]
            reweight_dict = [
                {"mode": "model", "weights": weights} for weights in weights_clipped
            ]
        else:
            reweight_dict = None
        self._plot_distributions(
            title="final",
            path=path,
            reweight_dict=reweight_dict,
            mask_dict=None,
        )

        # Masked distributions
        if self.cfg.plotting.distributions_masked.use and self.cfg.reweight:
            # Currently only mask defined by weight cut implemented
            # Feel free to implement more masks, if needed
            weight_cut = self.cfg.plotting.distributions_masked.weight_cut
            cut_above = self.cfg.plotting.distributions_masked.cut_above
            mask_dict = [
                {
                    "condition": r"$w<{%s}$" % weight_cut
                    if cut_above
                    else r"$w>{%s}$" % weight_cut,
                    "mask": weights < weight_cut if cut_above else weights > weight_cut,
                    "color": "violet",
                }
                for weights in self.generator.weights
            ]
            self._plot_distributions(
                title=f"weightcut_{weight_cut}",
                path=path,
                reweight_dict=None,
                mask_dict=mask_dict,
            )

    def _plot_loss_gen(self, path, title):
        # plot the currently saved generator loss curve

        os.makedirs(path, exist_ok=True)
        filename = os.path.join(path, f"loss_{title}.pdf")
        plotter.plot_gen_metrics(self, filename)

    def _plot_discformed_distributions(self, path, weights):
        """
        Plot the distributions of the data after reweighting and discformation.
        """
        if self.cfg.plotting.discformer.reweighted_test:
            if self.cfg.discformer.discformation.sample_model:
                # we have to compute the weights for test data, since generator.tstloader contains generated samples
                w_data = self.discifier.evaluate(
                    [data["tst"] for data in self.generator.truth_raw],
                    "discformer-reweighted-test",
                )
                weights_reweight_data = [1 / w for w in w_data]
            else:
                # data weights are already stored in generator.tstloader
                weights_reweight_data = [
                    1 / w for w in self.generator.tstloader.dataset.weights
                ]
            weights_clipped = [
                w.clip(max=self.cfg.plotting.w_max) for w in weights_reweight_data
            ]
            reweight_dict = [
                {"mode": "test", "weights": weight} for weight in weights_clipped
            ]
            self._plot_distributions(
                title=f"discifier-reweighted-test",
                path=path,
                reweight_dict=reweight_dict,
            )

        if self.cfg.plotting.discformer.reweighted_model:
            if self.cfg.discformer.discformation.sample_model:
                # model weights are already stored in generator.tstloader
                weights_reweight_model = weights
            else:
                # we have to compute the weights for generated samples, since generator.tstloader contains truth data
                weights_reweight_model = self.discifier.evaluate(
                    self.generator.gen_raw, "discformer-reweighted-model"
                )

            weights_clipped = [
                w.clip(max=self.cfg.plotting.w_max) for w in weights_reweight_model
            ]
            reweight_dict = [
                {"mode": "model", "weights": weight} for weight in weights_clipped
            ]
            self._plot_distributions(
                title=f"discifier-reweighted-model",
                path=path,
                reweight_dict=reweight_dict,
            )

        mode = "model" if self.cfg.discformer.discformation.sample_model else "test"
        weights_base = (
            weights_reweight_model
            if mode == "model"
            else self.generator.tstloader.dataset.weights
        )

        if self.cfg.plotting.discformer.discformed:
            discformed_weights = [
                self.discformer.get_weight(torch.tensor(weight)).numpy()
                for weight in weights_base
            ]
            reweight_dict = [
                {
                    "mode": mode,
                    "weights": weight,
                    "pre_label": "Discf.",
                }
                for weight in discformed_weights
            ]
            self._plot_distributions(
                title=f"discformed", path=path, reweight_dict=reweight_dict
            )

    def _plot_distributions(self, title, path, reweight_dict=None, mask_dict=None):
        # plot a set of distributions
        # (might call this several times with different weights, mask etc)

        if title is not None:
            path = os.path.join(path, title)
        os.makedirs(os.path.join(path), exist_ok=True)
        LOGGER.info(f"Creating {title} plots in {path}")

        # prepare kwargs
        kwargs = {
            "exp": self,
            "reweight_dict": reweight_dict,
            "mask_dict": mask_dict,
        }

        if self.cfg.plotting.jetmomenta:
            filename = os.path.join(path, "jetmomenta.pdf")
            plotter.plot_jetmomenta(filename=filename, **kwargs)

        if self.cfg.plotting.preprocessed:
            filename = os.path.join(path, "preprocessed.pdf")
            plotter.plot_preprocessed(filename=filename, **kwargs)

        if self.cfg.plotting.virtual:
            filename = os.path.join(path, "virtual.pdf")
            plotter.plot_virtual(filename=filename, **kwargs)

        if self.cfg.plotting.delta:
            filename = os.path.join(path, "delta.pdf")
            plotter.plot_delta(filename=filename, **kwargs)

        if self.cfg.plotting.deta_dphi:
            filename = os.path.join(path, "deta_dphi.pdf")
            plotter.plot_deta_dphi(filename=filename, **kwargs)
