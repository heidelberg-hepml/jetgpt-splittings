import numpy as np
import torch

import os, time
from omegaconf import open_dict

from src.base_experiment import BaseExperiment
from src.eventgen.dataset import EventDataset
import src.eventgen.plotter as plotter
from src.eventgen.helpers import enforce_pt_ordering, ensure_onshell
from src.logger import LOGGER

MODELNAME_DICT = {
    "JetGPT": "Full likelihood",
    "JetGPTBootstrap": "Bootstrap",
    "JetGPTTrunc": "Reduced loss",
    "JetGPTExtendLoss": "Perturbed target",
    "JetGPTPhysicsLoss": r"Override $p_\mathrm{split}$",
}


class EventGenerationExperiment(BaseExperiment):
    def init_physics(self):
        assert len(self.cfg.data.n_jets) > 1
        assert max(self.cfg.data.n_jets) <= self.cfg.data.n_jets_max
        assert len(self.cfg.data.n_jets_train) <= len(self.cfg.data.n_jets)

        self.modelname = self.cfg.model._target_.rsplit(".", 1)[-1].replace("_", r"\_")
        self.modelname = MODELNAME_DICT[self.modelname]

        with open_dict(self.cfg):
            # TBD later: generalize this
            self.cfg.model.pformer.in_channels = 4 + (self.n_hard_particles + 1)
            self.cfg.model.n_gauss = (
                self.cfg.model.n_gauss
                if self.cfg.model.n_gauss is not None
                else self.cfg.model.pformer.hidden_channels // 3
            )
            self.cfg.model.cformer.in_channels = (
                1 + 4 + (self.cfg.model.pformer.out_channels - 1)
            )
            self.cfg.model.cformer.out_channels = 3 * self.cfg.model.n_gauss
            self.cfg.model.preprocessing_cfg = self.cfg.preprocessing

    def init_data(self):
        LOGGER.info(
            f"Using n_jets={self.cfg.data.n_jets}, n_jets_train={self.cfg.data.n_jets_train}"
        )
        LOGGER.info(f"Loading datasets")
        # create train, test, val datasets
        empty = {
            ijet: torch.empty((0, self.n_hard_particles + ijet, 4), dtype=torch.float64)
            for ijet in range(self.n_hard_particles + self.cfg.data.n_jets_max)
        }
        self.data = {
            ijet: empty[ijet]
            for ijet in range(self.n_hard_particles + self.cfg.data.n_jets_max)
        }
        for ijet in self.cfg.data.n_jets:
            data_path = eval(f"self.cfg.data.data_path_{ijet}j")
            assert os.path.exists(data_path)
            data = np.load(data_path)
            data = data.reshape(data.shape[0], data.shape[1] // 4, 4)
            data = torch.tensor(data, dtype=torch.float64)
            data = ensure_onshell(data, self.onshell_list, self.onshell_mass)
            if self.cfg.data.enforce_pt_order:
                data = enforce_pt_ordering(data, self.n_hard_particles)
            self.data[ijet] = data

        self.model.extra_information(
            self.units,
            self.pt_min,
            self.onshell_list,
            self.n_hard_particles,
            self.cfg.data.n_jets_max,
            self.device,
        )

        events, idxs = [], []
        for ijet in self.cfg.data.n_jets:
            events.append(self.data[ijet])
            idx = torch.tensor(
                list(range(self.n_hard_particles)) + [self.n_hard_particles] * ijet,
            )
            idxs.append(idx)
        self.model.preprocessing.initialize_parameters(events, idxs)

    def _init_dataloader(self):
        assert sum(self.cfg.data.test_val_train) <= 1
        self.data_split = {}
        for ijet in self.cfg.data.n_jets_train:
            n_data = self.data[ijet].shape[0]
            splits = np.round(np.cumsum(self.cfg.data.test_val_train) * n_data).astype(
                "int"
            )
            tst, val, trn, _ = np.split(self.data[ijet], splits)
            self.data_split[ijet] = {
                "trn": trn,
                "val": val,
                "tst": tst,
            }

        self.loaders = {}
        shuffle = {"trn": True, "val": False, "tst": False}
        for label in ["trn", "val", "tst"]:
            dataset = EventDataset(
                [self.data_split[ijet][label] for ijet in self.cfg.data.n_jets_train]
            )
            self.loaders[label] = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.cfg.training.batchsize,
                shuffle=shuffle[label],
            )
        LOGGER.info(
            f"Created datasets with {len(self.loaders['trn'].dataset)}/{len(self.loaders['tst'].dataset)}/{len(self.loaders['val'].dataset)} (train/test/val) events"
        )

        if self.model.bayesian:
            self.model.traindata_size = torch.tensor(
                len(self.loaders["trn"].dataset), device=self.model.device
            )

    @torch.no_grad()
    def evaluate(self):
        n_evaluations = self.cfg.evaluation.nsamples_BNN if self.model.bayesian else 1
        samples, prob_stop = [], []
        for ieval in range(n_evaluations):
            LOGGER.info(f"Starting to evaluate model {ieval}/{n_evaluations}")
            samples0, prob_stop0 = self.evaluate_single(ieval)
            samples.append(samples0)
            prob_stop.append(prob_stop0)

        # organize samples and prob_stop
        self.samples, self.prob_stop = {}, {}
        for ijet in range(self.cfg.data.n_jets_max):
            self.samples[ijet] = [
                samples[ieval][ijet] for ieval in range(n_evaluations)
            ]
            self.prob_stop[ijet] = [
                prob_stop[ieval][ijet] for ieval in range(n_evaluations)
            ]

    def evaluate_single(self, ieval):
        self.model.eval()
        self.model.reset_BNN()
        for label in self.cfg.evaluation.log_prob:
            assert not self.model.bayesian
            if label == "gen":
                # log_probs of generated events are not interesting
                # + they are not well-defined, because generated events might be in regions
                # that are not included in the base distribution (because of pt_min, delta_r_min)
                continue
            self._evaluate_log_prob(self.loaders[label], label)

        if self.cfg.evaluation.sample:
            samples, prob_stop = self._sample_events(ieval)
        else:
            samples, prob_stop = None, None

        return samples, prob_stop

    def _evaluate_log_prob(self, loader, label):
        LOGGER.info(f"Starting to evaluate log_prob for model on {label} dataset")
        t0 = time.time()
        log_probs = []
        for batch in loader:
            events, ptr = batch
            events, ptr = events.to(self.device, dtype=self.dtype), ptr.to(self.device)
            log_prob = self.model.log_prob(events, ptr)
            log_probs.append(log_prob)
        dt = time.time() - t0
        LOGGER.info(f"Finished evaluating log_prob after {(dt/60):.2f}min")

        log_probs = torch.cat(log_probs)
        neg_log_prob = -log_probs.mean()
        LOGGER.info(f"log_prob on {label} dataset: NLL = {neg_log_prob:.4f}")

    def _sample_events(self, ieval):
        nbatches = self.cfg.evaluation.nsamples // self.cfg.evaluation.batchsize + 1
        samples = {ijet: [] for ijet in range(self.cfg.data.n_jets_max)}
        prob_stop = {ijet: [] for ijet in range(self.cfg.data.n_jets_max)}
        LOGGER.info(
            f"Starting to generate {self.cfg.evaluation.nsamples} events "
            f"using {nbatches} batches with batchsize {self.cfg.evaluation.batchsize}"
        )

        t0 = time.time()
        for _ in range(nbatches):
            new_samples, new_prob_stop = self.model.sample(
                self.cfg.evaluation.batchsize, self.device, self.dtype
            )
            for ijet in range(self.cfg.data.n_jets_max):
                new_samples[ijet] = ensure_onshell(
                    new_samples[ijet], self.onshell_list, self.onshell_mass
                )
                samples[ijet].append(new_samples[ijet])
                prob_stop[ijet].append(new_prob_stop[ijet])
        for ijet in range(self.cfg.data.n_jets_max):
            samples[ijet] = torch.cat(samples[ijet]).cpu()
        prob_stop = {
            ijet: torch.cat(prob_stop[ijet]).cpu()
            for ijet in range(self.cfg.data.n_jets_max)
        }
        dt = time.time() - t0

        LOGGER.info(f"Finished generating events after {dt/60:.2f}min")
        counts = [samples[ijet].shape[0] for ijet in range(self.cfg.data.n_jets_max)]
        LOGGER.info(f"Multiplicities of generated events: {counts}")

        if self.cfg.evaluation.save_samples:
            path = os.path.join(
                self.cfg.run_dir,
                "samples",
            )
            os.makedirs(path, exist_ok=True)
            filename = os.path.join(
                path,
                f"samples_run{self.cfg.run_idx}_set{ieval}.npz",
            )
            samples_kwargs = {
                f"samples_{ijet}j": samples[ijet]
                for ijet in range(self.cfg.data.n_jets_max)
            }
            probs_kwargs = {
                f"probs_{ijet}j": prob_stop[ijet]
                for ijet in range(self.cfg.data.n_jets_max)
            }
            np.savez(filename, **samples_kwargs, **probs_kwargs)
            LOGGER.info(f"Saved generated samples to {filename}")

        return samples, prob_stop

    def plot(self):
        path = os.path.join(self.cfg.run_dir, f"plots_{self.cfg.run_idx}")
        os.makedirs(os.path.join(path), exist_ok=True)
        LOGGER.info(f"Creating plots in {path}")

        self.plot_titles = [
            r"${%s}+{%s}j$" % (self.plot_title, ijet)
            for ijet in range(self.cfg.data.n_jets_max)
        ]
        kwargs = {
            "exp": self,
            "model_label": self.modelname,
        }

        if self.cfg.train:
            filename = os.path.join(path, "training.pdf")
            plotter.plot_losses(filename=filename, **kwargs)

            filename = os.path.join(path, "extra_plots.pdf")
            self.model.extra_plots(filename=filename)

        if not self.cfg.evaluate:
            return

        if self.cfg.plotting.prob_stop and self.cfg.evaluation.sample:
            filename = os.path.join(path, "prob_stop.pdf")
            plotter.plot_prob_stop(filename=filename, **kwargs)

        kwargs["weights"] = [None for ijet in range(self.cfg.data.n_jets_max)]
        kwargs["mask_dict"] = [None for ijet in range(self.cfg.data.n_jets_max)]

        if self.cfg.evaluation.sample:
            if self.cfg.plotting.fourmomenta:
                filename = os.path.join(path, "fourmomenta.pdf")
                plotter.plot_fourmomenta(filename=filename, **kwargs)

            if self.cfg.plotting.jetmomenta:
                filename = os.path.join(path, "jetmomenta.pdf")
                plotter.plot_jetmomenta(filename=filename, **kwargs)

            if self.cfg.plotting.preprocessed:
                filename = os.path.join(path, "preprocessed.pdf")
                plotter.plot_preprocessed(filename=filename, **kwargs)

            if self.cfg.plotting.conservation:
                filename = os.path.join(path, "conservation.pdf")
                plotter.plot_conservation(filename=filename, **kwargs)

            if self.cfg.plotting.virtual and len(self.virtual_components) > 0:
                filename = os.path.join(path, "virtual.pdf")
                plotter.plot_virtual(filename=filename, **kwargs)

            if self.cfg.plotting.delta:
                filename = os.path.join(path, "delta.pdf")
                plotter.plot_delta(filename=filename, **kwargs)

            if self.cfg.plotting.deta_dphi:
                filename = os.path.join(path, "deta_dphi.pdf")
                plotter.plot_deta_dphi(
                    filename=filename, exp=self, model_label=self.modelname
                )

    def _batch_loss(self, batch, step=None):
        events, num_particles = batch
        events, num_particles = (
            events.to(self.device, dtype=self.dtype),
            num_particles.to(self.device),
        )
        loss = self.model.batch_loss(events, num_particles, step=step)
        return loss, {}

    def _init_metrics(self):
        return {}
