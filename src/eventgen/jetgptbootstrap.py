import time
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from src.eventgen.jetgpt import JetGPT
from src.eventgen.helpers import EPPP_to_PtPhiEtaM2
from src.logger import LOGGER


class JetGPTBootstrap(JetGPT):
    def __init__(
        self,
        *args,
        bootstrap_ijet=[3],
        warmup_steps=0,
        sample_every_n_steps=10,
        buffer_size=10000,
        bootstrap_fraction=[0.1],
        sampling_multiplier=4,
        sampling_batches=1,
        initial_buffer_size=0,
        unweight_phi=False,
        unweight_phi_bins=10,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert len(bootstrap_ijet) == len(bootstrap_fraction)
        assert (
            len(bootstrap_ijet) == 1 or initial_buffer_size == 0
        ), f"initial_buffer_size>0 only implemented for having one extra jet"
        self.sample_every_n_steps = sample_every_n_steps
        self.buffer_size = buffer_size
        self.bootstrap_fraction = bootstrap_fraction
        self.bootstrap_ijet = bootstrap_ijet
        self.warmup_steps = warmup_steps
        self.sampling_multiplier = sampling_multiplier
        self.sampling_batches = sampling_batches
        self.buffer = None
        self.initial_buffer_size = initial_buffer_size
        self.unweight_phi = unweight_phi
        self.unweight_phi_bins = unweight_phi_bins

        self.tracker = {
            ibuffer: {
                "step": [],
                "buffer_size": [],
                "num_events": [],
                "uw_efficiency": [],
                "uw_weights_initial": None,
                "uw_weights_final": None,
            }
            for ibuffer in range(len(self.bootstrap_ijet))
        }

    def _batch_loss(self, events, num_particles, step):
        t0 = time.time()
        if self.buffer is None:
            self.buffer = [
                torch.empty(
                    (0, events.shape[1] + 1 + i, 4),
                    device=events.device,
                    dtype=events.dtype,
                )
                for i in range(len(self.bootstrap_ijet))
            ]

        batchsize = events.shape[0]
        for ibuffer in range(len(self.buffer)):
            if self.buffer[ibuffer].shape[0] > 0:
                events = torch.cat((events, torch.zeros_like(events[:, [0], :])), dim=1)
                num_bootstrap = int(batchsize * self.bootstrap_fraction[ibuffer])
                idx = torch.randint(0, self.buffer[ibuffer].shape[0], (num_bootstrap,))
                events_bootstrap = self.buffer[ibuffer][idx]
                events = torch.cat([events, events_bootstrap])
                num_particles_bootstrap = (
                    2 + self.bootstrap_ijet[ibuffer]
                ) * torch.ones(
                    num_bootstrap,
                    device=num_particles.device,
                    dtype=num_particles.dtype,
                )
                num_particles = torch.cat([num_particles, num_particles_bootstrap])

        # actually calculate loss
        loss = super()._batch_loss(events, num_particles, step)

        # optional initial buffer fill
        if (
            self.initial_buffer_size > 0
            and step is not None
            and step == self.warmup_steps
        ):
            assert len(self.buffer) == 1
            t0 = time.time()
            i = 0
            LOGGER.info(
                f"Initiating buffer fill with {self.initial_buffer_size} events"
            )
            while self.buffer[0].shape[0] < self.initial_buffer_size:
                valid_events_dict = self.buffer_step(
                    batchsize, events.device, events.dtype
                )
                self.append_to_buffer(valid_events_dict[self.bootstrap_ijet[0]], 0)
                i += 1
            LOGGER.info(
                f"Finished initial buffer fill after {i} iterations/{i*batchsize*self.sampling_multiplier*self.sampling_batches} events with {self.buffer[0].shape[0]} events"
            )

        # sample new events
        if (
            step is not None
            and step % self.sample_every_n_steps == 0
            and step >= self.warmup_steps
        ):
            t1 = time.time()
            valid_events_dict = self.buffer_step(batchsize, events.device, events.dtype)
            for ibuffer in range(len(self.buffer)):
                keep_events = valid_events_dict[self.bootstrap_ijet[ibuffer]]
                if (
                    self.buffer[ibuffer].shape[0] < self.buffer_size
                    and self.buffer[ibuffer].shape[0] + keep_events.shape[0]
                    > self.buffer_size
                ):
                    sampling_steps = (
                        step - self.warmup_steps
                    ) // self.sample_every_n_steps
                    LOGGER.info(
                        f"{self.bootstrap_ijet[ibuffer]}j buffer full for the first time in iteration {step}, "
                        f"i.e. after {sampling_steps}x sampling"
                    )

                self.append_to_buffer(keep_events, ibuffer, step=step)
                if step is not None and step - self.warmup_steps in [
                    0,
                    self.sample_every_n_steps,
                ]:
                    t2 = time.time()
                    LOGGER.info(
                        f"{self.bootstrap_ijet[ibuffer]}j sampling efficiency: {keep_events.shape[0]}/{batchsize*self.sampling_multiplier} = {keep_events.shape[0]/(batchsize*self.sampling_multiplier):.2f}"
                    )
                    LOGGER.info(
                        f"Fraction of time spent sampling {self.bootstrap_ijet[ibuffer]}j events in step {step}: {(t2-t1)/(t2-t1 + (t1-t0)*self.sample_every_n_steps):.2f}"
                    )
        return loss

    def buffer_step(self, batchsize, device, dtype):
        with torch.no_grad():
            valid_events_dict = {}
            for _ in range(self.sampling_batches):
                valid_events_dict_mini, _ = self.sample(
                    batchsize * self.sampling_multiplier, device, dtype
                )
                for key in valid_events_dict_mini.keys():
                    if key not in valid_events_dict:
                        valid_events_dict[key] = valid_events_dict_mini[key]
                    else:
                        valid_events_dict[key] = torch.cat(
                            [valid_events_dict[key], valid_events_dict_mini[key]], dim=0
                        )
        return valid_events_dict

    def append_to_buffer(self, events, ibuffer, step=None):
        if events.shape[0] == 0:
            return

        if step is not None:
            self.tracker[ibuffer]["step"].append(step)
            self.tracker[ibuffer]["buffer_size"].append(self.buffer[ibuffer].shape[0])
            self.tracker[ibuffer]["num_events"].append(events.shape[0])

        events0 = self.unweight_events(events, ibuffer, step=step)
        self.buffer[ibuffer] = torch.cat([self.buffer[ibuffer], events0], dim=0)
        self.buffer[ibuffer] = self.buffer[ibuffer][-self.buffer_size :]

    def unweight_events(self, events, ibuffer, step=None):
        if self.unweight_phi:
            # unweight for flat phi distributions
            # we can do that bc we know that the phi distributions should be flat
            ptphietam2 = EPPP_to_PtPhiEtaM2(events)
            x = ptphietam2[..., 1]

            # compute histograms (for all phi's in parallel)
            bin_edges = torch.linspace(
                -torch.pi,
                torch.pi,
                steps=self.unweight_phi_bins + 1,
                device=events.device,
            )
            bin_indices = torch.searchsorted(bin_edges, x.contiguous())
            bin_indices = torch.clamp(bin_indices - 1, min=0, max=len(bin_edges) - 1)
            one_hot_bins = torch.nn.functional.one_hot(
                bin_indices, num_classes=len(bin_edges) - 1
            )
            hist = one_hot_bins.sum(dim=0).permute(1, 0)
            N_CRIT = 10
            if (hist < N_CRIT).any():
                LOGGER.warning(
                    f"Unweighting phi: low statistics in bins {hist[hist < N_CRIT]}"
                )
            second_idx = torch.arange(hist.shape[1], device=events.device)
            weights = 1 / hist.float()[bin_indices, second_idx]

            # collect weights
            weights = torch.prod(weights, dim=-1)
            weights = weights / weights.max()

            if step is not None:
                self.tracker[ibuffer]["uw_efficiency"].append(weights.mean().item())
            if self.tracker[ibuffer]["uw_weights_initial"] is None:
                print(weights.shape)
                self.tracker[ibuffer]["uw_weights_initial"] = weights.cpu().numpy()
            self.tracker[ibuffer]["uw_weights_final"] = weights.cpu().numpy()

            # unweighting
            mask = torch.rand_like(events[:, 0, 0]) < weights
            events = events[mask]
        return events

    def extra_plots(self, filename):
        with PdfPages(filename) as file:
            for ibuffer in range(len(self.buffer)):
                plt.plot(
                    self.tracker[ibuffer]["step"],
                    self.tracker[ibuffer]["buffer_size"],
                )
                _, ymax = plt.ylim()
                plt.ylim(0, ymax)
                plt.xlabel("Step")
                plt.ylabel(f"Buffer size for {self.bootstrap_ijet[ibuffer]}j")
                plt.savefig(file, format="pdf", bbox_inches="tight")
                plt.close()

                plt.plot(
                    self.tracker[ibuffer]["step"],
                    self.tracker[ibuffer]["num_events"],
                )
                plt.xlabel("Step")
                plt.ylabel(f"Number of new events for {self.bootstrap_ijet[ibuffer]}j")
                plt.savefig(file, format="pdf", bbox_inches="tight")
                plt.close()

                if self.unweight_phi:
                    plt.plot(
                        self.tracker[ibuffer]["step"],
                        self.tracker[ibuffer]["uw_efficiency"],
                    )
                    plt.xlabel("Step")
                    plt.ylabel(
                        f"Unweighting efficiency for {self.bootstrap_ijet[ibuffer]}j"
                    )
                    plt.savefig(file, format="pdf", bbox_inches="tight")
                    plt.close()

                    plt.hist(
                        self.tracker[ibuffer]["uw_weights_initial"],
                        bins=100,
                        range=(0, 1),
                    )
                    plt.xlim(0, 1)
                    plt.xlabel(
                        f"Unweighting weights for {self.bootstrap_ijet[ibuffer]}j"
                    )
                    plt.savefig(file, format="pdf", bbox_inches="tight")
                    plt.close()

                    plt.hist(
                        self.tracker[ibuffer]["uw_weights_final"],
                        bins=100,
                        range=(0, 1),
                    )
                    plt.xlim(0, 1)
                    plt.xlabel(
                        f"Unweighting weights for {self.bootstrap_ijet[ibuffer]}j"
                    )
                    plt.savefig(file, format="pdf", bbox_inches="tight")
                    plt.close()
