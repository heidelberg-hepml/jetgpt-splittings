import numpy as np

from src.experiments.jets import JetExperiment


class zmumuExperiment(JetExperiment):
    def __init__(self, cfg):
        self.plot_title = "Z"

        self.n_hard_particles = 2
        self.n_jets_max = 5
        self.channels_out = [3, 7]

        self.obs_names_index = ["l1", "l2"]
        for ijet in range(self.n_jets_max):
            self.obs_names_index.append(f"j{ijet+1}")

        self.obs_names = []
        for name in self.obs_names_index:
            self.obs_names.append("p_{T," + name + "}")
            self.obs_names.append("\phi_{" + name + "}")
            self.obs_names.append("\eta_{" + name + "}")
            self.obs_names.append("m_{" + name + "}")

        self.obs_ranges = []
        for _ in range(self.n_hard_particles + self.n_jets_max):
            self.obs_ranges.append([10, 150])
            self.obs_ranges.append([-np.pi, np.pi])
            self.obs_ranges.append([-6, 6])
            self.obs_ranges.append([0, 20])

        self.virtual_components = [[0, 1]]
        self.virtual_ranges = [[0, 300], [-np.pi, np.pi], [-6, 6], [75, 115]]
        self.virtual_names = [
            r"p_{T,\mu\mu}",
            r"\phi_{\mu\mu}",
            r"\eta_{\mu\mu}",
            r"m_{\mu\mu}",
        ]

        super().__init__(cfg)
