import numpy as np

from src.experiments.jets import JetExperiment


class ttbarExperiment(JetExperiment):
    def __init__(self, cfg):

        self.plot_title = r"t\bar t"

        self.n_hard_particles = 6
        self.n_jets_max = 4
        self.channels_out = []

        self.obs_names_index = ["b1", "q1", "q2", "b2", "q3", "q4"]
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

        self.virtual_components = [
            [0, 1, 2, 3, 4, 5],
            [0, 1, 2],
            [3, 4, 5],
            [1, 2],
            [4, 5],
        ]
        self.virtual_names = [
            r"p_{T,t\bar t}",
            r"\phi_{t\bar t}",
            r"\eta_{t\bar t}",
            r"m_{t\bar t}",
            "p_{T,t}",
            "\phi_t",
            "\eta_t",
            "m_{ t }",
            r"p_{T,\bar t}",
            r"\phi_{\bar t}",
            r"\eta_{\bar t}",
            r"m_{\bar t}",
            "p_{T,W^+}",
            "\phi_{W^+}",
            "\eta_{W^+}",
            "m_{W^+}",
            "p_{T,W^-}",
            "\phi_{W^-}",
            "\eta_{W^-}",
            "m_{W^-}",
        ]
        self.virtual_ranges = [
            [0, 500],
            [-np.pi, np.pi],
            [-6, 6],
            [200, 1000],
            [0, 500],
            [-np.pi, np.pi],
            [-6, 6],
            [50, 400],
            [0, 500],
            [-np.pi, np.pi],
            [-6, 6],
            [50, 400],
            [0, 300],
            [-np.pi, np.pi],
            [-6, 6],
            [30, 150],
            [0, 300],
            [-np.pi, np.pi],
            [-6, 6],
            [30, 150],
        ]

        super().__init__(cfg)
