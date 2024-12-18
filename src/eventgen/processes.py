import numpy as np

from src.eventgen.experiment import EventGenerationExperiment


class ttbarExperiment(EventGenerationExperiment):
    """
    Process: p p > t t~ at reco-level with hadronic top decays and 0-4 extra jets
    """

    def __init__(self, cfg):
        self.plot_title = r"t\bar t"
        self.n_hard_particles = 6
        self.onshell_list = []
        self.onshell_mass = []
        self.units = 206.6
        self.pt_min = [22.0] * 7
        self.delta_r_min = 0.5
        self.obs_names_index = ["b1", "q1", "q2", "b2", "q3", "q4"]
        for ijet in range(cfg.data.n_jets_max):
            self.obs_names_index.append(f"j{ijet+1}")
        self.fourmomentum_ranges = [[0, 200], [-150, 150], [-150, 150], [-150, 150]]
        self.jetmomentum_ranges = [[10, 150], [-np.pi, np.pi], [-6, 6], [0, 20]]
        self.virtual_components = [
            [0, 1, 2, 3, 4, 5],
            [0, 1, 2],
            [3, 4, 5],
            [1, 2],
            [4, 5],
        ]
        self.virtual_names = [
            r"p_{T,t\bar t} \;[\mathrm{GeV}]",
            r"\phi_{t\bar t}",
            r"\eta_{t\bar t}",
            r"m_{t\bar t} \;[\mathrm{GeV}]",
            r"p_{T,t} \;[\mathrm{GeV}]",
            r"\phi_t",
            r"\eta_t",
            r"m_{ t } \;[\mathrm{GeV}]",
            r"p_{T,\bar t} \;[\mathrm{GeV}]",
            r"\phi_{\bar t}",
            r"\eta_{\bar t}",
            r"m_{\bar t} \;[\mathrm{GeV}]",
            r"p_{T,W^+} \;[\mathrm{GeV}]",
            r"\phi_{W^+}",
            r"\eta_{W^+}",
            r"m_{W^+} \;[\mathrm{GeV}]",
            r"p_{T,W^-} \;[\mathrm{GeV}]",
            r"\phi_{W^-}",
            r"\eta_{W^-}",
            r"m_{W^-} \;[\mathrm{GeV}]",
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


class zmumuExperiment(EventGenerationExperiment):
    """
    Process: p p > z > mu+ mu- at reco-level with 0-5 extra jets
    """

    def __init__(self, cfg):
        self.plot_title = r"Z"
        self.n_hard_particles = 2
        self.onshell_list = [0, 1]
        self.onshell_mass = [0.1, 0.1]
        self.units = 258.1108
        self.delta_r_min = 0.40
        self.pt_min = [0.0] * 2 + [20.0] * 10
        self.obs_names_index = ["l1", "l2"]
        for ijet in range(cfg.data.n_jets_max):
            self.obs_names_index.append(f"j{ijet+1}")
        self.fourmomentum_ranges = [[0, 200], [-100, 100], [-100, 100], [-200, 200]]
        self.jetmomentum_ranges = [[10, 100], [-np.pi, np.pi], [-6, 6], [0, 20]]
        self.virtual_components = [[0, 1]]
        self.virtual_ranges = [
            [0, 300],
            [-np.pi, np.pi],
            [-6, 6],
            [70, 110],
        ]
        self.virtual_names = [
            r"p_{T,Z} \;[\mathrm{GeV}]",
            r"\phi_Z",
            r"\eta_Z",
            r"m_Z \;[\mathrm{GeV}]",
        ]

        super().__init__(cfg)
