import torch
from torch import nn

from src.models.base_generator import BaseGenerator


class JetGPT2(BaseGenerator):
    def __init__(self, cfg, warm_start, device, dtype):
        super().__init__(cfg, warm_start, device, dtype)

    def forward(self, x, context):
        raise NotImplementedError

    def log_prob(self, x, context):
        raise NotImplementedError

    def sample(self, num_samples, context):
        raise NotImplementedError
