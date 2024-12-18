import torch
import numpy as np


class JointGeneratorDataset(torch.utils.data.Dataset):
    def __init__(self, data, log_probs=None, weights=None):
        self.data = data
        if log_probs is not None or weights is not None:
            self.discformer_mode = True
            assert log_probs is not None and weights is not None
            assert np.all(
                [
                    dat.shape[0] == log_prob.shape[0]
                    and dat.shape[0] == weight.shape[0]
                    for dat, log_prob, weight in zip(data, log_probs, weights)
                ]
            )
            self.log_probs = log_probs
            self.weights = weights
        else:
            self.discformer_mode = False
        self.length = min([len(data) for data in self.data])

        self.data_eff = [data[: self.length, ...] for data in self.data]
        if self.discformer_mode:
            self.log_probs_eff = [
                log_prob[: self.length] for log_prob in self.log_probs
            ]
            self.weights_eff = [weight[: self.length] for weight in self.weights]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.discformer_mode:
            return [
                [data_eff[idx], log_prob[idx], weight[idx]]
                for data_eff, log_prob, weight in zip(
                    self.data_eff, self.log_probs_eff, self.weights_eff
                )
            ]
        else:
            return [[data_eff[idx]] for data_eff in self.data_eff]


class JointGeneratorDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False):
        super().__init__(
            dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
        )

    def __iter__(self):
        # reshuffle data_eff
        perms = [
            torch.randperm(len(data))[: self.dataset.length]
            for data in self.dataset.data
        ]
        self.dataset.data_eff = [
            data[perm] for data, perm in zip(self.dataset.data, perms)
        ]
        if self.dataset.discformer_mode:
            self.dataset.log_probs_eff = [
                log_prob[perm] for log_prob, perm in zip(self.dataset.log_probs, perms)
            ]
            self.dataset.weights_eff = [
                weight[perm] for weight, perm in zip(self.dataset.weights, perms)
            ]

        # return mother iterator
        return super().__iter__()


class JointClassifierDataset(torch.utils.data.Dataset):
    # this is written for use with a TransformerClassifier
    # can also be used with a MLP, but is less meaningful
    def __init__(self, data_true, data_fake):
        self.data_true = data_true
        self.data_fake = data_fake
        self.length = min(
            [len(data) for data in data_true] + [len(data) for data in data_fake]
        )

        self.data_true_eff = [data[: self.length, ...] for data in data_true]
        self.data_fake_eff = [data[: self.length, ...] for data in data_fake]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data = [
            [data_true[idx], data_fake[idx]]
            for data_true, data_fake in zip(self.data_true_eff, self.data_fake_eff)
        ]
        return data


class JointClassifierDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False):
        super().__init__(
            dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
        )

    def __iter__(self):
        # reshuffle data_true_eff and data_fake_eff
        perms_true = [
            torch.randperm(len(data))[: self.dataset.length]
            for data in self.dataset.data_true
        ]
        perms_fake = [
            torch.randperm(len(data))[: self.dataset.length]
            for data in self.dataset.data_fake
        ]
        self.dataset.data_true_eff = [
            data[perm] for (data, perm) in zip(self.dataset.data_true, perms_true)
        ]
        self.dataset.data_fake_eff = [
            data[perm] for (data, perm) in zip(self.dataset.data_fake, perms_fake)
        ]

        # return mother iterator
        return super().__iter__()
