import torch
from torch.utils.data import Dataset


class EventDataset(Dataset):
    def __init__(self, events_list):
        super().__init__()

        # zero pad all events into shape (max_particles, 4)
        self.events = []
        self.num_particles = []
        max_particles = max([x.shape[-2] for x in events_list])
        for events in events_list:
            num_particles = events.shape[1] * torch.ones(
                events.shape[0], dtype=torch.long, device=events.device
            )
            self.num_particles.append(num_particles)
            events_buffer = events.clone()
            events = torch.zeros(
                events.shape[0],
                max_particles,
                4,
                dtype=events.dtype,
                device=events.device,
            )
            events[:, : events_buffer.shape[1]] = events_buffer
            self.events.append(events)
        self.events = torch.cat(self.events)
        self.num_particles = torch.cat(self.num_particles)

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        return self.events[idx], self.num_particles[idx]
