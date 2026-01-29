import math
import torch
from torch.utils.data import Sampler


class ShardEpochSampler(Sampler):
    """Samples a moving shard (window) of indices per epoch.

    - Only materializes indices for the current shard to keep memory bounded.
    - Shard rotates with epoch: epoch % num_shards.
    - Within a shard, indices are shuffled deterministically via seed+epoch.
    """

    def __init__(self, data_len: int, shard_size: int, seed: int = 0):
        if shard_size <= 0:
            raise ValueError("shard_size must be > 0")
        self.data_len = int(data_len)
        self.shard_size = int(min(shard_size, data_len))
        self.num_shards = int(math.ceil(self.data_len / float(self.shard_size)))
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        # Pick shard for this epoch
        shard_idx = self.epoch % self.num_shards
        start = shard_idx * self.shard_size
        end = min(start + self.shard_size, self.data_len)
        length = end - start

        # Shuffle within shard deterministically
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        local = torch.randperm(length, generator=g)
        idxs = (local + start).tolist()
        return iter(idxs)

    def __len__(self):
        # Length of the current shard (last shard may be shorter).
        shard_idx = self.epoch % self.num_shards
        start = shard_idx * self.shard_size
        end = min(start + self.shard_size, self.data_len)
        return end - start

