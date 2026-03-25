import numpy as np
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class SubsetSequentialSampler(Sampler):
    """Yields subset indices in fixed (sorted) order — no shuffle.

    Used for the validation DataLoader so that batch positions map
    deterministically to dataset indices, enabling date-aware metrics
    (e.g. seasonal RMSE) without modifying __getitem__.
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate):
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        if self.shuffle:
            train_sampler = SubsetRandomSampler(train_idx)
        else:
            # Sequential train sampler: batch position maps deterministically
            # to a dataset index, enabling date-aware metrics (e.g. seasonal RMSE).
            train_idx = np.sort(train_idx)
            train_sampler = SubsetSequentialSampler(train_idx)
            self._iter_indices = train_idx  # indices iterated by this DataLoader, in order

        # Always sort validation holdout indices (used by split_validation())
        valid_idx = np.sort(valid_idx)
        valid_sampler = SubsetSequentialSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
