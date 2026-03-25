from base import BaseDataLoader
from data_loader.data_sets import SimSatelliteDataset


class DefaultDataLoader(BaseDataLoader):
    def __init__(
        self,
        data_dir,
        batch_size,
        shuffle=True,
        validation_split=0.0,
        num_workers=1,
        training=True,
        previous_days=1,
        horizon_days=1,
        dataset_type="regular",
        input_vars=None,
        selected_vars=None,
        all_input_vars=None,
        patch_size=8,
        pkl_files=None,         # ← NEW: forwarded to SimSatelliteDataset
    ):
        self.data_dir = data_dir
        if not training:
            # Validation never needs shuffling; sequential order enables
            # date-to-batch mapping for metrics like seasonal RMSE.
            shuffle = False
        self.dataset = SimSatelliteDataset(
            self.data_dir,
            transform=None,
            previous_days=previous_days,
            training=training,
            dataset_type=dataset_type,
            input_vars=input_vars,
            selected_vars=selected_vars,
            all_input_vars=all_input_vars,
            patch_size=patch_size,
            pkl_files=pkl_files,    # ← forwarded
        )
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

    @property
    def iter_indices(self):
        """Dataset indices actually iterated by this DataLoader, in order.

        Only set when shuffle=False and validation_split > 0. Returns None when
        the full dataset is iterated without a split (all indices 0..len-1 in order).
        """
        return getattr(self, '_iter_indices', None)
