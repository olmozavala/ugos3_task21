# For testing the dataset
import sys
sys.path.append("/unity/g2/jvelasco/gitraw/ugos3_task21/DA_Chlora") # Only for testing purposes
import os
import bisect
import pickle
import numpy as np
import xarray as xr
import torch
from os.path import join
from data_loader.loader_utils import *
import matplotlib.pyplot as plt
import re
import glob
import cv2
import pandas as pd

# Canonical variable order for stacking X in the cached pkl.
# IMPORTANT: indices in config refer to this ordering.
DEFAULT_INPUT_VARS = ["sst", "chlora", "ssh_track", "swot", "fused_ssh"]


def _resolve_selected_vars(*, all_input_vars, input_vars=None, selected_vars=None):
    """
    Resolve requested input variables to:
    - selected_names: list[str] in the canonical order of `all_input_vars`
    - selected_indices: list[int] indices into `all_input_vars`

    Priority:
    1) `input_vars` (names) if provided
    2) `selected_vars` (names or indices) if provided
    3) default: all_input_vars (no clipping)
    """
    if input_vars is not None:
        if not isinstance(input_vars, (list, tuple)) or not all(isinstance(v, str) for v in input_vars):
            raise ValueError(f"`input_vars` must be a list of strings, got: {type(input_vars)} -> {input_vars}")
        requested_names = list(input_vars)
    elif selected_vars is not None:
        if not isinstance(selected_vars, (list, tuple)):
            raise ValueError(f"`selected_vars` must be a list (of indices or names), got: {type(selected_vars)}")
        if len(selected_vars) == 0:
            raise ValueError("`selected_vars` cannot be empty.")
        if all(isinstance(v, int) for v in selected_vars):
            requested_names = []
            for idx in selected_vars:
                if idx < 0 or idx >= len(all_input_vars):
                    raise ValueError(
                        f"selected_vars index {idx} out of range for all_input_vars "
                        f"(len={len(all_input_vars)}): {all_input_vars}"
                    )
                requested_names.append(all_input_vars[idx])
        elif all(isinstance(v, str) for v in selected_vars):
            requested_names = list(selected_vars)
        else:
            raise ValueError(f"`selected_vars` must be all ints or all strings, got: {selected_vars}")
    else:
        requested_names = list(all_input_vars)

    unknown = [v for v in requested_names if v not in all_input_vars]
    if unknown:
        raise ValueError(f"Unknown input variables requested: {unknown}. Valid options: {all_input_vars}")

    # Keep canonical ordering to match the cached X stacking.
    selected_names = [v for v in all_input_vars if v in requested_names]
    selected_indices = [all_input_vars.index(v) for v in selected_names]
    return selected_names, selected_indices


def scale_data_dataset(data, scalers, name, training=True):
    """Apply StandardScaler to an array and persist the scaler."""
    reshaped_data = data.data.flatten()
    if training:
        min  = np.nanmin(reshaped_data).compute()
        max  = np.nanmax(reshaped_data).compute()
        mean = np.nanmean(reshaped_data).compute()
        std  = np.nanstd(reshaped_data).compute()
    else:
        print(f"Loading {name} scalers...")
        min  = scalers[name]['min']
        max  = scalers[name]['max']
        mean = scalers[name]['mean']
        std  = scalers[name]['std']

    print(f"For {name}: Min: {min}, Max: {max}, Mean: {mean}, Std: {std}")

    scaled_data = (reshaped_data - mean) / std
    scalers[name] = {'mean': mean, 'std': std, 'min': min, 'max': max}
    scaled_data = scaled_data.reshape(data.shape)
    scaled_data = np.where(np.isnan(data.data), np.nan, scaled_data)
    return scaled_data, scalers


def clean_gulf_mask(mask, min_size=500):
    mask_uint8 = (mask > 0).astype(np.uint8) * 255

    # Remove small islands in the land
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    cleaned_mask = np.zeros_like(mask_uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned_mask[labels == i] = 255

    # Fill small holes in the ocean
    inverted = cv2.bitwise_not(cleaned_mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)
    filled_mask = np.zeros_like(inverted)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            filled_mask[labels == i] = 255

    return cv2.bitwise_not(filled_mask) / 255.0


# ---------------------------------------------------------------------------
# Internal helper – load and preprocess one pkl segment into RAM
# ---------------------------------------------------------------------------

def _load_pkl_segment(
    pkl_path,
    selected_var_indices,
    new_height,
    new_width,
    patch_size,
    dataset_type,
    previous_days,
):
    """
    Load a single pkl file, select variables, replace NaNs, convert to tensors,
    and crop to the patch-aligned spatial dimensions.

    Parameters
    ----------
    new_height, new_width : int or None
        Target crop dimensions.  Pass ``None`` on the *first* call and they will
        be derived from the loaded data; the computed values are returned so every
        subsequent segment is cropped identically.

    Returns
    -------
    seg : dict  - keys: ``X`` (Tensor), ``Y`` (Tensor), ``time`` (ndarray),
                        ``length`` (int, number of valid samples in this segment)
    new_height, new_width : int
    """
    print(f"  Loading: {pkl_path}")
    with open(pkl_path, "rb") as f:
        X, Y, lats, lons, time = pickle.load(f)

    # Variable selection
    if X.shape[1] >= (max(selected_var_indices) + 1):
        X = X[:, selected_var_indices]

    # NaN → 0
    X = np.where(np.isnan(X), 0, X)
    Y = np.where(np.isnan(Y), 0, Y)

    # Convert to tensors
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    # Derive crop dimensions from the first segment if not yet set
    if new_height is None:
        new_height = (X.shape[2] // patch_size) * patch_size
        new_width  = (X.shape[3] // patch_size) * patch_size

    X = X[..., :new_height, :new_width]
    Y = Y[..., :new_height, :new_width]

    # Number of valid samples in this segment
    # (mirror the same end_cutoff logic as the original __init__)
    end_cutoff = 2 if dataset_type == "nemo_mdt" else 0
    length = Y.shape[0] - previous_days - end_cutoff

    if length <= 0:
        raise ValueError(
            f"Segment '{pkl_path}' has only {Y.shape[0]} time steps, which is "
            f"insufficient for previous_days={previous_days} and "
            f"dataset_type='{dataset_type}'."
        )

    seg = dict(X=X, Y=Y, time=time, length=length)
    return seg, new_height, new_width


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SimSatelliteDataset:
    """
    PyTorch-style dataset for satellite ocean data.

    Supports loading **one or multiple** temporally-disjoint ``.pkl`` segment
    files entirely into RAM.  Because there are large temporal gaps between
    segments, samples are never constructed across segment boundaries — each
    segment maintains its own ``previous_days`` window.

    Parameters
    ----------
    data_dir : str
        Root directory that contains the ``.pkl`` files and auxiliary files
        (``mdt_normalized.nc``, etc.).
    pkl_files : list[str] | None
        Explicit list of ``.pkl`` *filenames* (not full paths) to load.
        When provided the ``training`` flag is **ignored** for file selection.
        Example::

            pkl_files=["training_2018.pkl", "training_2019.pkl"]

        When ``None`` the original single-file behaviour is preserved
        (``training.pkl`` or ``validation.pkl`` selected by ``training``).
    training : bool
        Used only when ``pkl_files`` is ``None``.  Selects ``training.pkl``
        (True) or ``validation.pkl`` (False).

    All other parameters are identical to the original implementation.
    """

    # Total 1758*2 = 3516 training examples
    # 10% validation split -> 351 examples
    # 90% training split -> 3165 examples

    def __init__(
        self,
        data_dir,
        transform=None,
        previous_days=1,
        plot_data=False,
        training=True,
        dataset_type="regular",
        input_vars=None,
        selected_vars=None,
        all_input_vars=None,
        patch_size=8,
        pkl_files=None,         # ← NEW: list of filenames, or None for legacy mode
    ):
        self.data_dir     = data_dir
        self.transform    = transform
        self.scalers      = {}
        self.previous_days = previous_days
        self.plot_data    = plot_data
        self.dataset_type = dataset_type
        self.dt           = np.timedelta64(1, 'D')
        self.patch_size   = patch_size
        self.valid_start_idx = self.previous_days

        # ── variable resolution ──────────────────────────────────────────────
        self.all_input_vars = (
            list(all_input_vars) if all_input_vars is not None else list(DEFAULT_INPUT_VARS)
        )
        self.input_vars, self.selected_var_indices = _resolve_selected_vars(
            all_input_vars=self.all_input_vars,
            input_vars=input_vars,
            selected_vars=selected_vars,
        )

        # ── resolve segment file paths ───────────────────────────────────────
        if not training:
            pkl_files = None
        if pkl_files is not None:
            if not isinstance(pkl_files, (list, tuple)) or len(pkl_files) == 0:
                raise ValueError("`pkl_files` must be a non-empty list of filename strings.")
            segment_paths = [join(data_dir, f) for f in pkl_files]
        else:
            # Legacy single-file mode
            pkl_file = "training.pkl" if training else "validation.pkl"
            segment_paths = [join(data_dir, pkl_file)]

        # ── load all segments into RAM ───────────────────────────────────────
        print(f"Loading {len(segment_paths)} segment(s) into RAM ...")
        new_height, new_width = None, None
        self._segments = []

        for path in segment_paths:
            seg, new_height, new_width = _load_pkl_segment(
                pkl_path=path,
                selected_var_indices=self.selected_var_indices,
                new_height=new_height,
                new_width=new_width,
                patch_size=patch_size,
                dataset_type=dataset_type,
                previous_days=previous_days,
            )
            self._segments.append(seg)
            print(
                f"    → X={tuple(seg['X'].shape)}  Y={tuple(seg['Y'].shape)}  "
                f"valid_samples={seg['length']}"
            )

        # ── cumulative offsets for O(log n) index routing ────────────────────
        # _seg_offsets[i] is the first global index that belongs to segment i.
        lengths = [s['length'] for s in self._segments]
        self._seg_offsets = np.cumsum([0] + lengths[:-1]).tolist()
        self.length = sum(lengths)

        # ── shared spatial objects (derived once from the first segment) ─────
        # Re-read the first file to get lats/lons and the raw Y for the mask.
        # This is a small extra I/O hit but avoids keeping the raw arrays in
        # memory alongside the already-processed tensors.
        print(f"  Building shared spatial objects from {segment_paths[0]} ...")
        with open(segment_paths[0], "rb") as f:
            _X0, _Y0_raw, lats0, lons0, _ = pickle.load(f)
        del _X0

        self.lats = lats0[:new_height]
        self.lons = lons0[:new_width]

        # Gulf mask
        gulf_mask = np.where(~np.isnan(_Y0_raw[0, :new_height, :new_width]), 1, 0)
        gulf_mask[:100, :300] = 0
        #TODO: We need to add a masking for the erronious data points in all the ssh

        gulf_mask = cv2.erode(
            gulf_mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=3
        )
        gulf_mask = clean_gulf_mask(gulf_mask, min_size=1000).astype(np.uint8)

        self.gulf_mask = torch.tensor(gulf_mask, dtype=torch.float32)
        del _Y0_raw

        # TODO: Add logic to only load this data when the correct dataset_type is selected (background)
        # TODO: We also need logic to decide between Nemo MTD or actual MDT

        # MDT products are only needed for the nemo_mdt dataset type.
        if dataset_type == "nemo_mdt":
            # MDT
            self.mdt_normalized = xr.open_dataset(join(data_dir, "mdt_normalized.nc")).load()
            self.mdt_normalized.ssh.data = np.where(
                np.isnan(self.mdt_normalized.ssh.data), 0, self.mdt_normalized.ssh.data
            )
            self.mdt_normalized = self.mdt_normalized.isel(
                lat=slice(None, new_height),
                lon=slice(None, new_width),
            ).load()

            # MDT STD
            self.mdt_std_normalized = xr.open_dataset(join(data_dir, "mdt_std_normalized.nc")).load()
            self.mdt_std_normalized.ssh.data = np.where(
                np.isnan(self.mdt_std_normalized.ssh.data), 0, self.mdt_std_normalized.ssh.data
            )
            self.mdt_std_normalized = self.mdt_std_normalized.isel(
                lat=slice(None, new_height),
                lon=slice(None, new_width),
            ).load()
        else:
            self.mdt_normalized = None
            self.mdt_std_normalized = None

        # ── tot_inputs ───────────────────────────────────────────────────────
        n_ch = self._segments[0]['X'].shape[1]
        if dataset_type == "regular":
            self.tot_inputs = n_ch * self.previous_days + 1
        elif dataset_type in ("nemo_mdt", "gaussian_noise"):
            self.tot_inputs = n_ch * self.previous_days + 3
        elif dataset_type == "gaussian_noise_only":
            # Two previous SSH snapshots (with Gaussian noise) + gulf mask.
            self.tot_inputs = 3
        
        # TODO: Need to add logic to add std to the input tensor

        print(
            f"Dataset ready: {len(self._segments)} segment(s), "
            f"{self.length} total samples, "
            f"spatial={new_height}x{new_width}, "
            f"tot_inputs={self.tot_inputs}"
        )
        print("Preloading by the data loader is done!")

    # ── index routing ────────────────────────────────────────────────────────

    def _resolve_index(self, global_index):
        """Map a global index to the owning segment dict and the real array index."""
        # bisect_right gives the insertion point after the last offset ≤ global_index,
        # so subtracting 1 gives the segment that owns this index.
        seg_idx   = bisect.bisect_right(self._seg_offsets, global_index) - 1
        local_idx = global_index - self._seg_offsets[seg_idx]
        real_idx  = local_idx + self.valid_start_idx   # skip the leading previous_days rows
        return self._segments[seg_idx], real_idx

    # ── standard Dataset interface ───────────────────────────────────────────

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        seg, real_index = self._resolve_index(index)

        # Spatial dimensions
        height = seg['X'].shape[2]
        width  = seg['X'].shape[3]

        # Special case: gaussian_noise_only → only two noisy SSH snapshots + mask.
        if self.dataset_type == "gaussian_noise_only":
            X_with_mask = np.zeros((self.tot_inputs, height, width), dtype=np.float32)

            noise_level = 0.2
            noise_ssh = np.random.randn(self.lats.shape[0], self.lons.shape[0]) * noise_level

            # Channels: [SSH(t-1)+noise, SSH(t-2)+noise, gulf_mask]
            X_with_mask[0, :, :] = seg['Y'][real_index - 1] + noise_ssh
            X_with_mask[1, :, :] = seg['Y'][real_index - 2] + noise_ssh
            X_with_mask[2, :, :] = self.gulf_mask

            return X_with_mask, seg['Y'][real_index].unsqueeze(0)

        # Default path: include X variables for previous_days plus auxiliary channels.
        X_with_mask = np.zeros((self.tot_inputs, height, width), dtype=np.float32)

        size_per_day = seg['X'].shape[1]
        for i in range(self.previous_days):
            start_idx = i * size_per_day
            end_idx   = start_idx + size_per_day
            X_with_mask[start_idx:end_idx, :, :] = (
                seg['X'][real_index - self.previous_days + i + 1, :, :, :]
            )

        # Gulf mask is always the last channel
        X_with_mask[-1, :, :] = self.gulf_mask

        if self.dataset_type == "nemo_mdt":
            time_index = np.datetime64(seg['time'][real_index])
            t1   = time_index - self.dt
            t2   = t1 - self.dt
            doy1 = int(pd.Timestamp(t1).day_of_year)
            doy2 = int(pd.Timestamp(t2).day_of_year)
            X_with_mask[-2, :, :] = self.mdt_normalized.ssh.sel(dayofyear=doy1).data
            X_with_mask[-3, :, :] = self.mdt_normalized.ssh.sel(dayofyear=doy2).data

        if self.dataset_type == "gaussian_noise":
            noise_level = 0.2
            noise_ssh = np.random.randn(self.lats.shape[0], self.lons.shape[0]) * noise_level
            X_with_mask[-2, :, :] = seg['Y'][real_index - 1] + noise_ssh
            X_with_mask[-3, :, :] = seg['Y'][real_index - 2] + noise_ssh

        # TODO: Need to add logic to add std to the input tensor, first we need to decide the tensor configuration

        return X_with_mask, seg['Y'][real_index].unsqueeze(0)

    # ── misc helpers ─────────────────────────────────────────────────────────

    def get_scaler(self):
        return self.scaler

    def normalize(self, x):
        return self.scaler.transform(x)

    def denormalize(self, x):
        return self.scaler.inverse_transform(x)

    def get_coords(self):
        """
        Return (lats, lons, times) where *times* is the concatenation of the
        valid time slices from every segment (i.e. skipping the leading
        ``previous_days`` rows that can never be returned as samples).
        """
        all_times = np.concatenate([
            seg['time'][self.valid_start_idx : self.valid_start_idx + seg['length']]
            for seg in self._segments
        ])
        return self.lats, self.lons, all_times


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data_dir      = "/unity/g2/jvelasco/dataset/ugos/datasets_v2/"
    batch_size    = 64
    training      = False
    plot_data     = False
    previous_days = 7
    dataset_type  = "nemo_mdt"
    shuffle       = True
    input_vars    = ["fused_ssh"]
    # ── single-file mode (original behaviour, unchanged) ────────────────────
    dataset_single = SimSatelliteDataset(
        data_dir, previous_days=previous_days, transform=None,
        plot_data=plot_data, training=training,
        dataset_type=dataset_type, input_vars=input_vars,
    )
    print(f"Single-file dataset length: {len(dataset_single)}")

    # ── multi-file mode ──────────────────────────────────────────────────────
    dataset_multi = SimSatelliteDataset(
        data_dir, previous_days=previous_days, transform=None,
        plot_data=plot_data, training=training,
        dataset_type=dataset_type, input_vars=input_vars,
        pkl_files=["training.pkl", "training_p2.pkl", "training_p3.pkl"],
    )
    print(f"Multi-file dataset length: {len(dataset_multi)}")

    loader = torch.utils.data.DataLoader(dataset_multi, batch_size=batch_size, shuffle=shuffle)
    for batch_idx, (x, y) in enumerate(loader):
        print(f"Batch {batch_idx}: x.shape={x.shape}  y.shape={y.shape}")
        if batch_idx > 0:
            break

    import torch
    import torch.nn.functional as F

    def compute_derivative_variances(dataloader):
        """
        Compute dataset-level variances of first and second derivatives
        of SSH, to be used as fixed normalizers in the loss.
        Batches from the loader are (inputs, target) with inputs (B, C, H, W)
        and target (B, 1, H, W); the last channel of inputs is the gulf mask.
        """
        _SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32, requires_grad=False).view(1, 1, 3, 3)
        _SOBEL_Y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32, requires_grad=False).view(1, 1, 3, 3)
        _LAPLACE = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32, requires_grad=False).view(1, 1, 3, 3)

        kx = _SOBEL_X
        ky = _SOBEL_Y
        k_lap = _LAPLACE

        sum_grad_sq = 0.0
        sum_curv_sq = 0.0
        n = 0

        for inputs, eta in dataloader:
            # eta: (B, 1, H, W), mask from last channel of inputs: (B, H, W)
            mask = inputs[:, -1, :, :]
            grad_x = F.conv2d(eta, kx, padding=1)
            grad_y = F.conv2d(eta, ky, padding=1)
            curv = F.conv2d(eta, k_lap, padding=1)

            valid = mask.bool()  # (B, H, W)
            valid_3d = valid.unsqueeze(1)  # (B, 1, H, W) for indexing grad/curv
            sum_grad_sq += (grad_x[valid_3d].pow(2) + grad_y[valid_3d].pow(2)).sum().item()
            sum_curv_sq += curv[valid_3d].pow(2).sum().item()
            n += valid.sum().item()

        if n == 0:
            return float("nan"), float("nan")
        var_grad = sum_grad_sq / n
        var_curv = sum_curv_sq / n
        return var_grad, var_curv

    var_grad, var_curv = compute_derivative_variances(loader)
    print(f"Variance of first derivatives: {var_grad}")
    print(f"Variance of second derivatives: {var_curv}")

    print("Done!")
