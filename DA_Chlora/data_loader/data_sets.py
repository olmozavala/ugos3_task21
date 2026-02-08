# For testing the dataset
import sys
#sys.path.append("/unity/f1/ozavala/CODE/ugos3_task21/DA_Chlora") # Only for testing purposes
sys.path.append("/unity/g2/jvelasco/gitraw/ugos3_task21/DA_Chlora") # Only for testing purposes
import os
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
                        f"selected_vars index {idx} out of range for all_input_vars (len={len(all_input_vars)}): {all_input_vars}"
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

# Function to apply StandardScaler to an array and persist the scaler
def scale_data_dataset(data, scalers, name, training=True):
    # Flatten the data to 2D, where each row is a sample
    reshaped_data = data.data.flatten()
    if training:
        min = np.nanmin(reshaped_data).compute()
        max = np.nanmax(reshaped_data).compute()
        mean = np.nanmean(reshaped_data).compute()
        std = np.nanstd(reshaped_data).compute()
    else:
        print(f"Loading {name} scalers...")
        min = scalers[name]['min']
        max = scalers[name]['max']
        mean = scalers[name]['mean']
        std = scalers[name]['std']

    print(f"For {name}: Min: {min}, Max: {max}, Mean: {mean}, Std: {std}")

    scaled_data = (reshaped_data - mean) / std
    
    # Save the scaler for later use
    scalers[name] = {'mean': mean, 'std': std, 'min': min, 'max': max}

    # Reshape the scaled data back to its original shape
    scaled_data = scaled_data.reshape(data.shape)
    # Any nan values in the original data should be nan in the scaled data
    scaled_data = np.where(np.isnan(data.data), np.nan, scaled_data)
    return scaled_data, scalers


def clean_gulf_mask(mask, min_size=500):
    # Ensure mask is uint8 binary (0 and 255)
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    
    # 1. Remove small yellow "islands" in the land
    # Find all connected components of valid pixels
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    
    cleaned_mask = np.zeros_like(mask_uint8)
    for i in range(1, num_labels): # Skip background (label 0)
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned_mask[labels == i] = 255
            
    # 2. Fill small purple "pockmarks" (holes) in the ocean
    # Invert, remove small components, invert back
    inverted = cv2.bitwise_not(cleaned_mask)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted, connectivity=8)
    
    filled_mask = np.zeros_like(inverted)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            filled_mask[labels == i] = 255
            
    return cv2.bitwise_not(filled_mask) / 255.0
# %% Simulate DUACs background field
def groundto2background(data, lat=(14.18613, 30.61901), lon=(-89.33899, -78.666664), 
                        x=712, y=648, resolution=0.25):
    downsampled_lats = np.arange(lat[0], lat[1], resolution)
    downsampled_lons = np.arange(lon[0], lon[1], resolution)
    upsampled_lats = np.linspace(lat[0], lat[1], y)
    upsampled_lons = np.linspace(lon[0], lon[1], x)

    ds = xr.Dataset({'ssh': (['latitude', 'longitude'], data)},
                    coords={'latitude': ('latitude', upsampled_lats),
                            'longitude': ('longitude', upsampled_lons)})
    ds = ds.interp(
        latitude=downsampled_lats, 
        longitude=downsampled_lons, 
        method='linear').interp(
            latitude=upsampled_lats, 
            longitude=upsampled_lons, 
            method='linear')
    ds.ssh.data = np.where(np.isnan(ds.ssh.data), 0, ds.ssh.data)
    return ds.ssh.data

class SimSatelliteDataset:
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
    ):
        self.data_dir = data_dir
        self.transform = transform
        self.scalers = {}  # To store scalers for each variable
        self.previous_days = previous_days
        self.plot_data = plot_data
        self.dataset_type = dataset_type

        self.valid_start_idx = self.previous_days

        # Input variables (canonical order used for stacking cached X)
        self.all_input_vars = list(all_input_vars) if all_input_vars is not None else list(DEFAULT_INPUT_VARS)
        # Var order {0:sst, 1:LOG(CHLORA), 2:ssh_track, 3:swot, 4:fused_ssh}
        self.input_vars, self.selected_var_indices = _resolve_selected_vars(
            all_input_vars=self.all_input_vars,
            input_vars=input_vars,
            selected_vars=selected_vars,
        )

        output_vars = ["ssh"]
        all_var_names = self.all_input_vars + output_vars
        # We always normalize/stack in canonical order for caching; selection happens after load.
        input_normalized_vars = [f"{var}_normalized" for var in self.all_input_vars]
        output_var = [f"{var}_normalized" for var in output_vars][0]

        scalers_file = "scalers.pkl"
        # DO not delete this section it is used to select the input dataset as the computation takes some time 
        if training:
            pkl_file = "training.pkl"
            # pkl_file = "training_full.pkl"
            # pkl_file = "training_small.pkl"
        else:
            pkl_file = "validation.pkl"

        # Verify if 'training.pkl' file exists
        training_pkl_path = join(data_dir, pkl_file)

        print(f"Reading {pkl_file} file...")
        with open(training_pkl_path, "rb") as f:
            X, self.Y, self.lats, self.lons = pickle.load(f)
        # Clip channels based on config selection, if the cached X has those channels.
        if X.shape[1] >= (max(self.selected_var_indices) + 1):
            X = X[:, self.selected_var_indices]
        self.X = X
        # assert self.X.shape[1] == 2, f"self.X.shape: {self.X.shape}"

        # Make a mask of the gulf of Mexico
        self.gulf_mask = np.zeros_like(self.Y[0,:,:])
        # Create a mask for the Gulf of Mexico
        self.gulf_mask = np.where(~np.isnan(self.Y[0,:,:]), 1, 0)
        # Remove Pacific Ocean region
        self.gulf_mask[:100, :300] = 0

        # Erode the mask
        self.gulf_mask = cv2.erode(self.gulf_mask.astype(np.uint8), np.ones((3,3), dtype=np.uint8), iterations=3)
        self.gulf_mask = clean_gulf_mask(self.gulf_mask, min_size=1000).astype(np.uint8)

        # Replace all the nan values in X and Y with 0s
        self.X = np.where(np.isnan(self.X), 0, self.X)
        self.Y = np.where(np.isnan(self.Y), 0, self.Y)

        #  Make tensors
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.Y = torch.tensor(self.Y, dtype=torch.float32)

        # Crop the last two dimensions to the largest dimension divisible by 8
        new_height = (self.X.shape[2] // 8) * 8
        new_width = (self.X.shape[3] // 8) * 8
        self.X = self.X[..., :new_height, :new_width]
        self.Y = self.Y[..., :new_height, :new_width]
        self.gulf_mask = self.gulf_mask[:new_height, :new_width]
        
        if dataset_type == "regular":
            # +1 because of the Gulf Mask
            self.tot_inputs = self.X.shape[1] * self.previous_days + 1
        elif dataset_type == "extended":
            # +3 because of the Gulf Mask and the two previous states with some noise
            self.tot_inputs = self.X.shape[1] * self.previous_days + 3
        elif dataset_type == "gradient":
            # + 5 because of the Gulf Mask, and the two previous states with some noise and the gradient (2 * 2)
            self.tot_inputs = self.X.shape[1] * self.previous_days + 3

        # Make the mask a float32 tensor
        self.gulf_mask = torch.tensor(self.gulf_mask, dtype=torch.float32)

        # Get the length of the dataset
        end_cutoff = 0
        if self.dataset_type == "extended":
            end_cutoff = 2
        if self.dataset_type == "gradient":
            end_cutoff = 2

        # 3. Calculate effective length
        # Total available - start offset - end cutoff
        self.length = self.Y.shape[0] - self.valid_start_idx - end_cutoff

        # # Verify the dimensions
        print(f"X shape: {self.X.shape}")
        print(f"Y shape: {self.Y.shape}")
        print("Preloading by the data loader is done!")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Append the Gulf Mask to the X array
        real_index = index + self.valid_start_idx

        X_with_mask = np.zeros((self.tot_inputs, self.X.shape[2], self.X.shape[3]), dtype=np.float32)

        # The +1 is because the last element of X_with_mask is the Gulf Mask
        size_per_day = self.X.shape[1]
        for i in range(self.previous_days):
            # Append the previous days to the X_with_mask
            start_index = i * size_per_day
            end_idx = (i * size_per_day) + size_per_day
            # print(f"start_index: {start_index}, end_idx: {end_idx}. Index: {index - self.previous_days + i + 1}. Original index: {index}")
            X_with_mask[start_index : end_idx, :, :] = self.X[real_index - self.previous_days + i + 1, :, :, :]
        
        # Add the Gulf Mask as the last channel
        X_with_mask[-1, :, :] = self.gulf_mask

        if self.dataset_type == "extended":
            noise_level = 0.2  # Default is 0.5 low is 0.1
            noise = np.random.randn(self.Y.shape[1],self.Y.shape[2]) * noise_level
            # Add the previous two states with some noise at locations -2 and -3
            X_with_mask[-2, :, :] = self.Y[real_index-1, :, :] + noise
            X_with_mask[-3, :, :] = self.Y[real_index-2, :, :] + noise

        if self.dataset_type == "gradient":
            noise_level_ssh = 0.2
            noise_ssh = np.random.randn(self.Y.shape[1],self.Y.shape[2]) * noise_level_ssh
            # Add the previous two states with some noise and its gradient
            X_with_mask[-2, :, :] = self.Y[real_index-1, :, :] + noise_ssh
            X_with_mask[-3, :, :] = self.Y[real_index-2, :, :] + noise_ssh


        # Only for testing purposes plot the input data
        if self.plot_data:
            input_names = ["chl", "ssh_track", "swot"]
            plot_single_batch_element(X_with_mask, self.Y[index], input_names, self.previous_days, 
                                      #f"/unity/f1/ozavala/OUTPUTS/HR_SSH_from_Chlora/trainings/batch_example_{index}.jpg",
                                      f"/unity/g2/jvelasco/ai_outs/task21_set1/higos/batch_example_{index}.jpg",
                                      self.lats, self.lons, dataset_type=self.dataset_type)

        return X_with_mask, self.Y[real_index].unsqueeze(0)

    def get_scaler(self):
        return self.scaler
    
    def normalize(self, x):
        return self.scaler.transform(x)

    def denormalize(self, x):
        return self.scaler.inverse_transform(x)


if __name__ == "__main__":
# Main function to test the dataset
    data_dir = "/Net/work/ozavala/OUTPUTS/HR_SSH_from_Chlora/training_data"
    batch_size = 1
    training = False
    plot_data = True
    previous_days = 7
    dataset_type = "gradient"
    shuffle = True
    input_vars = ["fused_ssh"]

    # Create an instance of the SimSatelliteDataset
    dataset = SimSatelliteDataset(data_dir, previous_days=previous_days, transform=None,
                                   plot_data=plot_data, training=training, dataset_type=dataset_type, input_vars=input_vars)

    # Create a data loader for the dataset
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
#
    # Iterate over the data loader
    for batch_idx, (x, y) in enumerate(data_loader):
        # Print the batch index and the batch size
        print(f"Batch {batch_idx}: x.shape = {x.shape}, y.shape = {y.shape}")
        if batch_idx > 0:
            break

    print("Done!")