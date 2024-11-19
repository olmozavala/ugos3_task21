# For testing the dataset
import sys
#sys.path.append("/unity/f1/ozavala/CODE/ugos3_task21/DA_Chlora") # Only for testing purposes
sys.path.append("/unity/g2/jvelasco/github/ugos3_task21/DA_Chlora") # Only for testing purposes
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
    def __init__(self, data_dir, transform=None, previous_days=1, plot_data=False, training=True, dataset_type="regular"):
        self.data_dir = data_dir
        self.transform = transform
        self.scalers = {}  # To store scalers for each variable
        self.previous_days = previous_days
        self.plot_data = plot_data
        self.dataset_type = dataset_type
        # Input variables
        input_vars = ["sst", "chlora", "ssh_track", "swot"]
        output_vars = ["ssh"]
        all_var_names = input_vars + output_vars
        input_normalized_vars = [f"{var}_normalized" for var in input_vars]
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
        if not os.path.exists(training_pkl_path):
            print(f"{pkl_file} file does not exist. Reading netcdfs...")
            # Reads all the netcdfs in the data_dir
            all_files = glob.glob(join(data_dir, "*.nc"))
            pattern = re.compile(r".*_(\d+)\.nc$")
            if training:
                # In this case we should read 0 to 1582 and 1759 to 3340
                if pkl_file == "training.pkl":
                    filtered_files = [f for f in all_files if pattern.match(f) and (
                        0 <= int(pattern.match(f).group(1)) <= 1582 or
                    1759 <= int(pattern.match(f).group(1)) <= 3340)]
                elif pkl_file == "training_small.pkl":
                    filtered_files = [f for f in all_files if pattern.match(f) and (
                        int(pattern.match(f).group(1)) <= 100)]
                elif pkl_file == "training_full.pkl":
                    # Use all the files
                    filtered_files = all_files
            else:
                filtered_files = [f for f in all_files if pattern.match(f) and (
                    1583 <= int(pattern.match(f).group(1)) <= 1758 or
                    3340 <= int(pattern.match(f).group(1)) <= 3515
                )]

            # Sort the files
            filtered_files.sort()

            all_data = xr.open_mfdataset(filtered_files, engine="netcdf4", concat_dim="ex_num", combine="nested")
            
            # Rechunk the data so that ex_num is in a single chunk
            all_data = all_data.chunk({'ex_num': 1})

            lats = all_data.latitude
            lons = all_data.longitude

            self.lats = lats
            self.lons = lons

            # Apply log to the chlorophyll variable
            all_data['chlora'] = np.log(all_data['chlora'])
            # Make nans all the values that are 0s in ssh_track
            all_data['ssh_track'].data = np.where(all_data['ssh_track'].data==0, np.nan, all_data['ssh_track'].data)

            # If we are not training read the scalers from the file
            if not training: 
                print(f"Reading {scalers_file} file...")
                with open(join(data_dir, scalers_file), "rb") as f:
                    self.scalers = pickle.load(f)

            # Apply the scaling function to each variable
            for var_name in all_var_names:
                print(f"Scaling {var_name}...")
                norm_data, self.scalers = scale_data_dataset(all_data[var_name], self.scalers, var_name, training=training)
                all_data[f'{var_name}_normalized'] = xr.DataArray(
                    norm_data,
                    dims=all_data[var_name].dims,
                    coords=all_data[var_name].coords
                )

            if training:
                # Saving the scalers
                with open(join(data_dir, scalers_file), "wb") as f:
                    print(f"Saving scalers to {join(data_dir, scalers_file)}...")
                    pickle.dump(self.scalers, f)
                    print("Scalers saved!")

            if self.plot_data:
                print("Plotting some data...")
                for i in range(10):
                    idx = np.random.randint(0, len(all_data.ex_num))
                    plot_dataset_data(idx, all_data, lats, lons)
             
            print("Stack the data and assign to X and Y")
            self.X = np.stack([all_data[var_name].compute().data for var_name in input_normalized_vars], axis=0)
            self.Y = all_data[output_var].compute().data
            # Flip the first and second dimensions in X  
            self.X = np.transpose(self.X, (1, 0, 2, 3))

            print("Saving the training data...")
            # Saving the training data
            with open(training_pkl_path, "wb") as f:
                pickle.dump((self.X, self.Y, self.lats, self.lons), f)
            print("Training data saved!")
        else:
            print(f"Reading {pkl_file} file...")
            with open(training_pkl_path, "rb") as f:
                self.X, self.Y, self.lats, self.lons = pickle.load(f)
        
        # Make a mask of the gulf of guinea
        self.gulf_mask = np.zeros_like(self.Y[0,:,:])
        # Create a mask for the Gulf of Guinea
        self.gulf_mask = np.where(~np.isnan(self.Y[0,:,:]), 1, 0)
        # Replace all the nan values in X and Y with 0s
        self.X = np.where(np.isnan(self.X), 0, self.X)
        self.Y = np.where(np.isnan(self.Y), 0, self.Y)

        #  Make tensors
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.Y = torch.tensor(self.Y, dtype=torch.float32)

        # Crop the last two dimensions to the largest dimension divisible by 8
        new_height = (self.X.shape[2] // 8) * 8
        new_width = (self.X.shape[3] // 8) * 8
        self.X = self.X[:, :, :new_height, :new_width]
        self.Y = self.Y[:, :new_height, :new_width]
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

        # If the dataset is gradient, then we append a second output with the magnitude of the gradient of the ssh
        # if dataset_type == "gradient":
        #    print("Calculating the gradient of the SSH...")
        #    all_grad_magitudes = []
        #    for i in range(self.Y.shape[0]):
        #        gradient_y, gradient_x = np.gradient(self.Y[i, :, :])
        #        gradient_magnitude = np.sqrt(gradient_y**2 + gradient_x**2)
        #        all_grad_magitudes.append(gradient_magnitude)

        #    all_grad_magitudes = np.array(all_grad_magitudes)
        #    # Normalize to mean 0 and std 1
        #    all_grad_magitudes = (all_grad_magitudes - np.mean(all_grad_magitudes)) / np.std(all_grad_magitudes)
        #    self.Y = np.stack([self.Y, all_grad_magitudes], axis=0)
        #    # Flip the first and second dimensions in Y
        #    self.Y = np.transpose(self.Y, (1, 0, 2, 3))

        # Make the mask a float32 tensor
        self.gulf_mask = torch.tensor(self.gulf_mask, dtype=torch.float32)

        # Get the length of the dataset
        self.length = self.Y.shape[0]
        if self.dataset_type == "extended":
            self.length = self.length - 2
        if self.dataset_type == "gradient":
            self.length = self.length - 2

        # # Verify the dimensions
        print(f"X shape: {self.X.shape}")
        print(f"Y shape: {self.Y.shape}")
        print("Preloading by the data loader is done!")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Append the Gulf Mask to the X array
        if index <= self.previous_days:
            index = self.previous_days + 1

        X_with_mask = np.zeros((self.tot_inputs, self.X.shape[2], self.X.shape[3]), dtype=np.float32)

        # The +1 is because the last element of X_with_mask is the Gulf Mask
        size_per_day = self.X.shape[1]
        for i in range(self.previous_days):
            # Append the previous days to the X_with_mask
            start_index = i * size_per_day
            end_idx = (i * size_per_day) + size_per_day
            # print(f"start_index: {start_index}, end_idx: {end_idx}. Index: {index - self.previous_days + i + 1}. Original index: {index}")
            X_with_mask[start_index : end_idx, :, :] = self.X[index - self.previous_days + i + 1, :, :, :]
        
        # Add the Gulf Mask as the last channel
        X_with_mask[-1, :, :] = self.gulf_mask

        if self.dataset_type == "extended":
            noise_level = 0.5  # Default is 0.5 low is 0.1
            noise = np.random.randn(self.Y.shape[1],self.Y.shape[2]) * noise_level
            # Add the previous two states with some noise at locations -2 and -3
            X_with_mask[-2, :, :] = self.Y[index-1, :, :] + noise
            X_with_mask[-3, :, :] = self.Y[index-2, :, :] + noise

        if self.dataset_type == "gradient":
            #noise_level_ssh = 0.2
            #noise_ssh = np.random.randn(self.Y.shape[1],self.Y.shape[2]) * noise_level_ssh
            # Add the previous two states with some noise and its gradient
            # X_with_mask[-2, :, :] = self.Y[index-1, :, :] + noise_ssh
            # X_with_mask[-3, :, :] = self.Y[index-2, :, :] + noise_ssh
            X_with_mask[-2, :, :] = groundto2background(self.Y[index-1, :, :]) 
            X_with_mask[-3, :, :] = groundto2background(self.Y[index-2, :, :]) 



        # Only for testing purposes plot the input data
        if self.plot_data:
            input_names = ["sst", "chlora", "ssh_track", "swot"]
            plot_single_batch_element(X_with_mask, self.Y[index], input_names, self.previous_days, 
                                      #f"/unity/f1/ozavala/OUTPUTS/HR_SSH_from_Chlora/trainings/batch_example_{index}.jpg",
                                      f"/unity/g2/jvelasco/ai_outs/task21_set1/higos/batch_example_{index}.jpg",
                                      self.lats, self.lons, dataset_type=self.dataset_type)

        return X_with_mask, self.Y[index]

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

    # Create an instance of the SimSatelliteDataset
    dataset = SimSatelliteDataset(data_dir, previous_days=previous_days, transform=None,
                                   plot_data=plot_data, training=training, dataset_type=dataset_type)

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