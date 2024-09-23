# For testing the dataset
import sys
sys.path.append("/unity/f1/ozavala/CODE/ugos3_task21/DA_Chlora") # Only for testing purposes
import os
import pickle
import numpy as np
import xarray as xr
import torch
from os.path import join
from data_loader.loader_utils import *
import matplotlib.pyplot as plt
# Function to apply StandardScaler to an array and persist the scaler
def scale_data_dataset(data, scalers, name):
    # Flatten the data to 2D, where each row is a sample
    reshaped_data = data.data.flatten()
    min = np.nanmin(reshaped_data).compute()
    max = np.nanmax(reshaped_data).compute()
    mean = np.nanmean(reshaped_data).compute()
    std = np.nanstd(reshaped_data).compute()
    print(f"For {name}: Min: {min}, Max: {max}, Mean: {mean}, Std: {std}")

    scaled_data = (reshaped_data - mean) / std
    
    # Save the scaler for later use
    scalers[name] = {'mean': mean, 'std': std}
    # Reshape the scaled data back to its original shape
    scaled_data = scaled_data.reshape(data.shape)
    # Any nan values in the original data should be nan in the scaled data
    scaled_data = np.where(np.isnan(data.data), np.nan, scaled_data)
    return scaled_data, scalers

class SimSatelliteDataset:
    def __init__(self, data_dir, transform=None, previous_days=1, plot_data=False, training=True):
        self.data_dir = data_dir
        self.transform = transform
        self.scalers = {}  # To store scalers for each variable
        self.previous_days = previous_days
        self.plot_data = plot_data
        # Input variables
        input_vars = ["sst", "chlora", "ssh_track", "swot"]
        output_vars = ["ssh"]
        all_var_names = input_vars + output_vars
        input_normalized_vars = [f"{var}_normalized" for var in input_vars]
        output_var = [f"{var}_normalized" for var in output_vars][0]

        if training:
            pkl_file = "training_full.pkl"
        # pkl_file = "training.pkl"
        else:
            pkl_file = "validation.pkl"

        # Verify if 'training.pkl' file exists
        training_pkl_path = join(data_dir, pkl_file)
        if not os.path.exists(training_pkl_path):
            print(f"{pkl_file} file does not exist. Reading netcdfs...")
            # Reads all the netcdfs in the data_dir
            all_data = xr.open_mfdataset(join(data_dir, "*.nc"), engine="netcdf4", concat_dim="ex_num", combine="nested")
            # all_data = xr.open_mfdataset(join(data_dir, "*[0-9][0-9][0].nc"), engine="netcdf4", concat_dim="ex_num", combine="nested")
            
            # Rechunk the data so that ex_num is in a single chunk
            all_data = all_data.chunk({'ex_num': 1})

            lats = all_data.latitude
            lons = all_data.longitude

            # Apply log to the chlorophyll variable
            all_data['chlora'] = np.log(all_data['chlora'])
            # Make nans all the values that are 0s in ssh_track
            all_data['ssh_track'].data = np.where(all_data['ssh_track'].data==0, np.nan, all_data['ssh_track'].data)

            # Apply the scaling function to each variable
            for var_name in all_var_names:
                print(f"Scaling {var_name}...")
                norm_data, self.scalers = scale_data_dataset(all_data[var_name], self.scalers, var_name)
                all_data[f'{var_name}_normalized'] = xr.DataArray(
                    norm_data,
                    dims=all_data[var_name].dims,
                    coords=all_data[var_name].coords
                )

            # Saving the scalers
            with open(join(data_dir, "scalers.pkl"), "wb") as f:
                print(f"Saving scalers to {join(data_dir, 'scalers.pkl')}...")
                pickle.dump(self.scalers, f)

            if self.plot_data:
                print("Plotting some data...")
                for i in range(10):
                    idx = np.random.randint(0, len(all_data.ex_num))
                    plot_dataset_data(idx, all_data, lats, lons)
# 
            self.X = np.stack([all_data[var_name].compute().data for var_name in input_normalized_vars], axis=0)
            self.Y = all_data[output_var].compute().data
            # Flip the first and second dimensions in X  
            self.X = np.transpose(self.X, (1, 0, 2, 3))

            # Saving the training data
            with open(training_pkl_path, "wb") as f:
                pickle.dump((self.X, self.Y), f)
        else:
            print(f"Reading {pkl_file} file...")
            with open(training_pkl_path, "rb") as f:
                self.X, self.Y = pickle.load(f)
        
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
        self.tot_inputs = self.X.shape[1] * self.previous_days + 1

        # Make the mask a float32 tensor
        self.gulf_mask = torch.tensor(self.gulf_mask, dtype=torch.float32)

        # Get the length of the dataset
        self.length = self.Y.shape[0]

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
            # print(f"start_index: {start_index}, end_idx: {end_idx}. Index: {index - i}")
            X_with_mask[start_index : end_idx, :, :] = self.X[index - i, :, :, :]
        X_with_mask[-1, :, :] = self.gulf_mask

        # Only for testing purposes plot the input data
        if self.plot_data:
            input_names = ["sst", "chlora", "ssh_track", "swot"]
            plot_single_batch_element(X_with_mask, input_names, self.previous_days, f"/unity/f1/ozavala/OUTPUTS/HR_SSH_from_Chlora/trainings/{index}.jpg")

        return X_with_mask, self.Y[index]

    def get_scaler(self):
        return self.scaler
    
    def normalize(self, x):
        return self.scaler.transform(x)

    def denormalize(self, x):
        return self.scaler.inverse_transform(x)


if __name__ == "__main__":
# Main function to test the dataset
    data_dir = "/unity/f1/ozavala/OUTPUTS/HR_SSH_from_Chlora/training_data/"
    batch_size = 2

    # Create an instance of the SimSatelliteDataset
    dataset = SimSatelliteDataset(data_dir, previous_days=4, transform=None, plot_data=True)

    # Create a data loader for the dataset
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
#
    # Iterate over the data loader
    for batch_idx, (x, y) in enumerate(data_loader):
        # Print the batch index and the batch size
        print(f"Batch {batch_idx}: x.shape = {x.shape}, y.shape = {y.shape}")
        exit()