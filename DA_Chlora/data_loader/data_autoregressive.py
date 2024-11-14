import sys
sys.path.append("/unity/g2/jvelasco/github/ugos3_task21/DA_Chlora") # Only for testing purposes
import os
import pickle
import numpy as np
import xarray as xr
import torch
from os.path import join
from data_loader.loader_utils import *
import matplotlib.pyplot as plt


# Function to apply StandardScaler to an array and persist the scaler
def scale_data_dataset(data, scalers, name, training=True)->tuple[xr.DataArray, dict]:
    """
    Scales the data using the scalers and returns the scaled data and the scalers
    Args:
        data (xr.DataArray): The data to scale
        scalers (dict): The scalers to use
        name (str): The name of the variable
        training (bool): Whether the data is for training or not
    Returns:
        tuple (xr.DataArray, dict): The scaled data and the scalers dictionary
    """

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

# Class to load the autoregressive dataset
class AutoregressiveDataset:
    def __init__(self, data_dir, transform=None, 
                 previous_days=1, horizon_days=1,
                 plot_data=False, training=True, 
                 input_vars=["sst", "chlora", "ssh_track", "swot"], 
                 output_vars=["ssh"], 
                 demo=False,
                 dataset_type="regular"):
        
        if horizon_days < 1:
            raise ValueError("Horizon days must be greater than 0")
        self.data_dir = data_dir
        self.previous_days = previous_days
        self.horizon_days = horizon_days
        self.transform = transform
        self.scalers = {}  # To store scalers for each variable
        self.plot_data = plot_data

        # Input variables
        self.input_vars = input_vars
        self.output_vars = output_vars
        all_var_names = input_vars + output_vars
        self.input_normalized_vars = [f"{var}_normalized" for var in input_vars]
        output_var = [f"{var}_normalized" for var in output_vars][0]

        # Scaling the data
        scalers_file = "scalers.pkl"
        if training and not demo:
            pkl_file = "training.pkl"
        elif training and demo:
            pkl_file = "training_small.pkl"
        else:
            pkl_file = "validation.pkl"

        # Verify if 'training.pkl' file exists (Not needed at the moment)
        training_pkl_path = join(data_dir, pkl_file)
        if not os.path.exists(training_pkl_path):
            print(f"File {training_pkl_path} does not exist")
            raise FileNotFoundError(f"File {training_pkl_path} does not exist")
        else: # Read the file
            print(f"Reading {pkl_file} file...")
            with open(training_pkl_path, "rb") as f:
                self.X, self.Y, self.lats, self.lons = pickle.load(f)

        # Convert the arrays to PyTorch tensors
        self.xyarrays2Tensors()

        # Allocate total inputs and outputs nvars * (previous_days + horizon_days) + 3 (gulf mask + SSH-1 + SSH-2)
        self.tot_inputs = self.X.shape[1] * (self.previous_days + self.horizon_days) + 3

        # Get the length of the datas
        # Not sure why but we need to skip the first two days because of (SSH-1 and SSH-2 + `horizon_days`, else there is not enough data to predict the autoregressive part)
        self.length = self.Y.shape[0] - (2 + self.horizon_days)

        # Verify the dimensions
        print(f"X shape: {self.X.shape}")
        print(f"Y shape: {self.Y.shape}")
        print("Preloading by the data loader is done!")

    def make_gulf_mask(self):
        """
        Generate a mask for the Gulf of Mexico in torch tensor format
        """
        gulf = np.zeros_like(self.Y[0,:,:])
        # Create a mask for the Gulf of Mexico
        gulf = np.where(~np.isnan(gulf), 1, 0)
        self.gulf_mask = torch.tensor(gulf, dtype=torch.float32)

    def xyarrays2Tensors(self, crop_factor=8):
        """
        Remove the nan values from the training and target data and convert to PyTorch tensors and make the Gulf Mask
        """
        self.X = np.where(np.isnan(self.X), 0, self.X)
        self.Y = np.where(np.isnan(self.Y), 0, self.Y)
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.Y = torch.tensor(self.Y, dtype=torch.float32)

        if crop_factor is not None:
            # Crop the last two dimensions to the largest dimension divisible by n factor
            new_height = (self.X.shape[2] // crop_factor) * crop_factor
            new_width = (self.X.shape[3] // crop_factor) * crop_factor
            self.X = self.X[:, :, :new_height, :new_width]
            self.Y = self.Y[:, :new_height, :new_width]
            self.make_gulf_mask()

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        # We can start at day 0 since we need `previous_days` days as predictors
        if index <= self.previous_days:
            index = self.previous_days + 1
        
        # Allocate the input tensor
        X_with_mask = np.zeros((self.tot_inputs, self.X.shape[2], self.X.shape[3]), dtype=np.float32)
        size_per_day = self.X.shape[1]

        # Append previous and future days to the input tensor
        for i in range(self.previous_days + self.horizon_days):
            # Append the previous days to the X_with_mask
            start_index = i * size_per_day
            end_idx = (i * size_per_day) + size_per_day
            X_with_mask[start_index : end_idx, :, :] = self.X[index - self.previous_days + i + 1, :, :, :]
        
        # Add the Gulf Mask as the last channel
        X_with_mask[-1, :, :] = self.gulf_mask

        # Append the SSH-1 and SSH-2 with noise
        noise_level = 0.2
        noise_ssh = np.random.randn(self.Y.shape[1],self.Y.shape[2]) * noise_level
        X_with_mask[-2, :, :] = self.Y[index-1, :, :] + noise_ssh
        X_with_mask[-3, :, :] = self.Y[index-2, :, :] + noise_ssh

        # LAST INDEX CONFIGURATION Important! To reutilize the pre-trained weights
        # -3: SSH-2
        # -2: SSH-1
        # -1: Gulf Mask

        # Allocate the output tensor
        Y_autoregressive = self.Y[index : index + self.horizon_days, :, :]
        
        if self.plot_data:
            #self.plot_batch(X_with_mask, Y_autoregressive, index)
            input_names = ["sst", "chlora", "ssh_track", "swot"]
            plot_single_batch_element(X_with_mask, self.Y[index], input_names, self.previous_days+self.horizon_days, 
                                      f"/unity/g2/jvelasco/ai_outs/task21_set1/higos/batch_example_{index}.jpg",
                                      self.lats, self.lons, dataset_type="gradient")

        return X_with_mask, Y_autoregressive
    
    def get_scaler(self):
        return self.scalers

    def normalize(self, x):
        return self.scaler.transform(x)

    def denormalize(self, x):
        return self.scaler.inverse_transform(x)
    
    def plot_batch(self, X, Y, index):
        input_names = self.input_vars #+ ["gulf_mask", "SSH-1", "SSH-2"]
        fig, ax = plt.subplots((self.horizon_days + self.previous_days + 1), len(input_names), figsize=(15, 5))
        for i in range(self.horizon_days + self.previous_days):
            for j in range(len(input_names)):
                ax[i, j].imshow(X[i+j, :, :], cmap="viridis")
                ax[i, j].set_title(input_names[j])
        ax[-1, 0].imshow(X[-3, :, :], cmap="viridis")
        ax[-1, 1].imshow(X[-2, :, :], cmap="viridis")
        ax[-1, 2].imshow(X[-1, :, :], cmap="viridis")
        ax[-1, 0].set_title("SSH-2")
        ax[-1, 1].set_title("SSH-1")
        ax[-1, 2].set_title("Gulf Mask")
        plt.tight_layout()
        fig.savefig(f"/unity/g2/jvelasco/batch_example_{index}.jpg")
        fig.clf()

        fig, ax = plt.subplots(len(Y), 1, figsize=(15, 5))
        for i in range(len(Y)):
            ax[i].imshow(Y[i, :, :], cmap="viridis")

        plt.tight_layout()
        fig.savefig(f"/unity/g2/jvelasco/target_example_{index}.jpg")
        fig.clf()

if __name__ == "__main__":
    # Main function to test the dataset
    data_dir = "/unity/f1/ozavala/OUTPUTS/HR_SSH_from_Chlora/training_data"
    batch_size = 1
    training = False
    plot_data = False
    previous_days = 7
    horizon_days = 2
    shuffle = False
    demo = True

    # Create an instance of the AutoregressiveDataset
    dataset = AutoregressiveDataset(data_dir, previous_days=previous_days, 
                                    horizon_days=horizon_days, plot_data=plot_data, training=training, demo=demo)
    
    # Create a data loader for the dataset
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    # Iterate over the data loader
    for batch_idx, (x, y) in enumerate(data_loader):
        print(f"Batch {batch_idx}: x.shape = {x.shape}, y.shape = {y.shape}")
        if batch_idx > 0:
            break

    print("Done!")