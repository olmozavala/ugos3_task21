import sys
# sys.path.append("/unity/f1/ozavala/CODE/ugos3_task21/DA_Chlora") # Only for testing purposes
from base import BaseDataLoader
import torch
from os.path import join
import os
import xarray as xr
import pickle
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from data_loader.data_sets import SimSatelliteDataset

class DefaultDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, 
                 num_workers=1, training=True, previous_days=1, dataset_type="regular"):
        self.data_dir = data_dir
        self.dataset = SimSatelliteDataset(self.data_dir, transform=None, previous_days=previous_days, 
                                           training=training, dataset_type=dataset_type)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
