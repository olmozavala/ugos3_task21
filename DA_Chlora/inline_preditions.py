import argparse
import torch
import pickle
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from utils import prepare_device
from os.path import join
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from data_loader.loader_utils import plot_predictions
import xarray as xr

def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=config['data_loader']['args']['batch_size'],
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=config['data_loader']['args']['num_workers'],
        previous_days=config['data_loader']['args']['previous_days'],
        dataset_type=config['data_loader']['args']['dataset_type']
    )

    batch_size = config['data_loader']['args']['batch_size']
    data_dir = config['data_loader']['args']['data_dir']
    dataset_type = config['data_loader']['args']['dataset_type']

    # Read the scalers from the data_dir
    with open(join(data_dir, 'scalers.pkl'), 'rb') as f:
        scalers = pickle.load(f)

    mean_ssh = scalers["ssh"]["mean"]
    std_ssh = scalers["ssh"]["std"]

    weights_dir = config['tester']['weights_dir']
    with open(join(weights_dir, 'config.json'), 'r') as f:
        training_config = json.load(f)

    # Model name
    model_name = training_config['name']

    # Setup output directory
    output_dir = join(config['tester']['output_dir'], f"{model_name}_inline")
    os.makedirs(output_dir, exist_ok=True)

    weights_file = join(weights_dir, 'model_best.pth')

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    # Load weights
    logger.info('Loading checkpoint: {} ...'.format(weights_file))
    checkpoint = torch.load(weights_file)
    state_dict = checkpoint['state_dict']

    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)

    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model)

    model = torch.compile(model)
    model.load_state_dict(state_dict)
    model.eval()

    # OPTIMIZATION
    torch.set_float32_matmul_precision('medium')

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    # Read the lats and lons from 
    lats = data_loader.dataset.lats
    lons = data_loader.dataset.lons

    prev_prediction_file_1 = "/unity/f1/ozavala/OUTPUTS/HR_SSH_from_Chlora/testing/UNet_with_upsample_AdamW_Wdecay_1e-4_opt_on_regular_sep_validation/pred_batch_0_sample_1.nc"
    prev_prediction_file_2 = "/unity/f1/ozavala/OUTPUTS/HR_SSH_from_Chlora/testing/UNet_with_upsample_AdamW_Wdecay_1e-4_opt_on_regular_sep_validation/pred_batch_0_sample_0.nc"
    files = [prev_prediction_file_1, prev_prediction_file_2]
    prev_pred = []
    for c_file in files:
        ds = xr.open_dataset(c_file)
        cur_pred = ds['output'].data
        # Normalize the data
        cur_pred = (cur_pred - mean_ssh) / std_ssh
        prev_pred.append(cur_pred)

    prev_pred = np.stack(prev_pred, axis=0)
    print("Prev pred shape: ", prev_pred.shape)
    # Get torch tensor
    prev_prediction = torch.tensor(prev_pred)
    # Send to device
    prev_prediction = prev_prediction.to(device)

    validation_loss = []
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            if i == 0:
                # Crop the lats and lons to the shape of the data
                lats = lats[:data.shape[2]]
                lons = lons[:data.shape[3]]

            print(f"Batch {i} of {len(data_loader)}")
            data, target = data.to(device), target.to(device)

            # We need to replace the elements corresponding to the '
            # last two predictions with the prev_predictions
            outputs = []
            if i == 0:
                start_index = 2
                prev_prediction[-2, :, :] = target[0, :, :]
                prev_prediction[-1, :, :] = target[1, :, :]
            else:
                start_index = 0

            for j in range(start_index, data.shape[0]): # Iterate from 2 to the end in current batch
                print(f"Sample {j} of {data.shape[0]}")
                data_input = data[j, :, :, :]
                # Extend the input one dimension to simulate batch of 1
                data_input[-2, :, :] = prev_prediction[-1, :, :]
                data_input[-3, :, :] = prev_prediction[-2, :, :]
                data_input = data_input.unsqueeze(0)
                output = model(data_input) 
                # Move the predictions and append the new predictions to the prev_predictions
                prev_prediction[-2, :, :] = prev_prediction[-1, :, :]
                prev_prediction[-1, :, :] = output[0, :, :]
                outputs.append(output)

            # Stack the outputs
            outputs = torch.stack(outputs, dim=0)
            # computing loss, metrics on test set only for the first index of the output and target (the SSH)
            # Scale the output and target
            outputs = outputs * std_ssh + mean_ssh
            outputs = outputs.squeeze()
            target = target[start_index:,:,:] * std_ssh + mean_ssh

            # Plotting the output
            print(f"Shape of output: {outputs.shape} and target: {target.shape}")
            # Bring the data to numpy
            data_cpu = data.cpu().numpy()
            target_cpu = target.cpu().numpy()
            output_cpu = outputs.cpu().numpy()
            # For each batch plot the first 20 samples
            for j in range(min(outputs.shape[0], 10)):
                ex_num = i*batch_size + j + 1
                file_name = join(output_dir, f"ex_{ex_num:03d}.png")
                plot_predictions(data_cpu[j, :, :, :], target_cpu[j, :, :], output_cpu[j, :, :], 
                                 file_name, lats, lons, dataset_type=dataset_type)

            loss = loss_fn(outputs, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for j, metric in enumerate(metric_fns):
                total_metrics[j] += metric(outputs, target) * batch_size

            # Compute the RMSE for each sample in the batch
            rmse = torch.sqrt(loss)
            validation_loss.append(rmse.item())

            # Save the output to a netcdf file
            for j in range(outputs.shape[0]):
                output_file = join(output_dir, f"pred_batch_{i}_sample_{j}.nc")
                xr.Dataset({
                    'output': (['latitude', 'longitude'], output_cpu[j, :, :]),
                    'target': (['latitude', 'longitude'], target_cpu[j, :, :])
                }, coords={
                    'latitude': lats,
                    'longitude': lons
                }).to_netcdf(output_file)

    # Save the loss
    loss_file = join(output_dir, "loss.csv")
    with open(loss_file, "w") as f:
        for loss in validation_loss:
            f.write(f"{loss}\n")

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='config.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
