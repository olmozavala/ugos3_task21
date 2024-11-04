# %%
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
import matplotlib.pyplot as plt
import numpy as np
from data_loader.loader_utils import plot_predictions
import xarray as xr

import torch._dynamo
torch._dynamo.config.suppress_errors = True
# %%

def main(config):
    logger = config.get_logger('test')

    batch_size = config['data_loader']['args']['batch_size']
    data_dir = config['data_loader']['args']['data_dir']
    dataset_type = config['data_loader']['args']['dataset_type']
    previous_days = config['data_loader']['args']['previous_days']

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=config['data_loader']['args']['batch_size'],
        shuffle=False,
        validation_split=0.0,
        training=False,
        # training=True,
        num_workers=config['data_loader']['args']['num_workers'],
        previous_days=config['data_loader']['args']['previous_days'],
        dataset_type=dataset_type
    )

    # Read the scalers from the data_dir
    with open(join(data_dir, 'scalers.pkl'), 'rb') as f:
        scalers = pickle.load(f)

    mean_ssh = scalers["ssh"]["mean"]
    std_ssh = scalers["ssh"]["std"]

    # Read config.json from the weights_file directory
    weights_dir = config['tester']['weights_dir']
    with open(join(weights_dir, 'config.json'), 'r') as f:
        training_config = json.load(f)

    # Model name
    model_name = training_config['name']

    # Setup output directory
    output_dir = join(config['tester']['output_dir'], model_name)
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

    validation_loss = []
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            if i == 0:
                # Crop the lats and lons to the shape of the data
                lats = lats[:data.shape[2]]
                lons = lons[:data.shape[3]]

            print(f"Batch {i} of {len(data_loader)}")
            data, target = data.to(device), target.to(device)
            output = model(data)

            # computing loss, metrics on test set
            # Scale the output and target
            output = output * std_ssh + mean_ssh
            target = target * std_ssh + mean_ssh

            # Plotting the output
            print(f"Shape of output: {output.shape}")
            # Plotting the output
            # Bring the data to numpy
            data_cpu = data.cpu().numpy()
            target_cpu = target.cpu().numpy()
            output_cpu = output.cpu().numpy()
            # For each batch plot the first 20 samples
            # for j in range(min(output.shape[0], 20)):
            for j in range(previous_days, min(output.shape[0], previous_days + 10)):
                ex_num = i*batch_size + j + 1
                file_name = join(output_dir, f"{model_name}_ex_{ex_num:03d}.png")
                plot_predictions(data_cpu[j, :, :, :], target_cpu[j, :, :], 
                                 output_cpu[j, :, :], file_name, lats, lons, dataset_type)

            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for j, metric in enumerate(metric_fns):
                total_metrics[j] += metric(output, target) * batch_size

            # Compute the RMSE for each sample in the batch
            rmse = torch.sqrt(torch.mean((output - target)**2, dim=(1, 2)))
            for j in range(rmse.shape[0]):
                validation_loss.append(rmse[j].item())

            if save_predictions:
                # Save the output to a netcdf file
                for j in range(output.shape[0]):
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

    # Make a scatter plot of the validation loss
    mean_rmse = np.mean(validation_loss)
    plt.figure()
    title = f"Mean RMSE: {mean_rmse:.4f} m"
    plt.scatter(range(len(validation_loss)), validation_loss)
    plt.xlabel("Examples from validation set")
    plt.ylabel("RMSE (m)")
    plt.title(title)
    plt.savefig(join(output_dir, "validation_loss.png"), dpi=300, bbox_inches='tight')
    plt.close()
    # Save the RMSE as a csv file
    np.savetxt(join(output_dir, "validation_loss.csv"), validation_loss, delimiter=",")

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

    save_predictions = False
    args.add_argument('-s', '--save_predictions', default=False, type=bool,
                      help='Save the predictions to a netcdf file (default: False)')

    config = ConfigParser.from_args(args)
    main(config)

# %% Redoo RMSE plot
# Read the RMSE from the csv file
folder = "/unity/g2/jvelasco/ai_outs/task21_set1/testing/Debug_model_gradient_mode_full_dataset"
# folder = "/unity/f1/ozavala/OUTPUTS/HR_SSH_from_Chlora/testing/UNet_with_upsample_AdamW_Wdecay_1e-4_opt_on_regular_sep_validation"
file_name = join(folder, "loss.csv")
rmse_data = np.loadtxt(file_name, delimiter=",")
mean_rmse = np.mean(rmse_data)
vmin = 0.005
vmax = 0.030
# Make a scatter plot of the RMSE
plt.figure()
plt.scatter(range(len(rmse_data)), rmse_data)
plt.xlabel("Examples from validation set")
plt.ylabel("RMSE (m)")
plt.title(f"Mean RMSE: {mean_rmse:.4f} m")
# Set the x and y limits
plt.ylim(vmin, vmax)
plt.savefig(join(folder, "validation_loss.png"), dpi=300, bbox_inches='tight')
plt.close()
# %%
