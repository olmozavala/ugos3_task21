import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from os.path import join
import os
import json
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
        # training=False,
        training=True,
        num_workers=config['data_loader']['args']['num_workers'],
        previous_days=config['data_loader']['args']['previous_days']
    )

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
    # if config['n_gpu'] > 1:
        # model = torch.nn.DataParallel(model)
    model = torch.compile(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    # Read the lats and lons from 
    lats = data_loader.dataset.lats
    lons = data_loader.dataset.lons

    #  Set the precision of the model
    torch.set_float32_matmul_precision('medium')

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            print(f"Batch {i} of {len(data_loader)}")
            data, target = data.to(device), target.to(device)
            output = model(data)

            # Plotting the output
            print(f"Shape of output: {output.shape}")
            # Plotting the output
            # Bring the data to numpy
            data_cpu = data.cpu().numpy()
            target_cpu = target.cpu().numpy()
            output_cpu = output.cpu().numpy()
            for j in range(output.shape[0]):
                file_name = join(output_dir, f"pred_batch_{i}_sample_{j}.jpg")
                plot_predictions(data_cpu[j, :, :, :], target_cpu[j, :, :], output_cpu[j, :, :], file_name)

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

            # Save the output to a netcdf file
            output_file = join(output_dir, f"pred_batch_{i}.nc")
            xr.Dataset({
                'output': (['latitude', 'longitude'], output_cpu),
                'target': (['latitude', 'longitude'], target_cpu)
            }, coords={
                'latitude': lats,
                'longitude': lons
            }).to_netcdf(output_file)

            if i > 3:
                break

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
