import numpy as np
import torch
from torchvision.utils import make_grid
import torch.nn.functional as F
from base import BaseTrainer
from utils import inf_loop, MetricTracker


class TrainerAutoregressive(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            print("Epoch-based training")
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            print("Iteration-based training")
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        # # Just to save the graph in tensorboard
        # if self.device.type == 'cuda' and torch.cuda.current_device() == 0:
        #     print("Saving the graph in tensorboard")
        #     for batch_idx, (data, target) in enumerate(self.data_loader):
        #         dummy_input = torch.randn(1, *data.shape[1:]).to(self.device)
        #         self.writer.add_graph(self.model, dummy_input)
        #         break
        #     print("Graph saved in tensorboard")

            
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.data_loader):

            # Here we need to add the previous predictions as input and the future predictions as target and all the necessary modifications to the model
            # 1. Condition if ii = 0 we keep the data as is
            # 2. Condition if ii > 0 we need to concatenate the previous predictions to the data
            #if batch_idx == 0:
            data, target = data.to(self.device), target.to(self.device)

            ii = 0  # This is the index for the previous predictions
            sday = 4 # One day includes 4 channels (chlora, sst, altimeter, swot)
            dc = 7 * sday # For now is hardcoded to 7 previous days and 4 channels (chlora, sst, altimeter, swot)

            ss = 0

            self.optimizer.zero_grad()
            loss = 0
            final_output = target.clone()
            # I hardcoded the device type to cuda because I was getting an error when it tried to run on the CPU
            for ii in range(0, 2):
                # Insert the previous predictions into the data
                if ii > 0:
                    data[:, -3, :, :] = data[:, -2, :, :].clone()
                    data[:, -2, :, :] = output.squeeze()

                # Concatenate the batch to advance the predictions
                data_step = torch.cat((data[:, ss:(ss+dc), :, :], data[:, -3:, :, :]), dim=1)
                output = self.model(data_step)
                # final_output[:, ii] = output.clone()
                # Update the index for the previous predictions
                ss += sday
                # Define Sobel kernels for computing gradients along x and y directions
                sobel_kernel_x = torch.tensor([[[-1, 0, 1],
                                                [-2, 0, 2],
                                                [-1, 0, 1]]], dtype=output.dtype, device=output.device)

                sobel_kernel_y = torch.tensor([[[-1, -2, -1],
                                                [ 0,  0,  0],
                                                [ 1,  2,  1]]], dtype=output.dtype, device=output.device)

                # Reshape kernels to match the conv2d weight shape: [out_channels, in_channels, kH, kW]
                sobel_kernel_x = sobel_kernel_x.unsqueeze(1)  # Shape: [1, 1, 3, 3]
                sobel_kernel_y = sobel_kernel_y.unsqueeze(1)  # Shape: [1, 1, 3, 3]
                #print(f'sobel_kernel_x.shape: {sobel_kernel_x.shape}')
                #print(f'sobel_kernel_y.shape: {sobel_kernel_y.shape}')
                #print(final_output.shape)
                

                # Reshape kernels to match the conv2d weight shape: [1, in_channels, kH, kW]
                #sobel_kernel_x = torch.cat((sobel_kernel_x, sobel_kernel_x), dim=1)  # Shape: [1, 2, 3, 3]
                #sobel_kernel_y = torch.cat((sobel_kernel_y, sobel_kernel_y), dim=1)  # Shape: [1, 2, 3, 3]
                #print(f'sobel_kernel_x.shape: {sobel_kernel_x.shape}')
                #print(f'sobel_kernel_y.shape: {sobel_kernel_y.shape}')

                #final_output = final_output.unsqueeze(1)  # Shape: [batch_size, num_days (2), height, width]
                output = output.unsqueeze(1)  # Shape: [batch_size, 1, height, width]
                #print(f'output.shape: {output.shape}')
                #print(f'target.shape: {target.shape}')
                #print(f'target[:,ii,:,:].shape: {target[:,ii,:,:].shape}')
                # Compute gradients along x and y directions using conv2d
                grad_output_x = F.conv2d(output, sobel_kernel_x, padding=1)  # Shape: [batch_size, 1, height, width]
                grad_output_y = F.conv2d(output, sobel_kernel_y, padding=1)  # Shape: [batch_size, 1, height, width]

                # Compute gradients of the target
                grad_target_x = F.conv2d(target[:,ii,:,:].unsqueeze(1), sobel_kernel_x, padding=1)  # Shape: [batch_size, 1, height, width]
                grad_target_y = F.conv2d(target[:,ii,:,:].unsqueeze(1), sobel_kernel_y, padding=1)  # Shape: [batch_size, 1, height, width]

                # Compute the gradient magnitude (2D norm) for each element in the batch
                grad_magnitude_output = torch.sqrt(grad_output_x ** 2 + grad_output_y ** 2)  # Shape: [batch_size, 1, height, width]
                grad_magnitude_target = torch.sqrt(grad_target_x ** 2 + grad_target_y ** 2)  # Shape: [batch_size, 1, height, width]

                # Normalize the gradients with mean 0 and std 1
                output_gradient = (grad_magnitude_output - grad_magnitude_output.mean()) / grad_magnitude_output.std()
                target_gradient = (grad_magnitude_target - grad_magnitude_target.mean()) / grad_magnitude_target.std()
                    

                # For debugging purposes plot one of the gradients
                # import matplotlib.pyplot as plt
                # plt.imshow(target_gradient[0, 0, :, :].cpu().numpy(), vmin=0, vmax=2)
                # plt.colorbar()
                # plt.savefig("target_gradient.png")
                # plt.close()

                # End of the loop for the previous predictions and computing the loss
                output_loss = self.criterion(output, target[:,ii,:,:])
                gradient_loss = self.criterion(output_gradient, target_gradient)
                step_loss = (output_loss + gradient_loss)/ 2 # because accumulation of the loss is 2 steps in this case
                step_loss.backward(retain_graph=True)
                loss += step_loss

            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(final_output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                loss = 0
                ii = 1  # This is the index for the previous predictions
                sday = 4 # One day includes 4 channels (chlora, sst, altimeter, swot)
                dc = 7 * sday # For now is hardcoded to 7 previous days and 4 channels (chlora, sst, altimeter, swot)

                ss = 4
                #for ii in range(0, 2):
                #    # Insert the previous predictions into the data
                #    if ii > 0:
                #        data[:, -3, :, :] = data[:, -2, :, :].clone()
                #        data[:, -2, :, :] = output
                #    # Concatenate the batch to advance the predictions
                #    data_step = torch.cat((data[:, ss:(ss+dc), :, :], data[:, -3:, :, :]), dim=1)
                #    output = self.model(data_step)
                #    loss += self.criterion(output, target)/2

                # We are validating only the last prediction
                data_step = torch.cat((data[:, ss:(ss+dc), :, :], data[:, -3:, :, :]), dim=1)
                output = self.model(data_step)

                sobel_kernel_x = torch.tensor([[[-1, 0, 1],
                                                [-2, 0, 2],
                                                [-1, 0, 1]]], dtype=output.dtype, device=output.device)

                sobel_kernel_y = torch.tensor([[[-1, -2, -1],
                                                [ 0,  0,  0],
                                                [ 1,  2,  1]]], dtype=output.dtype, device=output.device)

                # Reshape kernels to match the conv2d weight shape: [out_channels, in_channels, kH, kW]
                sobel_kernel_x = sobel_kernel_x.unsqueeze(1)  # Shape: [1, 1, 3, 3]
                sobel_kernel_y = sobel_kernel_y.unsqueeze(1)  # Shape: [1, 1, 3, 3]
                output = output.unsqueeze(1)  # Shape: [batch_size, 1, height, width]

                grad_output_x = F.conv2d(output, sobel_kernel_x, padding=1)  # Shape: [batch_size, 1, height, width]
                grad_output_y = F.conv2d(output, sobel_kernel_y, padding=1)  # Shape: [batch_size, 1, height, width]

                # Compute gradients of the target
                grad_target_x = F.conv2d(target[:,ii,:,:].unsqueeze(1), sobel_kernel_x, padding=1)  # Shape: [batch_size, 1, height, width]
                grad_target_y = F.conv2d(target[:,ii,:,:].unsqueeze(1), sobel_kernel_y, padding=1)  # Shape: [batch_size, 1, height, width]

                # Compute the gradient magnitude (2D norm) for each element in the batch
                grad_magnitude_output = torch.sqrt(grad_output_x ** 2 + grad_output_y ** 2)  # Shape: [batch_size, 1, height, width]
                grad_magnitude_target = torch.sqrt(grad_target_x ** 2 + grad_target_y ** 2)  # Shape: [batch_size, 1, height, width]

                # Normalize the gradients with mean 0 and std 1
                output_gradient = (grad_magnitude_output - grad_magnitude_output.mean()) / grad_magnitude_output.std()
                target_gradient = (grad_magnitude_target - grad_magnitude_target.mean()) / grad_magnitude_target.std()

                # End of the loop for the previous predictions and computing the loss
                output_loss = self.criterion(output, target[:,ii,:,:])
                gradient_loss = self.criterion(output_gradient, target_gradient)
                loss = (output_loss + gradient_loss)
                # We are validating only the final prediction   
                #loss = self.criterion(output, target[:,1,:,:])

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
            # self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
