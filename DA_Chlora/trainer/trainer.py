import numpy as np
import torch
from torchvision.utils import make_grid
import torch.nn.functional as F
from base import BaseTrainer
from utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        self.register_sobel_kernels()
        self.register_curvature_kernels()
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
    
    def register_sobel_kernels(self):
        sobel_x = torch.tensor([
            [[-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]]], dtype=torch.float32)

        sobel_y = torch.tensor([
            [[-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]]], dtype=torch.float32)

        # Reshape to [out_channels, in_channels, kH, kW]
        self.sobel_kernel_x = sobel_x.unsqueeze(1).to(self.device)
        self.sobel_kernel_y = sobel_y.unsqueeze(1).to(self.device)

        # Make sure autograd does not track these
        self.sobel_kernel_x.requires_grad_(False)
        self.sobel_kernel_y.requires_grad_(False)

    def register_curvature_kernels(self):
        laplacian = torch.tensor([
            [[ 0,  1,  0],
            [ 1, -4,  1],
            [ 0,  1,  0]]
        ], dtype=torch.float32)

        self.laplace_kernel = laplacian.unsqueeze(1).to(self.device)
        self.laplace_kernel.requires_grad_(False)

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
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            # I hardcoded the device type to cuda because I was getting an error when it tried to run on the CPU

            output = self.model(data)
            # Create a mask for the valid points
            mask = data[:, -1, :, :].unsqueeze(1)  # [B,1,H,W]
            valid_points = torch.sum(mask) 
            #print(f"Output shape: {output.shape}")
            #print(f"Target shape: {target.shape}")

            eps = 1e-10
            output = output.unsqueeze(1)  # shape [B,1,H,W]
            # print(f"Output shape: {output.shape}")
            # compute sobel gradients
            grad_output_x = F.conv2d(output, self.sobel_kernel_x, padding=1)
            grad_output_y = F.conv2d(output, self.sobel_kernel_y, padding=1)
            grad_target_x = F.conv2d(target, self.sobel_kernel_x, padding=1)
            grad_target_y = F.conv2d(target, self.sobel_kernel_y, padding=1)

            # magnitude
            grad_magnitude_output = torch.sqrt(grad_output_x**2 + grad_output_y**2 + eps)
            grad_magnitude_target = torch.sqrt(grad_target_x**2 + grad_target_y**2 + eps)

            # normalization (fixed)
            output_gradient = (grad_magnitude_output - grad_magnitude_output.mean()) / (grad_magnitude_output.std() + eps)
            target_gradient = (grad_magnitude_target - grad_magnitude_target.mean()) / (grad_magnitude_target.std() + eps)

            # analyze curvature
            curv_output = F.conv2d(output, self.laplace_kernel, padding=1)
            curv_target = F.conv2d(target, self.laplace_kernel, padding=1)

            curv_output_norm = (curv_output - curv_output.mean()) / (curv_output.std() + eps)
            curv_target_norm = (curv_target - curv_target.mean()) / (curv_target.std() + eps)

            curvature_loss = ((curv_output_norm - curv_target_norm)**2 * mask).sum() / valid_points

            # pixelwise loss reduced by mask
            output_loss = ((output - target)**2 * mask).sum() / valid_points
            gradient_loss = ((output_gradient - target_gradient)**2 * mask).sum() / valid_points

            w_curvature = 0.5
            w_gradient = 0.9
            w_output = 1.0
            # print(f"Output loss: {output_loss}, Gradient loss: {gradient_loss}")
            loss = (output_loss * w_output + gradient_loss * w_gradient + curvature_loss * w_curvature) / (w_output + w_gradient + w_curvature + eps)
            #loss = output_loss / valid_points
            # If loss is nan, kill the process
            if torch.isnan(loss):
                print(f"Loss is nan at epoch {epoch}, batch {batch_idx}")
                print(f"Output loss: {output_loss}, Gradient loss: {gradient_loss}")
                #print(f"Output: {output}, Target: {target}")
                #print(f"Output gradient: {output_gradient}, Target gradient: {target_gradient}")
                raise ValueError("Loss is nan")

            loss.backward()
            
            # Clip gradients to prevent exploding gradients
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.01, norm_type=1)

            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

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

                output = self.model(data)
                mask = data[0, -1, :, :]
                valid_points = torch.sum(mask) + 1e-8

                eps = 1e-8
                output = output.unsqueeze(1)  # shape [B,1,H,W]

                # compute sobel gradients
                grad_output_x = F.conv2d(output, self.sobel_kernel_x, padding=1)
                grad_output_y = F.conv2d(output, self.sobel_kernel_y, padding=1)
                grad_target_x = F.conv2d(target, self.sobel_kernel_x, padding=1)
                grad_target_y = F.conv2d(target, self.sobel_kernel_y, padding=1)

                # magnitude
                grad_magnitude_output = torch.sqrt(grad_output_x**2 + grad_output_y**2 + eps)
                grad_magnitude_target = torch.sqrt(grad_target_x**2 + grad_target_y**2 + eps)

                # normalization (fixed)
                output_gradient = (grad_magnitude_output - grad_magnitude_output.mean()) / (grad_magnitude_output.std() + eps)
                target_gradient = (grad_magnitude_target - grad_magnitude_target.mean()) / (grad_magnitude_target.std() + eps)

                # analyze curvature
                curv_output = F.conv2d(output, self.laplace_kernel, padding=1)
                curv_target = F.conv2d(target, self.laplace_kernel, padding=1)

                curv_output_norm = (curv_output - curv_output.mean()) / (curv_output.std() + eps)
                curv_target_norm = (curv_target - curv_target.mean()) / (curv_target.std() + eps)

                curvature_loss = ((curv_output_norm - curv_target_norm)**2 * mask).sum() / valid_points

                # pixelwise loss reduced by mask
                output_loss = ((output - target)**2 * mask).sum() / valid_points
                gradient_loss = ((output_gradient - target_gradient)**2 * mask).sum() / valid_points

                w_curvature = 0.5
                w_gradient = 0.9
                w_output = 1.0
                # print(f"Output loss: {output_loss}, Gradient loss: {gradient_loss}")
                loss = (output_loss * w_output + gradient_loss * w_gradient + curvature_loss * w_curvature) / (w_output + w_gradient + w_curvature + eps)

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
