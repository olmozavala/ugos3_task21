import numpy as np
import torch
from torchvision.utils import make_grid
import torch.nn.functional as F
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from model.metric import gaussian_blur, masked_rmse, gradient_rmse


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
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            # I hardcoded the device type to cuda because I was getting an error when it tried to run on the CPU

            output = self.model(data)
            loss = self.criterion(output, target, data=data)
            # If loss is nan, kill the process
            if torch.isnan(loss):
                print(f"Loss is nan at epoch {epoch}, batch {batch_idx}")
                raise ValueError("Loss is nan")

            loss.backward()

            # Clip the gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, norm_type=2)

            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                output_metric = output.unsqueeze(1) if output.dim() == 3 else output
                self.train_metrics.update(met.__name__, met(output_metric, target))

            with torch.no_grad():
                mask = data[:, -1:, :, :]
                pred_lf = gaussian_blur(output.detach(), sigma=5)
                pred_hf = output.detach() - pred_lf
                tgt_lf = gaussian_blur(target, sigma=5)
                tgt_hf = target - tgt_lf
                self.writer.add_scalar('rmse/large_scale', masked_rmse(pred_lf, tgt_lf, mask).item())
                self.writer.add_scalar('rmse/small_scale', masked_rmse(pred_hf, tgt_hf, mask).item())
                self.writer.add_scalar('rmse/gradient', gradient_rmse(output.detach(), target, mask).item())

            if batch_idx % self.log_step == 0:
                loss_breakdown = ""
                if hasattr(self.criterion, "last_components") and getattr(self.criterion, "last_components"):
                    comps = getattr(self.criterion, "last_components")
                    # Print weight*loss_i (numerator terms)
                    parts = []
                    for name, v in comps.items():
                        wloss = v.get("weighted", None)
                        if wloss is None:
                            continue
                        try:
                            parts.append(f"{name}={wloss.item():.6f}")
                        except Exception:
                            pass
                    if parts:
                        loss_breakdown = " | " + ", ".join(parts)
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()) + loss_breakdown)
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
                loss = self.criterion(output, target, data=data)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    output_metric = output.unsqueeze(1) if output.dim() == 3 else output
                    self.valid_metrics.update(met.__name__, met(output_metric, target))

                mask = data[:, -1:, :, :]
                pred_lf = gaussian_blur(output, sigma=5)
                pred_hf = output - pred_lf
                tgt_lf = gaussian_blur(target, sigma=5)
                tgt_hf = target - tgt_lf
                self.writer.add_scalar('rmse/large_scale', masked_rmse(pred_lf, tgt_lf, mask).item())
                self.writer.add_scalar('rmse/small_scale', masked_rmse(pred_hf, tgt_hf, mask).item())
                self.writer.add_scalar('rmse/gradient', gradient_rmse(output, target, mask).item())
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
