import numpy as np
import pandas as pd
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

    def _log_channel_attribution(self, epoch):
        """Compute gradient of loss w.r.t. each input channel on one validation batch (P5)."""
        self.model.eval()
        data, target = next(iter(self.valid_data_loader))
        data, target = data.to(self.device), target.to(self.device)
        data = data.detach().requires_grad_(True)
        with torch.enable_grad():
            output = self.model(data)
            loss = self.criterion(output, target, data=data)
            loss.backward()
        if data.grad is not None:
            importance = data.grad.abs().mean(dim=(0, 2, 3))
            self.writer.set_step((epoch - 1) * self.len_epoch)
            for i, imp in enumerate(importance):
                self.writer.add_scalar(f'attribution/channel_{i}', imp.item())
        self.model.train()

    def _log_spectral_diagnostics(self, pred, target, mask):
        """Log effective resolution and PSD ratio via FFT (P1)."""
        pred_b1hw = pred.unsqueeze(1) if pred.dim() == 3 else pred
        tgt_b1hw  = target.unsqueeze(1) if target.dim() == 3 else target
        mask_b1hw = mask[:, :1, :, :]

        H, W = pred_b1hw.shape[-2], pred_b1hw.shape[-1]
        window_h = torch.hann_window(H, device=pred.device).view(-1, 1)
        window_w = torch.hann_window(W, device=pred.device).view(1, -1)
        window = window_h * window_w

        psd_pred   = torch.abs(torch.fft.rfft2(pred_b1hw  * mask_b1hw * window)) ** 2
        psd_target = torch.abs(torch.fft.rfft2(tgt_b1hw   * mask_b1hw * window)) ** 2

        ratio = psd_pred.mean(0) / (psd_target.mean(0) + 1e-10)  # [1, H, W//2+1]
        kx = torch.fft.rfftfreq(W, device=pred.device)
        ratio_1d = ratio.squeeze(0).mean(0)  # average over H -> [W//2+1]

        below_half = (ratio_1d < 0.5).float()
        eff_idx = below_half.argmax()
        k_eff = kx[eff_idx].item()
        eff_res_km = 1.0 / (k_eff + 1e-10)

        hi_k_mask = kx > 0.1
        psd_ratio_hik = ratio_1d[hi_k_mask].mean().item() if hi_k_mask.any() else 0.0

        self.writer.add_scalar('spectral/effective_resolution_km', eff_res_km)
        self.writer.add_scalar('spectral/psd_ratio_high_k', psd_ratio_hik)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        if hasattr(self.criterion, 'set_epoch'):
            self.criterion.set_epoch(epoch)

        attr_freq = self.config['trainer'].get('attribution_freq', 25)
        if epoch % attr_freq == 0 and self.valid_data_loader is not None:
            self._log_channel_attribution(epoch)

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

            # P4: gradient histograms every N epochs (first batch only)
            hist_freq = self.config['trainer'].get('gradient_hist_freq', 50)
            if epoch % hist_freq == 0 and batch_idx == 0:
                for pname, param in self.model.named_parameters():
                    if param.grad is not None:
                        self.writer.add_histogram(f'gradients/{pname}', param.grad)

            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())

            # P2: log raw (unweighted) contribution of each loss term
            if hasattr(self.criterion, 'last_components'):
                for tname, v in self.criterion.last_components.items():
                    raw = v['raw']
                    self.writer.add_scalar(f'loss_terms/{tname}', raw.item() if isinstance(raw, torch.Tensor) else float(raw))
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
        if hasattr(self.criterion, 'set_epoch'):
            self.criterion.set_epoch(epoch)

        spec_freq = self.config['trainer'].get('spectral_log_freq', 10)

        # P3: set up seasonal RMSE accumulators
        # _iter_indices maps batch position → dataset index (set when shuffle=False + split>0).
        # When None (no split, full dataset in order), use np.arange as fallback.
        _season_map = {'DJF': [12, 1, 2], 'MAM': [3, 4, 5], 'JJA': [6, 7, 8], 'SON': [9, 10, 11]}
        _season_accum = {s: {'sum_sq': 0.0, 'n': 0} for s in _season_map}
        _has_coords = hasattr(self.valid_data_loader, 'dataset') and hasattr(self.valid_data_loader.dataset, 'get_coords')
        if _has_coords:
            _, _, _all_times = self.valid_data_loader.dataset.get_coords()
            _iter_idx = getattr(self.valid_data_loader, 'iter_indices', None)
            if _iter_idx is None:
                # No split: full dataset iterated sequentially
                _iter_idx = np.arange(len(_all_times))
        else:
            _all_times = None
            _iter_idx = None

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

                # P2: log raw (unweighted) contribution of each loss term
                if hasattr(self.criterion, 'last_components'):
                    for tname, v in self.criterion.last_components.items():
                        raw = v['raw']
                        self.writer.add_scalar(f'loss_terms/{tname}', raw.item() if isinstance(raw, torch.Tensor) else float(raw))

                mask = data[:, -1:, :, :]
                pred_lf = gaussian_blur(output, sigma=5)
                pred_hf = output - pred_lf
                tgt_lf = gaussian_blur(target, sigma=5)
                tgt_hf = target - tgt_lf
                self.writer.add_scalar('rmse/large_scale', masked_rmse(pred_lf, tgt_lf, mask).item())
                self.writer.add_scalar('rmse/small_scale', masked_rmse(pred_hf, tgt_hf, mask).item())
                self.writer.add_scalar('rmse/gradient', gradient_rmse(output, target, mask).item())

                # P1: spectral diagnostics on first batch every spec_freq epochs
                if epoch % spec_freq == 0 and batch_idx == 0:
                    self._log_spectral_diagnostics(output.detach(), target.detach(), mask.detach())

                # P3: accumulate per-season RMSE (O(1) memory — running sums only)
                if _iter_idx is not None:
                    batch_start = batch_idx * self.valid_data_loader.batch_size
                    batch_end   = batch_start + data.shape[0]
                    batch_ds_idx = _iter_idx[batch_start:batch_end]
                    for sample_i, ds_idx in enumerate(batch_ds_idx):
                        month = pd.Timestamp(_all_times[ds_idx]).month
                        for season, months in _season_map.items():
                            if month in months:
                                rmse_val = masked_rmse(
                                    output[sample_i:sample_i + 1],
                                    target[sample_i:sample_i + 1],
                                    mask[sample_i:sample_i + 1],
                                ).item()
                                _season_accum[season]['sum_sq'] += rmse_val ** 2
                                _season_accum[season]['n'] += 1
                                break
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # P3: log per-season RMSE at epoch end
        if _valid_idx is not None:
            self.writer.set_step((epoch - 1) * len(self.valid_data_loader), 'valid')
            for season, acc in _season_accum.items():
                if acc['n'] > 0:
                    self.writer.add_scalar(f'rmse/seasonal/{season}', (acc['sum_sq'] / acc['n']) ** 0.5)

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
