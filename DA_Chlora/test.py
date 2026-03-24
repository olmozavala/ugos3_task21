# %%
import argparse
import torch
import pickle
from collections import defaultdict
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from utils import prepare_device
from os.path import join
import os
import yaml
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import torch.nn.functional as F
from dynamic_functions import compute_psd, plot_psd
import cmocean.cm as cmo
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
# Only for jvelasco (torch has some problems to compile models)
import torch._dynamo
torch._dynamo.config.suppress_errors = True

styles = {
    "sst": cmo.thermal,
    "chlora": cmo.algae,
    "ssh": cmo.balance,
    "ssh_track": cmo.balance,
    "swot": cmo.balance,
    "fused_ssh": cmo.balance,
}

# ------------------------------------------------------
# TODO: find a way to make the plot_data more flexible and robust, specifically handle cases where only one input variable is provided.
def plot_data(data: np.ndarray,
              target: np.ndarray,
              previous_days: int,
              file_name: str, 
              lats: np.ndarray, 
              lons: np.ndarray, 
              input_vars: list[str],
              time):
    """
    Plot the data
    """
    frame_time = np.datetime64(time)
    formatted_time = pd.Timestamp(frame_time).strftime('%Y-%m-%d')
    X_masked = np.copy(data)
    for i in range(X_masked.shape[0]-1):
        X_masked[i, :, :] = np.where(X_masked[-1, :, :] == 0, np.nan, X_masked[i, :, :])


    fig, ax = plt.subplots(len(input_vars), previous_days + 2, 
                            figsize=(11*int(np.ceil((previous_days+1)/2)), 4*len(input_vars)))
    
    for ii, var in enumerate(input_vars):
        for jj in range(previous_days):
            cur_index = jj * len(input_vars) + ii

            ax[ii, jj].pcolormesh(lons, lats, X_masked[cur_index, :, :], cmap=styles[var])
            ax[ii, jj].set_title(f"{var} at {previous_days - jj} days before")
    ax[0, -2].pcolormesh(lons, lats, X_masked[-3, :, :], cmap=styles["ssh"])
    ax[0, -2].set_title("Mean SSH 2 days before")
    ax[1, -2].pcolormesh(lons, lats, X_masked[-2, :, :], cmap=styles["ssh"])
    ax[1, -2].set_title("Mean SSH 1 day before")
    ax[0, -1].pcolormesh(lons, lats, X_masked[-1, :, :], cmap="gray")
    ax[0, -1].set_title("Gulf Mask")
    ax[1, -1].pcolormesh(lons, lats, target, cmap=styles["ssh"])
    ax[1, -1].set_title("Target SSH")

    fig.suptitle(f"Data for {formatted_time}", y=1, fontsize=18)
    fig.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close()

    

def plot_predictions(
                     target: np.ndarray,
                     output: np.ndarray, 
                     file_name: str, 
                     lats: np.ndarray, 
                     lons: np.ndarray, 
                     time):
    """
    Plot the predictions
    """
    #print(f"Time: {time}")
    frame_time = np.datetime64(time)
    formatted_time = pd.Timestamp(frame_time).strftime('%Y-%m-%d')
    grad_true = np.gradient(target)
    grad_pred = np.gradient(output)
    mag_true = np.sqrt(grad_true[0]**2 + grad_true[1]**2)
    mag_pred = np.sqrt(grad_pred[0]**2 + grad_pred[1]**2)
    
    diff_ssh = target - output
    diff_ssh_grad = mag_true - mag_pred
    # Percentil 95 of the target and output
    target_95 = np.percentile(np.abs(target), 95)
    output_95 = np.percentile(np.abs(output), 95)
    diff_ssh_95 = np.percentile(np.abs(diff_ssh), 95)
    diff_ssh_grad_95 = np.percentile(np.abs(diff_ssh_grad), 95)
    vmax = np.max([target_95, output_95])
    vmax_diff = np.max([diff_ssh_95])
    vmax_diff_grad = np.max([diff_ssh_grad_95])
    # Percentil 99 of the gradient of the target and output
    gradient_target_99 = np.percentile(np.abs(grad_true), 99)
    gradient_output_99 = np.percentile(np.abs(grad_pred), 99)
    vmax_gradient = np.max([gradient_target_99, gradient_output_99])
    fig, ax = plt.subplots(2, 3, figsize=(20, 10))
    ax = ax.flatten()
    img1 =ax[0].pcolormesh(lons, lats, target, vmin=-vmax, vmax=vmax, cmap=cmo.balance)
    plt.colorbar(img1)
    ax[0].set_title('Target SSH')

    img2 = ax[1].pcolormesh(lons, lats, output, vmin=-vmax, vmax=vmax, cmap=cmo.balance)
    plt.colorbar(img2)
    ax[1].set_title('Output SSH')

    img3 = ax[2].pcolormesh(lons, lats, diff_ssh, vmin=-vmax_diff, vmax=vmax_diff,cmap=cmo.balance)
    plt.colorbar(img3)
    ax[2].set_title('Difference SSH')

    img4 = ax[3].pcolormesh(lons, lats, mag_true, vmin=0, vmax=vmax_gradient, cmap=cmo.balance)
    plt.colorbar(img4)
    ax[3].set_title('Gradient of True SSH')

    img5 = ax[4].pcolormesh(lons, lats, mag_pred, vmin=0, vmax=vmax_gradient, cmap=cmo.balance)
    plt.colorbar(img5)
    ax[4].set_title('Gradient of Output SSH')

    img6 = ax[5].pcolormesh(lons, lats, diff_ssh_grad, vmin=0, vmax=vmax_diff_grad, cmap=cmo.rain)
    plt.colorbar(img6)
    ax[5].set_title('Difference Gradient of SSH')

    plt.tight_layout()
    fig.suptitle(f"Prediction for {formatted_time}", y=1.1, fontsize=18)
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close()


def _plot_one_sample(args):
    """
    Worker for parallel plotting: run plot_predictions, plot_data, and plot_psd
    for a single sample. Must be a top-level function for ProcessPoolExecutor.
    """
    (target_np, output_np, mask_np, data_np, ex_num, sample_time, output_dir,
     model_name, lats, lons, days_before, input_vars, mean_ssh) = args
    target_m = target_np * mask_np
    output_m = output_np * mask_np
    plot_predictions(
        target_m, output_m,
        join(output_dir, f"{model_name}_ex_{ex_num:03d}_predictions.png"),
        lats, lons, sample_time,
    )
    try:
        plot_data(
            data_np, target_m, days_before,
            join(output_dir, f"{model_name}_ex_{ex_num:03d}.png"),
            lats, lons, input_vars, sample_time,
        )
    except Exception as e:
        print(f"Error plotting data: {e}")

    psd_output = compute_psd(output_np - mean_ssh, lats, lons)
    psd_target = compute_psd(target_np - mean_ssh, lats, lons)
    plot_psd(
        [psd_output, psd_target],
        add_reference=True,
        labels=["ML spectrum", "Target spectrum"],
        path=output_dir,
        filename=f"{model_name}_ex_{ex_num:03d}_psd.png",
    )


# ---------------------------------------------------------------------------

def main(config):
    logger = config.get_logger("test")

    # Resolve the weights directory (CLI > config.yml > last_run.txt breadcrumb).
    weights_dir = config.weights_dir
    if weights_dir is None:
        raise ValueError(
            "Could not determine weights_dir. Provide one of:\n"
            "  • python test.py -c config.yml -wd /path/to/run\n"
            "  • Set tester.weights_dir in config.yml\n"
            "  • Run train.py first (it writes last_run.txt automatically)"
        )
    logger.info(f"Using weights from: {weights_dir}")

    # ------------------------------------------------------------------
    # Data loader
    # ------------------------------------------------------------------
    dl_args     = config["data_loader"]["args"]
    data_loader = getattr(module_data, config["data_loader"]["type"])(
        dl_args["data_dir"],
        batch_size=dl_args["batch_size"],
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=dl_args["num_workers"],
        previous_days=dl_args["previous_days"],
        dataset_type=dl_args["dataset_type"],
        input_vars=None,#dl_args["input_vars"],
    )

    # ------------------------------------------------------------------
    # Load scalers
    # ------------------------------------------------------------------
    with open(join(dl_args["data_dir"], "scalers.pkl"), "rb") as f:
        scalers = pickle.load(f)

    mean_ssh = scalers["ssh"]["mean"]
    std_ssh  = scalers["ssh"]["std"]

    # ------------------------------------------------------------------
    # Read the training config that was saved alongside the weights.
    # This ensures the model architecture matches what was trained.
    # ------------------------------------------------------------------
    training_cfg_path = join(weights_dir, "config.yml")
    with open(training_cfg_path, "r") as f:
        training_config = yaml.safe_load(f)

    model_name = training_config["name"]

    # ------------------------------------------------------------------
    # Output directory  →  <tester.output_dir>/<model_name>/<run_id>
    # ------------------------------------------------------------------
    run_id     = str(weights_dir).split(os.sep)[-1]   # last path component
    output_dir = join(config["tester"]["output_dir"], model_name, run_id)
    os.makedirs(output_dir, exist_ok=True)

    # Persist a copy of the test config alongside results for reproducibility.
    with open(join(output_dir, "test_config.yml"), "w") as f:
        yaml.dump(config.config, f, default_flow_style=False, sort_keys=False)

    # ------------------------------------------------------------------
    # Build model and load weights
    # ------------------------------------------------------------------
    weights_file = join(weights_dir, "model_best.pth")
    model = config.init_obj("arch", module_arch)
    logger.info(model)

    loss_fn    = module_loss.build_loss(config["loss"])
    metric_fns = [getattr(module_metric, met) for met in config["metrics"]]

    logger.info(f"Loading checkpoint: {weights_file} …")

    device, _ = prepare_device(1)

    checkpoint = torch.load(weights_file, weights_only=False, map_location=device)
    state_dict = checkpoint["state_dict"]

    
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    model = torch.compile(model)
    model.load_state_dict(state_dict)

    model.eval()

    torch.set_float32_matmul_precision("medium")

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------
    batch_size    = dl_args["batch_size"]
    dataset_type  = dl_args["dataset_type"]
    total_loss    = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    lats, lons, time = data_loader.dataset.get_coords()
    input_vars = data_loader.dataset.input_vars
    days_before = data_loader.dataset.previous_days

    validation_loss  = []
    validation_times = []  # sample time for each validation_loss entry
    psd_output_list  = []
    psd_target_list  = []
    save_predictions = True
    n_plot_workers = len(os.sched_getaffinity(0)) - 2
    print(f"Number of workers for plotting: {n_plot_workers}")
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            mask   = data[:, -1, :, :]
            target = target.squeeze()
            output = model(data)
            sample_time = time[i*batch_size : i*batch_size + batch_size]

            # Inverse-scale to physical units
            output = output * torch.tensor(std_ssh).to(device) + torch.tensor(mean_ssh).to(device)
            target = target * torch.tensor(std_ssh).to(device) + torch.tensor(mean_ssh).to(device)

            # Qualitative plots in parallel
            plot_tasks = []
            for j in range(output.shape[0]):
                ex_num = i * batch_size + j + 1
                target_np = target[j].detach().cpu().numpy()
                output_np = output[j].detach().cpu().numpy()
                mask_np = mask[j].detach().cpu().numpy()
                data_np = data[j].detach().cpu().numpy()
                plot_tasks.append((
                    target_np, output_np, mask_np, data_np, ex_num, sample_time[j],
                    output_dir, model_name, lats, lons, days_before, input_vars, mean_ssh,
                ))
            with ProcessPoolExecutor(max_workers=n_plot_workers) as executor:
                list(executor.map(_plot_one_sample, plot_tasks))

            # PSD accumulation over all samples
            for ii in range(output.shape[0]):
                psd_output = compute_psd(output[ii].detach().cpu().numpy() - mean_ssh, lats, lons)
                psd_target = compute_psd(target[ii].detach().cpu().numpy() - mean_ssh, lats, lons)
                if ii == 0:
                    k_bins = psd_output[1]
                psd_output_list.append(psd_output[0])
                psd_target_list.append(psd_target[0])

            # Mask-aware MSE loss
            loss  = F.mse_loss(output * mask, target * mask, reduction="sum")
            valid = mask.sum()
            loss  = loss / (valid + 1e-8)

            total_loss += loss.item() * data.shape[0]

            for j, metric in enumerate(metric_fns):
                total_metrics[j] += metric(output, target) * data.shape[0]

            # Per-sample mask-aware RMSE
            diff2 = ((output - target) ** 2) * mask
            rmse  = torch.sqrt(diff2.sum(dim=(1, 2)) / (mask.sum(dim=(1, 2)) + 1e-8))
            validation_loss.extend(float(r) for r in rmse.cpu().numpy())
            # One time per sample (align with output.shape[0] in case last batch is smaller than slice)
            n_in_batch = output.shape[0]
            validation_times.extend(
                pd.Timestamp(np.datetime64(sample_time[j])).strftime("%Y-%m-%d")
                for j in range(n_in_batch)
            )

            if save_predictions:
                for j in range(output.shape[0]):
                    xr.Dataset(
                        {
                            "output": (["latitude", "longitude"], output[j].detach().cpu().numpy()),
                            "target": (["latitude", "longitude"], target[j].detach().cpu().numpy()),
                            "mask": (["latitude", "longitude"], mask[j].detach().cpu().numpy()),
                        },
                        coords={"latitude": lats, "longitude": lons},
                        attrs={"date": pd.Timestamp(np.datetime64(sample_time[j])).strftime("%Y-%m-%d")},
                    ).to_netcdf(join(output_dir, f"pred_batch_{i}_sample_{j}.nc"))

    # ------------------------------------------------------------------
    # Aggregate results
    # ------------------------------------------------------------------
    # Per-sample RMSE CSV (date, rmse)
    with open(join(output_dir, "loss.csv"), "w") as f:
        f.write("date,rmse\n")
        for t, v in zip(validation_times, validation_loss):
            f.write(f"{t},{v}\n")

    # Mean PSD plot
    psd_output_array = np.array(psd_output_list).mean(axis=0)
    psd_target_array = np.array(psd_target_list).mean(axis=0)
    plot_psd(
        [(psd_output_array, k_bins), (psd_target_array, k_bins)],
        add_reference=True,
        labels=["ML spectrum", "DUACS spectrum"],
        path=output_dir,
        title="Mean PSD of the output and target",
        filename="mean_psd.png",
    )

    # PSD pickles
    for name, obj in [("psd_output", psd_output_list),
                      ("psd_target", psd_target_list),
                      ("k_bins",     k_bins)]:
        with open(join(output_dir, f"{name}.pkl"), "wb") as f:
            pickle.dump(obj, f)

    # RMSE scatter plot (date on x-axis, mean and mean±std as horizontal lines)
    mean_rmse = np.mean(validation_loss)
    std_rmse = np.std(validation_loss)
    dates = pd.to_datetime(validation_times)
    fig, ax = plt.subplots()
    ax.scatter(dates, validation_loss, alpha=0.7)
    ax.axhline(mean_rmse, color="red", linestyle="--", linewidth=1.5, label=f"Mean RMSE = {mean_rmse:.4f} m")
    ax.axhline(mean_rmse + std_rmse, color="gray", linestyle="--", linewidth=1, label=f"Mean + 1 std = {mean_rmse + std_rmse:.4f} m")
    ax.axhline(mean_rmse - std_rmse, color="gray", linestyle="--", linewidth=1, label=f"Mean - 1 std = {mean_rmse - std_rmse:.4f} m")
    ax.set_xlabel("Date")
    ax.set_ylabel("RMSE (m)")
    ax.set_title(f"Per-sample RMSE (mean = {mean_rmse:.4f} m, std = {std_rmse:.4f} m)")
    ax.legend(loc="upper right", fontsize=8)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(join(output_dir, "validation_loss.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Monthly RMSE: group by month (1-12) across all years
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly_rmse = defaultdict(list)
    for t, v in zip(validation_times, validation_loss):
        month_num = t[5:7]  # "MM" from "YYYY-MM-DD"
        monthly_rmse[month_num].append(v)
    monthly_means = {m: np.mean(vals) for m, vals in monthly_rmse.items()}
    monthly_stds = {m: np.std(vals) for m, vals in monthly_rmse.items()}
    # Sort by month number (01..12), only months that have data
    months_sorted = sorted(monthly_means.keys(), key=lambda m: int(m))
    best3_months = sorted(monthly_means.keys(), key=lambda m: monthly_means[m])[:3]
    worst3_months = sorted(monthly_means.keys(), key=lambda m: monthly_means[m])[-3:][::-1]

    # Bar plot: one bar per month with std as error bars
    fig, ax = plt.subplots(figsize=(max(8, len(months_sorted) * 0.6), 5))
    x_pos = np.arange(len(months_sorted))
    means = [monthly_means[m] for m in months_sorted]
    stds = [monthly_stds[m] for m in months_sorted]
    ax.bar(x_pos, means, yerr=stds, capsize=3, color="steelblue", edgecolor="navy", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([month_names[int(m) - 1] for m in months_sorted], rotation=45, ha="right")
    ax.set_ylabel("RMSE (m)")
    ax.set_xlabel("Month")
    ax.set_title("Monthly mean RMSE (±1 std) across all years")
    plt.tight_layout()
    plt.savefig(join(output_dir, "validation_loss_monthly.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Summary text: statistics and top 5 best / worst samples (index = 1-based, matches ex_XXX plot names)
    sorted_by_rmse = sorted(
        zip(range(1, len(validation_loss) + 1), validation_times, validation_loss),
        key=lambda x: x[2],
    )
    n = len(validation_loss)
    top5_best = sorted_by_rmse[: min(5, n)]
    top5_worst = sorted_by_rmse[-min(5, n) :][::-1]
    summary_lines = [
        "Validation RMSE summary",
        "=" * 50,
        f"  Samples:     {n}",
        f"  Mean RMSE:   {mean_rmse:.6f} m",
        f"  Std RMSE:    {std_rmse:.6f} m",
        f"  Min RMSE:    {np.min(validation_loss):.6f} m",
        f"  Max RMSE:    {np.max(validation_loss):.6f} m",
        "",
        "Monthly RMSE (mean ± std, aggregated across all years):",
        "-" * 50,
        "  month      mean (m)    std (m)   n",
    ]
    for m in months_sorted:
        name = month_names[int(m) - 1]
        summary_lines.append(
            f"  {name:<4}   {monthly_means[m]:.6f}   {monthly_stds[m]:.6f}   {len(monthly_rmse[m])}"
        )
    summary_lines.extend([
        "",
        "Top 3 best months (lowest mean RMSE):",
        "-" * 50,
    ])
    for m in best3_months:
        name = month_names[int(m) - 1]
        summary_lines.append(f"  {name}   mean = {monthly_means[m]:.6f} m   std = {monthly_stds[m]:.6f} m   n = {len(monthly_rmse[m])}")
    summary_lines.extend([
        "",
        "Top 3 worst months (highest mean RMSE):",
        "-" * 50,
    ])
    for m in worst3_months:
        name = month_names[int(m) - 1]
        summary_lines.append(f"  {name}   mean = {monthly_means[m]:.6f} m   std = {monthly_stds[m]:.6f} m   n = {len(monthly_rmse[m])}")
    summary_lines.extend([
        "",
        "Top 5 best (lowest RMSE):",
        "-" * 50,
        "  index    date         rmse (m)",
    ])
    for idx, date, rmse in top5_best:
        summary_lines.append(f"  {idx:<7}  {date}   {rmse:.6f}")
    summary_lines.extend([
        "",
        "Top 5 worst (highest RMSE):",
        "-" * 50,
        "  index    date         rmse (m)",
    ])
    for idx, date, rmse in top5_worst:
        summary_lines.append(f"  {idx:<7}  {date}   {rmse:.6f}")
    summary_text = "\n".join(summary_lines)
    with open(join(output_dir, "validation_loss_summary.txt"), "w") as f:
        f.write(summary_text)
        f.write("\n")
    print("\n" + summary_text)

    n_samples = len(data_loader.sampler)
    log = {"loss": total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples
        for i, met in enumerate(metric_fns)
    })
    logger.info(log)
    print(f"\nResults saved to: {output_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    args = argparse.ArgumentParser(description="UNet SSH predictor - testing")
    args.add_argument("-c", "--config", default="config.yml", type=str,
                      help="Path to config.yml (default: config.yml)")
    args.add_argument("-r", "--resume", default=None, type=str,
                      help="Path to a checkpoint (default: None)")
    args.add_argument("-d", "--device", default=None, type=str,
                      help="Comma-separated CUDA device indices (default: all visible)")
    args.add_argument("-i", "--uid", default=None, type=str,
                      help=(
                          "Unique run identifier prefix used during training. "
                          "Reconstructs the weights directory from config + uid, "
                          "identical to the path train.py created. "
                          "If omitted, falls back to tester.weights_dir in config.yml."
                      ))
    args.add_argument(
        "-wd", "--weights_dir", default=None, type=str,
        help=(
            "Path to the experiment directory that contains model_best.pth and "
            "config.yml (the copy saved by train.py).  "
            "When omitted, falls back to tester.weights_dir in config.yml."
        ),
    )

    config = ConfigParser.from_args(args)
    main(config)
    sys.exit(0)