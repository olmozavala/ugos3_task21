import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import cmocean as cmo
from scipy.ndimage import gaussian_filter
from math import ceil, floor
import cartopy.feature as cfeature

def plot_dataset_data(idx, all_data, lats, lons):
    """
    Plots the original and normalized fields for a given index in the dataset. It uses the coastlines from the original fields.
    """
    print("Plotting some examples...")
    fig, axs = plt.subplots(2, 5, figsize=(20, 8), subplot_kw={'projection': ccrs.PlateCarree()})

    # Set up the cartopy projection
    proj = ccrs.PlateCarree()
    fraction = 0.043
    pad = 0.04

    land = cfeature.NaturalEarthFeature('physical', 'land', '10m', 
                                        edgecolor='black', 
                                        facecolor='lightgrey')

    # Plot the original fields with coastlines and colorbars
    im1 = axs[0, 0].imshow(np.flipud(all_data['sst'][idx, :, :]), 
                           extent=[lons[0], lons[-1], lats[0], lats[-1]], 
                           transform=proj, cmap=cmo.cm.thermal)
    axs[0, 0].coastlines()
    axs[0, 0].set_title('SST')
    axs[0, 0].add_feature(land, zorder=1)
    fig.colorbar(im1, ax=axs[0, 0], orientation='vertical', pad=pad, fraction=fraction)

    # Plot Chlor-a
    im2 = axs[0, 1].imshow(np.flipud(all_data['chlora'][idx, :, :]), 
                           extent=[lons[0], lons[-1], lats[0], lats[-1]], 
                           transform=proj, cmap=cmo.cm.algae)
    axs[0, 1].coastlines()
    axs[0, 1].set_title('Chl-a')
    axs[0, 1].add_feature(land, zorder=1)
    fig.colorbar(im2, ax=axs[0, 1], orientation='vertical', pad=pad, fraction=fraction)

    # Plot SSH
    im3 = axs[0, 2].imshow(np.flipud(all_data['ssh'][idx, :, :]), 
                           extent=[lons[0], lons[-1], lats[0], lats[-1]], 
                           transform=proj, cmap=cmo.cm.balance)
    axs[0, 2].coastlines()
    axs[0, 2].set_title('SSH')
    axs[0, 2].add_feature(land, zorder=1)
    fig.colorbar(im3, ax=axs[0, 2], orientation='vertical', pad=pad, fraction=fraction)

    # Plot SSH Track
    sigma = 1
    temp_ssh_track = all_data['ssh_track'][idx, :, :].fillna(0)
    smooth_ssh_masked_data = gaussian_filter(temp_ssh_track, sigma=sigma)
    smooth_ssh_masked_data = np.where(smooth_ssh_masked_data != 0, smooth_ssh_masked_data, np.nan)
    img7 = axs[0, 3].imshow(np.flipud(smooth_ssh_masked_data), 
                            extent=[lons[0], lons[-1], lats[0], lats[-1]], 
                            transform=proj, cmap=cmo.cm.deep)
    axs[0, 3].coastlines()
    axs[0, 3].add_feature(land, zorder=1)
    axs[0, 3].set_title('SSH Track')
    fig.colorbar(img7, ax=axs[0, 3], orientation='vertical', pad=pad, fraction=fraction)

    # Plot SWOT
    im8 = axs[0, 4].imshow(np.flipud(all_data['swot'][idx, :, :]), 
                            extent=[lons[0], lons[-1], lats[0], lats[-1]], 
                            transform=proj, cmap=cmo.cm.delta)
    axs[0, 4].coastlines()
    axs[0, 4].add_feature(land, zorder=1)
    axs[0, 4].set_title('SWOT')
    fig.colorbar(im8, ax=axs[0, 4], orientation='vertical', pad=pad, fraction=fraction)

    # ============================ Normalized Fields ============================
    # Plot SST Normalized
    im4 = axs[1, 0].imshow(np.flipud(all_data['sst_normalized'][idx, :, :]), 
                           extent=[lons[0], lons[-1], lats[0], lats[-1]], 
                           transform=proj, cmap=cmo.cm.thermal)
    axs[1, 0].coastlines()
    axs[1, 0].add_feature(land, zorder=1)
    axs[1, 0].set_title('SST Normalized')
    fig.colorbar(im4, ax=axs[1, 0], orientation='vertical', pad=pad, fraction=fraction)

    # Plot Chlor-a Normalized
    im5 = axs[1, 1].imshow(np.flipud(all_data['chlora_normalized'][idx, :, :]), 
                           extent=[lons[0], lons[-1], lats[0], lats[-1]], 
                           transform=proj, cmap=cmo.cm.algae)
    axs[1, 1].coastlines()
    axs[1, 1].add_feature(land, zorder=1)
    axs[1, 1].set_title('Chl-a Normalized')
    fig.colorbar(im5, ax=axs[1, 1], orientation='vertical', pad=pad, fraction=fraction)

    # Plot SSH Normalized
    im6 = axs[1, 2].imshow(np.flipud(all_data['ssh_normalized'][idx, :, :]), 
                           extent=[lons[0], lons[-1], lats[0], lats[-1]], 
                           transform=proj, cmap=cmo.cm.balance)
    axs[1, 2].coastlines()
    axs[1, 2].add_feature(land, zorder=1)
    axs[1, 2].set_title('SSH Normalized')
    fig.colorbar(im6, ax=axs[1, 2], orientation='vertical', pad=pad, fraction=fraction)

    # Plot SSH Track Normalized
    temp_ssh_track_normalized = all_data['ssh_track_normalized'][idx, :, :].fillna(0)
    smooth_ssh_track_normalized = gaussian_filter(temp_ssh_track_normalized, sigma=sigma)
    smooth_ssh_track_normalized = np.where(smooth_ssh_track_normalized != 0, smooth_ssh_track_normalized, np.nan)
    img8 = axs[1, 3].imshow(np.flipud(smooth_ssh_track_normalized), 
                            extent=[lons[0], lons[-1], lats[0], lats[-1]], 
                            transform=proj, cmap=cmo.cm.deep)
    axs[1, 3].coastlines()
    axs[1, 3].add_feature(land, zorder=1)
    axs[1, 3].set_title('SSH Track Normalized')
    fig.colorbar(img8, ax=axs[1, 3], orientation='vertical', pad=pad, fraction=fraction)

    # Plot SWOT Normalized
    im9 = axs[1, 4].imshow(np.flipud(all_data['swot_normalized'][idx, :, :]), 
                            extent=[lons[0], lons[-1], lats[0], lats[-1]], 
                            transform=proj, cmap=cmo.cm.delta)
    axs[1, 4].coastlines()
    axs[1, 4].add_feature(land, zorder=1)
    axs[1, 4].set_title('SWOT Normalized')
    fig.colorbar(im9, ax=axs[1, 4], orientation='vertical', pad=pad, fraction=fraction)

    # Adjust layout for better visualization
    plt.tight_layout()

    # Save the figure
    file_name = f"/unity/f1/ozavala/OUTPUTS/HR_SSH_from_Chlora/data_loaders/normalized_fields_with_coastlines_colorbars_{idx}.png"
    print(f"Saving figure to {file_name}")
    plt.savefig(file_name)
    plt.close()


def plot_single_batch_element(X, Y, input_names, days_before, output_file, lats, lons, dataset_type="regular"):
    # It assumes the input names do not contain the mask.

    proj = ccrs.PlateCarree()
    fig, axs = plt.subplots(len(input_names), days_before+1, figsize=(11*int(ceil((days_before+1)/2)), 4*len(input_names)), 
                            subplot_kw={'projection': proj})

    # Apply the mask to all the input fields except the last one. Transform to nan where the mask is 0 
    X_masked = np.copy(X)
    for i in range(X.shape[0]-1):
        X_masked[i, :, :] = np.where(X[-1, :, :] == 0, np.nan, X[i, :, :])

    land = cfeature.NaturalEarthFeature('physical', 'land', '10m', 
                                        edgecolor='black', 
                                        facecolor='lightgrey')
    # Colorbar settings
    fraction = 0.040
    pad = 0.04

    for i, input_name in enumerate(input_names):
        for j in range(days_before):
            cur_index = j * len(input_names) + i
            # Set colormap based on input_name
            title = ""
            if input_name == 'sst':
                cmap = cmo.cm.thermal
                title = "SST"
            elif input_name == 'chlora':
                cmap = cmo.cm.algae
                title = "Chlora"
            elif input_name == 'ssh' or input_name == 'ssh_track':
                cmap = cmo.cm.balance
                title = "SSH"
            elif input_name == 'swot':
                cmap = cmo.cm.delta
                title = "SWOT"
            else:
                cmap = 'viridis'  # Default colormap

            # Apply gaussian filter to ssh_track
            if input_name == 'ssh_track':
                X_masked[cur_index, :, :] = gaussian_filter(X_masked[cur_index, :, :], sigma=2)

            X_masked[cur_index, :, :] = np.where(X_masked[cur_index, :, :] == 0, np.nan, X_masked[cur_index, :, :])
            
            im = axs[i, j].imshow(np.flipud(X_masked[cur_index, :, :]), cmap=cmap, extent=[lons[0], lons[-1], lats[0], lats[-1]], transform=proj)
            axs[i, j].set_title(f"{title} at {days_before - j} days before")
            axs[i, j].coastlines()
            axs[i, j].add_feature(land, zorder=1)
            # Show x and y labels
            axs[i, j].set_xlabel("Longitude")
            axs[i, j].set_ylabel("Latitude")
            fig.colorbar(im, ax=axs[i, j], orientation='vertical', fraction=fraction, pad=pad)

    # Plot the mask in the last column, first row
    im_mask = axs[0, -1].imshow(np.flipud(X[-1, :, :]), extent=[lons[0], lons[-1], lats[0], lats[-1]], transform=proj)
    axs[0, -1].coastlines()
    axs[0, -1].add_feature(land, zorder=1)
    axs[0, -1].set_title("Mask")
    axs[0, -1].set_xlabel("Longitude")
    axs[0, -1].set_ylabel("Latitude")
    fig.colorbar(im_mask, ax=axs[0, -1], orientation='vertical', fraction=fraction, pad=pad)

    # Plot the true SSH
    im_true = axs[-1, -1].imshow(np.flipud(Y[:, :]), cmap=cmo.cm.balance, extent=[lons[0], lons[-1], lats[0], lats[-1]], transform=proj)
    axs[-1, -1].set_title("True SSH")
    axs[-1, -1].coastlines()
    axs[-1, -1].add_feature(land, zorder=1)
    axs[-1, -1].set_xlabel("Longitude")
    axs[-1, -1].set_ylabel("Latitude")
    fig.colorbar(im_true, ax=axs[-1, -1], orientation='vertical', fraction=fraction, pad=pad)

    if dataset_type == "extended":
        # Plot the previous SSH + noise
        im_true = axs[1, -1].imshow(np.flipud(X[-2, :, :]), cmap=cmo.cm.balance, extent=[lons[0], lons[-1], lats[0], lats[-1]], transform=proj)
        axs[1, -1].set_title("Previous SSH + noise")
        axs[1, -1].coastlines()
        axs[1, -1].add_feature(land, zorder=1)
        axs[1, -1].set_xlabel("Longitude")
        axs[1, -1].set_ylabel("Latitude")
        fig.colorbar(im_true, ax=axs[1, -1], orientation='vertical', fraction=fraction, pad=pad)

        # Plot the two steps before SSH + noise
        im_pred = axs[2, -1].imshow(np.flipud(X[-3, :, :]), cmap=cmo.cm.balance, extent=[lons[0], lons[-1], lats[0], lats[-1]], transform=proj)
        axs[2, -1].set_title("Two steps before SSH + noise")
        axs[2, -1].coastlines()
        axs[2, -1].add_feature(land, zorder=1)
        axs[2, -1].set_xlabel("Longitude")
        axs[2, -1].set_ylabel("Latitude")
        fig.colorbar(im_pred, ax=axs[2, -1], orientation='vertical', fraction=fraction, pad=pad)
    else:
        # Hide axes for empty subplots
        for i in range(1, len(input_names)-1):
            axs[i, -1].axis('off')

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_predictions(X, Y, Y_pred, output_file, lats, lons, dataset_type="regular"):
    input_names = ["sst", "chlora", "ssh_track", "swot"]
    days_before = floor(X.shape[0]/len(input_names))

    land = cfeature.NaturalEarthFeature('physical', 'land', '10m', 
                                        edgecolor='black', 
                                        facecolor='lightgrey')
    
    proj = ccrs.PlateCarree()
    # Create a figure with the new dimensions
    fig, axs = plt.subplots(len(input_names), days_before + 1, 
                            figsize=(12*int(ceil((days_before+1)/2)), 20), 
                            subplot_kw={'projection': proj})
    
    # Apply the mask to all input fields except the last one
    X_masked = np.copy(X)
    for i in range(X.shape[0]-1):
        X_masked[i, :, :] = np.where(X[-1, :, :] == 0, np.nan, X[i, :, :])

    # Colorbar settings
    fraction = 0.040
    pad = 0.04

    # Plot input fields
    for i, input_name in enumerate(input_names):
        for j in range(days_before):
            cur_index = j * len(input_names) + i
            
            # Set colormap based on input_name
            if input_name == 'sst':
                cmap = cmo.cm.thermal
            elif input_name == 'chlora':
                cmap = cmo.cm.algae
            elif input_name == 'ssh' or input_name == 'ssh_track':
                cmap = cmo.cm.balance
            elif input_name == 'swot':
                cmap = cmo.cm.delta
            else:
                cmap = 'viridis'  # Default colormap

            # Apply gaussian filter to ssh_track
            if input_name == 'ssh_track':
                X_masked[cur_index, :, :] = gaussian_filter(X_masked[cur_index, :, :], sigma=2)

            X_masked[cur_index, :, :] = np.where(X_masked[cur_index, :, :] == 0, np.nan, X_masked[cur_index, :, :])
            
            im = axs[i, j].imshow(np.flipud(X_masked[cur_index, :, :]), cmap=cmap, extent=[lons[0], lons[-1], lats[0], lats[-1]], transform=proj)
            axs[i, j].set_title(f"{input_name} at {days_before - j - 1} days before")
            fig.colorbar(im, ax=axs[i, j], orientation='vertical', fraction=fraction, pad=pad)
            axs[i, j].set_xlabel("Longitude")
            axs[i, j].set_ylabel("Latitude")
            axs[i, j].coastlines()
            axs[i, j].add_feature(land, zorder=1)

    # Use the last column for mask, true SSH, predicted SSH, and difference
    # Plot the mask
    im_mask = axs[0, -1].imshow(np.flipud(X[-1, :, :]), extent=[lons[0], lons[-1], lats[0], lats[-1]], transform=proj)
    axs[0, -1].set_title("Mask")
    fig.colorbar(im_mask, ax=axs[0, -1], orientation='vertical', fraction=fraction, pad=pad)

    if dataset_type == "extended":
        # Plot the previous SSH + noise and two steps before SSH + noise
        im_true = axs[1, -1].imshow(np.flipud(X[-2, :, :]), cmap=cmo.cm.balance, extent=[lons[0], lons[-1], lats[0], lats[-1]], transform=proj)
        axs[1, -1].set_title("Previous SSH + noise")
        fig.colorbar(im_true, ax=axs[1, -1], orientation='vertical', fraction=fraction, pad=pad)
        axs[1, -1].coastlines()
        axs[1, -1].set_xlabel("Longitude")
        axs[1, -1].set_ylabel("Latitude")
        axs[1, -1].add_feature(land, zorder=1)

        im_true = axs[2, -1].imshow(np.flipud(X[-3, :, :]), cmap=cmo.cm.balance, extent=[lons[0], lons[-1], lats[0], lats[-1]], transform=proj)
        axs[2, -1].set_title("Two steps before SSH + noise")
        fig.colorbar(im_true, ax=axs[2, -1], orientation='vertical', fraction=fraction, pad=pad)
        axs[2, -1].coastlines()
        axs[2, -1].add_feature(land, zorder=1)
        axs[2, -1].set_xlabel("Longitude")
        axs[2, -1].set_ylabel("Latitude")
    else:
        # Hide axes for empty subplots
        for i in range(1, len(input_names)-1):
            axs[i, -1].axis('off')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    # Make a separate plot for the true SSH, predicted SSH, and difference
    fig, axs = plt.subplots(2, 3, figsize=(20, 10), subplot_kw={'projection': proj})

    # Plot the true SSH
    im_true = axs[0, 0].imshow(np.flipud(Y[:, :]), cmap=cmo.cm.balance, extent=[lons[0], lons[-1], lats[0], lats[-1]], transform=proj)
    axs[0, 0].set_title("True SSH")
    axs[0, 0].coastlines()
    axs[0, 0].add_feature(land, zorder=1)
    fig.colorbar(im_true, ax=axs[0, 0], orientation='vertical', fraction=fraction, pad=pad)

    # Plot the predicted SSH
    im_pred = axs[0, 1].imshow(np.flipud(Y_pred[:, :]), cmap=cmo.cm.balance, extent=[lons[0], lons[-1], lats[0], lats[-1]], transform=proj)
    axs[0, 1].set_title("Predicted SSH")
    axs[0, 1].coastlines()
    axs[0, 1].add_feature(land, zorder=1)
    fig.colorbar(im_pred, ax=axs[0, 1], orientation='vertical', fraction=fraction, pad=pad)

    # Plot the difference
    diff = Y - Y_pred
    im_diff = axs[0, 2].imshow(np.flipud(diff[:, :]), cmap=cmo.cm.balance, extent=[lons[0], lons[-1], lats[0], lats[-1]], transform=proj)
    axs[0, 2].set_title("Difference")
    axs[0, 2].coastlines()
    axs[0, 2].add_feature(land, zorder=1)
    axs[0, 2].set_xlabel("Longitude")
    axs[0, 2].set_ylabel("Latitude")
    fig.colorbar(im_diff, ax=axs[0, 2], orientation='vertical', fraction=fraction, pad=pad)

    # Compute the gradient of the true SSH and the predicted SSH
    dx, dy = np.gradient(Y)
    grad_true = np.sqrt(dx**2 + dy**2)
    dx, dy = np.gradient(Y_pred)
    grad_pred = np.sqrt(dx**2 + dy**2)
    vmax = 0.05

    # Plot the gradient of the true SSH
    im_grad_true = axs[1, 0].imshow(np.flipud(grad_true[:, :]), cmap=cmo.cm.balance, 
                                    extent=[lons[0], lons[-1], lats[0], lats[-1]], transform=proj,
                                    vmin=0, vmax=vmax)
    axs[1, 0].set_title("Gradient of True SSH")
    axs[1, 0].coastlines()
    axs[1, 0].add_feature(land, zorder=1)
    axs[1, 0].set_xlabel("Longitude")
    axs[1, 0].set_ylabel("Latitude")
    fig.colorbar(im_grad_true, ax=axs[1, 0], orientation='vertical', fraction=fraction, pad=pad)

    # Plot the gradient of the predicted SSH
    im_grad_pred = axs[1, 1].imshow(np.flipud(grad_pred[:, :]), cmap=cmo.cm.balance, 
                                    extent=[lons[0], lons[-1], lats[0], lats[-1]], transform=proj,
                                    vmin=0, vmax=vmax)
    axs[1, 1].set_title("Gradient of Predicted SSH")
    axs[1, 1].coastlines()
    axs[1, 1].add_feature(land, zorder=1)
    axs[1, 1].set_xlabel("Longitude")
    axs[1, 1].set_ylabel("Latitude")
    fig.colorbar(im_grad_pred, ax=axs[1, 1], orientation='vertical', fraction=fraction, pad=pad)

    # Set the final figure to axis off
    axs[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(output_file.replace(".jpg", "_predictions.jpg"))
    plt.close()


def general_plot(fig, ax, data, lats, lons, title, vmin=None, vmax=None, 
                 proj=ccrs.PlateCarree(), fraction=0.05, pad=0.05,
                 cmap=cmo.cm.balance):
    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)

    land = cfeature.NaturalEarthFeature(
            category='physical',
        name='land',
        scale='10m',
        facecolor='gray'
    )

    im = ax.imshow(np.flipud(data), 
                   extent=[lons[0], lons[-1], lats[0], lats[-1]],
                   transform=proj,
                   cmap=cmap, 
                   vmin=vmin, vmax=vmax)

    ax.set_title(title)
    ax.coastlines()
    ax.add_feature(land, zorder=1)
    fig.colorbar(im, ax=ax, orientation='vertical', pad=pad, fraction=fraction)

    return im