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
    general_plot(fig, axs[0, 0], all_data['sst'][idx, :, :], lats, lons, 'SST', 
                  proj=proj, fraction=fraction, pad=pad, cmap=cmo.cm.thermal)

    # Plot Chlor-a
    general_plot(fig, axs[0, 1], all_data['chlora'][idx, :, :], lats, lons, 'Chl-a', 
                  proj=proj, fraction=fraction, pad=pad, cmap=cmo.cm.algae)

    # Plot SSH
    general_plot(fig, axs[0, 2], all_data['ssh'][idx, :, :], lats, lons, 'SSH', 
                  proj=proj, fraction=fraction, pad=pad, cmap=cmo.cm.balance)

    # Plot SSH Track
    sigma = 1
    temp_ssh_track = all_data['ssh_track'][idx, :, :].fillna(0)
    smooth_ssh_masked_data = gaussian_filter(temp_ssh_track, sigma=sigma)
    smooth_ssh_masked_data = np.where(smooth_ssh_masked_data != 0, smooth_ssh_masked_data, np.nan)
    general_plot(fig, axs[0, 3], smooth_ssh_masked_data, lats, lons, 'SSH Track', 
                  proj=proj, fraction=fraction, pad=pad, cmap=cmo.cm.deep)

    # Plot SWOT
    general_plot(fig, axs[0, 4], all_data['swot'][idx, :, :], lats, lons, 'SWOT', 
                  proj=proj, fraction=fraction, pad=pad, cmap=cmo.cm.delta)

    # ============================ Normalized Fields ============================
    # Plot SST Normalized
    general_plot(fig, axs[1, 0], all_data['sst_normalized'][idx, :, :], lats, lons, 'SST Normalized', 
                  proj=proj, fraction=fraction, pad=pad, cmap=cmo.cm.thermal)

    # Plot Chlor-a Normalized
    general_plot(fig, axs[1, 1], all_data['chlora_normalized'][idx, :, :], lats, lons, 'Chl-a Normalized', 
                  proj=proj, fraction=fraction, pad=pad, cmap=cmo.cm.algae)

    # Plot SSH Normalized
    general_plot(fig, axs[1, 2], all_data['ssh_normalized'][idx, :, :], lats, lons, 'SSH Normalized', 
                  proj=proj, fraction=fraction, pad=pad, cmap=cmo.cm.balance)
    
    # Plot SSH Track Normalized
    temp_ssh_track_normalized = all_data['ssh_track_normalized'][idx, :, :].fillna(0)
    smooth_ssh_track_normalized = gaussian_filter(temp_ssh_track_normalized, sigma=sigma)
    smooth_ssh_track_normalized = np.where(smooth_ssh_track_normalized != 0, smooth_ssh_track_normalized, np.nan)
    general_plot(fig, axs[1, 3], smooth_ssh_track_normalized, lats, lons, 'SSH Track Normalized', 
                  proj=proj, fraction=fraction, pad=pad, cmap=cmo.cm.deep)

    # Plot SWOT Normalized
    general_plot(fig, axs[1, 4], all_data['swot_normalized'][idx, :, :], lats, lons, 'SWOT Normalized', 
                  proj=proj, fraction=fraction, pad=pad, cmap=cmo.cm.delta)

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
    if dataset_type == "gradient":
        fig, axs = plt.subplots(len(input_names), days_before+2, figsize=(11*int(ceil((days_before+1)/2)), 4*len(input_names)), 
                                subplot_kw={'projection': proj})
    else:
        fig, axs = plt.subplots(len(input_names), days_before+1, figsize=(11*int(ceil((days_before+1)/2)), 4*len(input_names)), 
                                subplot_kw={'projection': proj})

    # Apply the mask to all the input fields except the last one. Transform to nan where the mask is 0 
    X_masked = np.copy(X)
    for i in range(X.shape[0]-1):
        X_masked[i, :, :] = np.where(X[-1, :, :] == 0, np.nan, X[i, :, :])

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
            
            general_plot(fig, axs[i, j], X_masked[cur_index, :, :], lats, lons, 
                          f"{title} at {days_before - j} days before", 
                          cmap=cmap, vmin=None, vmax=None, proj=proj, fraction=fraction, pad=pad)
           
    if dataset_type != "gradient":
        plot_col = -1
    else:
        plot_col = -2
    # Plot the mask in the last column, first row
    general_plot(fig, axs[0, plot_col], X[-1, :, :], lats, lons, "Mask", 
                  cmap=cmo.cm.solar, vmin=None, vmax=None, proj=proj, fraction=fraction, pad=pad)

    if dataset_type != "gradient":
        # Plot the true SSH in the last column of the last row
        general_plot(fig, axs[-1, -1], Y, lats, lons, "True SSH", 
                cmap=cmo.cm.balance, vmin=None, vmax=None, proj=proj, fraction=fraction, pad=pad)
    else:
        # Plot the true SSH in the last row of the second to last column
        general_plot(fig, axs[-1, -2], Y[0, :, :], lats, lons, "True SSH", 
                cmap=cmo.cm.balance, vmin=None, vmax=None, proj=proj, fraction=fraction, pad=pad)
        # Plot the true gradient of SSH in the second to last row of the second to last column
        vmax_gradient = 1
        vmin_gradient = -vmax_gradient
        general_plot(fig, axs[-2, -2], Y[1, :, :], lats, lons, "True Gradient of SSH", 
                cmap=cmo.cm.balance, vmin=vmin_gradient, vmax=vmax_gradient, proj=proj, fraction=fraction, pad=pad)

    if dataset_type == "extended": 
        # Plot the previous SSH + noise
        general_plot(fig, axs[1, plot_col], X[-2, :, :], lats, lons, "Previous SSH + noise", 
                      cmap=cmo.cm.balance, vmin=None, vmax=None, proj=proj, fraction=fraction, pad=pad)
       
        # Plot the two steps before SSH + noise
        general_plot(fig, axs[2, plot_col], X[-3, :, :], lats, lons, "Two steps before SSH + noise", 
                      cmap=cmo.cm.balance, vmin=None, vmax=None, proj=proj, fraction=fraction, pad=pad)
    
    elif dataset_type == "gradient":
        vmax_gradient = 1
        vmin_gradient = -vmax_gradient
        # Plot the previous ssh + noise
        general_plot(fig, axs[0, -1], X[-2, :, :], lats, lons, "Previous SSH + noise", 
                      cmap=cmo.cm.balance, vmin=None, vmax=None, proj=proj, fraction=fraction, pad=pad)
        # Plot the previous gradient of ssh + noise
        general_plot(fig, axs[1, -1], X[-3, :, :], lats, lons, "Previous Gradient of SSH + noise", 
                      cmap=cmo.cm.balance, vmin=vmin_gradient, vmax=vmax_gradient, proj=proj, fraction=fraction, pad=pad)
        # Plot the two steps before ssh + noise
        general_plot(fig, axs[2, -1], X[-4, :, :], lats, lons, "Two steps before SSH + noise", 
                      cmap=cmo.cm.balance, vmin=None, vmax=None, proj=proj, fraction=fraction, pad=pad)
        # Plot the two steps before gradient of ssh + noise
        general_plot(fig, axs[3, -1], X[-5, :, :], lats, lons, "Two steps before Gradient of SSH + noise", 
                      cmap=cmo.cm.balance, vmin=vmin_gradient, vmax=vmax_gradient, proj=proj, fraction=fraction, pad=pad)
       
    else:
        # Hide axes for empty subplots
        for i in range(1, len(input_names)-1):
            axs[i, plot_col].axis('off')

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_predictions(X, Y, Y_pred, output_file, lats, lons, dataset_type="regular"):
    input_names = ["sst", "chlora", "ssh_track", "swot"]
    days_before = floor(X.shape[0]/len(input_names))

    proj = ccrs.PlateCarree()

    if dataset_type == "gradient":  # One extra column for the gradient of the two inputs
        fig, axs = plt.subplots(len(input_names), days_before + 2, 
                                figsize=(12*int(ceil((days_before+1)/2)), 20), 
                                subplot_kw={'projection': proj})
    else:
        fig, axs = plt.subplots(len(input_names), days_before + 1, 
                                # figsize=(12*int(ceil((days_before+1)/2)), 20), 
                                figsize=(8*int(ceil((days_before+1)/2)), 12), 
                                subplot_kw={'projection': proj})
    
    # Apply the mask to all input fields except the last one
    X_masked = np.copy(X)
    for i in range(X.shape[0]-1):
        X_masked[i, :, :] = np.where(X[-1, :, :] == 0, np.nan, X[i, :, :])

    # Colorbar settings
    fraction = 0.038
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
            
            general_plot(fig, axs[i, j], X_masked[cur_index, :, :], lats, lons, 
                          f"{input_name} at {days_before - j - 1} days before", 
                          cmap=cmap, vmin=None, vmax=None, proj=proj, fraction=fraction, pad=pad)

    # Use the last column for mask, true SSH, predicted SSH, and difference
    # Plot the mask
    im_mask = axs[0, -1].imshow(np.flipud(X[-1, :, :]), extent=[lons[0], lons[-1], lats[0], lats[-1]], transform=proj)
    axs[0, -1].set_title("Mask")
    fig.colorbar(im_mask, ax=axs[0, -1], orientation='vertical', fraction=fraction, pad=pad)

    if dataset_type == "extended":
        # Plot the previous SSH + noise and two steps before SSH + noise
        general_plot(fig, axs[1, -1], X[-2, :, :], lats, lons, "Previous SSH + noise", 
                      cmap=cmo.cm.balance, vmin=None, vmax=None, proj=proj, fraction=fraction, pad=pad)

        general_plot(fig, axs[2, -1], X[-3, :, :], lats, lons, "Two steps before SSH + noise", 
                      cmap=cmo.cm.balance, vmin=None, vmax=None, proj=proj, fraction=fraction, pad=pad)

    else:
        # Hide axes for empty subplots
        for i in range(1, len(input_names)):
            axs[i, -1].axis('off')

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    # Make a separate plot for the true SSH, predicted SSH, and difference
    fig, axs = plt.subplots(2, 3, figsize=(20, 10), subplot_kw={'projection': proj})

    vmin = -0.6
    vmax = 0.6
    # Plot the true SSH
    if dataset_type != "gradient":
        general_plot(fig, axs[0, 0], Y, lats, lons, "True SSH", 
                    cmap=cmo.cm.balance, vmin=vmin, vmax=vmax, proj=proj, fraction=fraction, pad=pad)

    # Plot the predicted SSH
    general_plot(fig, axs[0, 1], Y_pred, lats, lons, "Predicted SSH", 
                  cmap=cmo.cm.balance, vmin=vmin, vmax=vmax, proj=proj, fraction=fraction, pad=pad)

    # Plot the difference
    vmin = -0.05
    vmax = 0.05
    diff = Y - Y_pred
    general_plot(fig, axs[0, 2], diff, lats, lons, "Difference", 
                  cmap=cmo.cm.balance, vmin=vmin, vmax=vmax, proj=proj, fraction=fraction, pad=pad)
    
    # Compute the gradient of the true SSH and the predicted SSH
    dx, dy = np.gradient(Y)
    grad_true = np.sqrt(dx**2 + dy**2)
    dx, dy = np.gradient(Y_pred)
    grad_pred = np.sqrt(dx**2 + dy**2)
    vmax = 0.05
    # Plot the gradient of the true SSH
    general_plot(fig, axs[1, 0], grad_true, lats, lons, "Gradient of True SSH", 
                  cmap=cmo.cm.balance, vmin=0, vmax=vmax, proj=proj, fraction=fraction, pad=pad)
   
    # Plot the gradient of the predicted SSH
    general_plot(fig, axs[1, 1], grad_pred, lats, lons, "Gradient of Predicted SSH", 
                  cmap=cmo.cm.balance, vmin=0, vmax=vmax, proj=proj, fraction=fraction, pad=pad)
   
    # Set the final figure to axis off
    axs[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(output_file.replace(".png", "_predictions.png"))
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