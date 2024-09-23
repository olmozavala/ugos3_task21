import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import cmocean as cmo
from scipy.ndimage import gaussian_filter
from math import ceil, floor


def plot_dataset_data(idx, all_data, lats, lons):
    print("Plotting some examples...")
    fig, axs = plt.subplots(2, 5, figsize=(20, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    # idx = np.random.randint(0, all_data['sst'].shape[0])

    # Set up the cartopy projection
    proj = ccrs.PlateCarree()

    # Plot the original fields with coastlines and colorbars
    im1 = axs[0, 0].imshow(np.flipud(all_data['sst'][idx, :, :]), extent=[lons[0], lons[-1], lats[0], lats[-1]], transform=proj, cmap=cmo.cm.thermal)
    axs[0, 0].coastlines()
    axs[0, 0].set_title('SST')
    fig.colorbar(im1, ax=axs[0, 0], orientation='vertical')

    # Plot Chlor-a
    im2 = axs[0, 1].imshow(np.flipud(all_data['chlora'][idx, :, :]), extent=[lons[0], lons[-1], lats[0], lats[-1]], transform=proj, cmap=cmo.cm.algae)
    axs[0, 1].coastlines()
    axs[0, 1].set_title('Chl-a')
    fig.colorbar(im2, ax=axs[0, 1], orientation='vertical')

    # Plot SSH
    im3 = axs[0, 2].imshow(np.flipud(all_data['ssh'][idx, :, :]), extent=[lons[0], lons[-1], lats[0], lats[-1]], transform=proj, cmap=cmo.cm.balance)
    axs[0, 2].coastlines()
    axs[0, 2].set_title('SSH')
    fig.colorbar(im3, ax=axs[0, 2], orientation='vertical')

    # Plot SSH Track
    sigma = 1
    temp_ssh_track = all_data['ssh_track'][idx, :, :].fillna(0)
    smooth_ssh_masked_data = gaussian_filter(temp_ssh_track, sigma=sigma)
    smooth_ssh_masked_data = np.where(smooth_ssh_masked_data != 0, smooth_ssh_masked_data, np.nan)
    img7 = axs[0, 3].imshow(np.flipud(smooth_ssh_masked_data), extent=[lons[0], lons[-1], lats[0], lats[-1]], transform=proj, cmap=cmo.cm.deep)
    axs[0, 3].coastlines()
    axs[0, 3].set_title('SSH Track')
    fig.colorbar(img7, ax=axs[0, 3], orientation='vertical')

    # Plot SWOT
    im8 = axs[0, 4].imshow(np.flipud(all_data['swot'][idx, :, :]), extent=[lons[0], lons[-1], lats[0], lats[-1]], transform=proj, cmap=cmo.cm.delta)
    axs[0, 4].coastlines()
    axs[0, 4].set_title('SWOT')
    fig.colorbar(im8, ax=axs[0, 4], orientation='vertical')

    # ============================ Normalized Fields ============================
    # Plot SST Normalized
    im4 = axs[1, 0].imshow(np.flipud(all_data['sst_normalized'][idx, :, :]), extent=[lons[0], lons[-1], lats[0], lats[-1]], transform=proj, cmap=cmo.cm.thermal)
    axs[1, 0].coastlines()
    axs[1, 0].set_title('SST Normalized')
    fig.colorbar(im4, ax=axs[1, 0], orientation='vertical')

    # Plot Chlor-a Normalized
    im5 = axs[1, 1].imshow(np.flipud(all_data['chlora_normalized'][idx, :, :]), extent=[lons[0], lons[-1], lats[0], lats[-1]], transform=proj, cmap=cmo.cm.algae)
    axs[1, 1].coastlines()
    axs[1, 1].set_title('Chl-a Normalized')
    fig.colorbar(im5, ax=axs[1, 1], orientation='vertical')

    # Plot SSH Normalized
    im6 = axs[1, 2].imshow(np.flipud(all_data['ssh_normalized'][idx, :, :]), extent=[lons[0], lons[-1], lats[0], lats[-1]], transform=proj, cmap=cmo.cm.balance)
    axs[1, 2].coastlines()
    axs[1, 2].set_title('SSH Normalized')
    fig.colorbar(im6, ax=axs[1, 2], orientation='vertical')

    # Plot SSH Track Normalized
    temp_ssh_track_normalized = all_data['ssh_track_normalized'][idx, :, :].fillna(0)
    smooth_ssh_track_normalized = gaussian_filter(temp_ssh_track_normalized, sigma=sigma)
    smooth_ssh_track_normalized = np.where(smooth_ssh_track_normalized != 0, smooth_ssh_track_normalized, np.nan)
    img8 = axs[1, 3].imshow(np.flipud(smooth_ssh_track_normalized), extent=[lons[0], lons[-1], lats[0], lats[-1]], transform=proj, cmap=cmo.cm.deep)
    axs[1, 3].coastlines()
    axs[1, 3].set_title('SSH Track Normalized')
    fig.colorbar(img8, ax=axs[1, 3], orientation='vertical')

    # Plot SWOT Normalized
    im9 = axs[1, 4].imshow(np.flipud(all_data['swot_normalized'][idx, :, :]), extent=[lons[0], lons[-1], lats[0], lats[-1]], transform=proj, cmap=cmo.cm.delta)
    axs[1, 4].coastlines()
    axs[1, 4].set_title('SWOT Normalized')
    fig.colorbar(im9, ax=axs[1, 4], orientation='vertical')

    # Adjust layout for better visualization
    plt.tight_layout()

    # Save the figure
    file_name = f"/unity/f1/ozavala/OUTPUTS/HR_SSH_from_Chlora/data_loaders/normalized_fields_with_coastlines_colorbars_{idx}.png"
    print(f"Saving figure to {file_name}")
    plt.savefig(file_name)
    plt.close()


def plot_single_batch_element(X, input_names, days_before, output_file):
    # It assumes the input names do not contain the mask.
    fig, axs = plt.subplots(days_before+1, len(input_names), figsize=(20, 10*int(ceil(days_before/2))))
    # Apply the mask to all the input fields except the last one. Transform to nan where the mask is 0 
    X_masked = np.copy(X)
    for i in range(X.shape[0]-1):
        X_masked[i, :, :] = np.where(X[-1, :, :] == 0, np.nan, X[i, :, :])

    for i in range(days_before):
        for j, input_name in enumerate(input_names):
            cur_index = i * len(input_names) + j
            # print(f"index: {cur_index}")
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
            
            axs[i, j].imshow(np.flipud(X_masked[cur_index, :, :]), cmap=cmap)
            axs[i, j].set_title(f"{input_name} at {i} days before")

    # Plot the mask in the last row, first column
    axs[-1, 0].imshow(np.flipud(X[-1, :, :]))
    axs[-1, 0].set_title("Mask")

    # Hide axes for empty subplots
    for i in range(1, len(input_names)):
        axs[-1, i].axis('off')

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_predictions(X, Y, Y_pred, output_file):
    input_names = ["sst", "chlora", "ssh_track", "swot"]
    days_before = floor(X.shape[0]/len(input_names))
    # It assumes the input names do not contain the mask.
    fig, axs = plt.subplots(days_before+1, len(input_names), figsize=(20, 10*int(ceil(days_before/2))))
    # Apply the mask to all the input fields except the last one. Transform to nan where the mask is 0 
    X_masked = np.copy(X)
    for i in range(X.shape[0]-1):
        X_masked[i, :, :] = np.where(X[-1, :, :] == 0, np.nan, X[i, :, :])

    for i in range(days_before):
        for j, input_name in enumerate(input_names):
            cur_index = i * len(input_names) + j
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
            
            im = axs[i, j].imshow(np.flipud(X_masked[cur_index, :, :]), cmap=cmap)
            axs[i, j].set_title(f"{input_name} at {i} days before")
            fig.colorbar(im, ax=axs[i, j], orientation='vertical', fraction=0.046, pad=0.04)

    # Plot the mask in the last row, first column
    im_mask = axs[-1, 0].imshow(np.flipud(X[-1, :, :]))
    axs[-1, 0].set_title("Mask")
    fig.colorbar(im_mask, ax=axs[-1, 0], orientation='vertical', fraction=0.046, pad=0.04)

    # Plot the true SSH
    im_true = axs[-1, 1].imshow(np.flipud(Y[:, :]), cmap=cmo.cm.balance)
    axs[-1, 1].set_title("True SSH")
    fig.colorbar(im_true, ax=axs[-1, 1], orientation='vertical', fraction=0.046, pad=0.04)

    # Plot the predicted SSH
    im_pred = axs[-1, 2].imshow(np.flipud(Y_pred[:, :]), cmap=cmo.cm.balance)
    axs[-1, 2].set_title("Predicted SSH")
    fig.colorbar(im_pred, ax=axs[-1, 2], orientation='vertical', fraction=0.046, pad=0.04)

    # Plot the difference
    diff = Y - Y_pred
    im_diff = axs[-1, 3].imshow(np.flipud(diff[:, :]), cmap=cmo.cm.balance)
    axs[-1, 3].set_title("Difference")
    fig.colorbar(im_diff, ax=axs[-1, 3], orientation='vertical', fraction=0.046, pad=0.04)

    # Hide axes for empty subplots
    for i in range(4, len(input_names)):
        axs[-1, i].axis('off')

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

