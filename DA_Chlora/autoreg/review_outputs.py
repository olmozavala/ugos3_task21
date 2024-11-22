# %% Loading outputs
import numpy as np
import matplotlib.pyplot as plt
import cmocean as cmo
from os.path import join

root_dir = "/unity/g2/jvelasco/ai_outs/debug"

hanning0 = 'output_hanning_50_0.npy'
hanning1 = 'output_hanning_50_1.npy'
randn0 = 'output_randn_0.npy'
randn1 = 'output_randn_1.npy'

h0 = np.load(join(root_dir, hanning0)).squeeze()
h1 = np.load(join(root_dir, hanning1)).squeeze()
r0 = np.load(join(root_dir, randn0)).squeeze()
r1 = np.load(join(root_dir, randn1)).squeeze()

print(f"h0 shape: {h0.shape}")
print(f"h1 shape: {h1.shape}")
print(f"r0 shape: {r0.shape}")
print(f"r1 shape: {r1.shape}")


# %% Compute graddients

gdx, gdy = np.gradient(h0)
h0_gradient = np.sqrt(gdx**2 + gdy**2)

gdx, gdy = np.gradient(h1)
h1_gradient = np.sqrt(gdx**2 + gdy**2)

gdx, gdy = np.gradient(r0)
r0_gradient = np.sqrt(gdx**2 + gdy**2)

gdx, gdy = np.gradient(r1)
r1_gradient = np.sqrt(gdx**2 + gdy**2)

# %% Obtaina metrics

# Means
h0_mean = np.mean(h0)
h1_mean = np.mean(h1)
print(f"Mean of h0: {h0_mean}")
print(f"Mean of h1: {h1_mean}")
r0_mean = np.mean(r0)
r1_mean = np.mean(r1)
print(f"Mean of r0: {r0_mean}")
print(f"Mean of r1: {r1_mean}") 

# Max and min
h0_max = np.nanmax(h0)
h0_min = np.nanmin(h0)
h1_max = np.nanmax(h1)
h1_min = np.nanmin(h1)
r0_max = np.nanmax(r0)
r0_min = np.nanmin(r0)
r1_max = np.nanmax(r1)
r1_min = np.nanmin(r1)
print(f"Max of h0: {h0_max}, Min of h0: {h0_min}")
print(f"Max of h1: {h1_max}, Min of h1: {h1_min}")
print(f"Max of r0: {r0_max}, Min of r0: {r0_min}")
print(f"Max of r1: {r1_max}, Min of r1: {r1_min}")

# STD
h0_std = np.std(h0)
h1_std = np.std(h1)
r0_std = np.std(r0)
r1_std = np.std(r1)
print(f"STD of h0: {h0_std}")
print(f"STD of h1: {h1_std}")
print(f"STD of r0: {r0_std}")
print(f"STD of r1: {r1_std}")

# Check for nans
print(f"Nans in h0: {np.isnan(h0).sum()}")
print(f"Nans in h1: {np.isnan(h1).sum()}")
print(f"Nans in r0: {np.isnan(r0).sum()}")
print(f"Nans in r1: {np.isnan(r1).sum()}")

# Check for values close to 0
z0 = 1e-6
print(f"Values close to 0 in h0: {(h0 < z0).sum()}")
print(f"Values close to 0 in h1: {(h1 < z0).sum()}")
print(f"Values close to 0 in r0: {(r0 < z0).sum()}")
print(f"Values close to 0 in r1: {(r1 < z0).sum()}")

# %% Plotting outputs
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].imshow(h0[:, :], cmap=cmo.cm.haline)
axs[0, 1].imshow(h0_gradient[:, :], cmap=cmo.cm.balance)
axs[1, 0].imshow(h1[:, :], cmap=cmo.cm.haline)
axs[1, 1].imshow(h1_gradient[:, :], cmap=cmo.cm.balance)
axs[0, 0].set_title("Output 0")
axs[0, 1].set_title("Gradient 0")
axs[1, 0].set_title("Output 1")
axs[1, 1].set_title("Gradient 1")
fig.suptitle("Hanning 50")
plt.show()
# %% 
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].imshow(r0[:, :], cmap=cmo.cm.haline)
axs[0, 1].imshow(r0_gradient[:, :], cmap=cmo.cm.balance)
axs[1, 0].imshow(r1[:, :], cmap=cmo.cm.haline)
axs[1, 1].imshow(r1_gradient[:, :], cmap=cmo.cm.balance)
axs[0, 0].set_title("Output 0")
axs[0, 1].set_title("Gradient 0")
axs[1, 0].set_title("Output 1")
axs[1, 1].set_title("Gradient 1")
fig.suptitle("Randn")
plt.show()
# %%
