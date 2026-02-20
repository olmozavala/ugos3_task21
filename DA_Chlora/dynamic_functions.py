import numpy as np
import matplotlib.pyplot as plt
from os.path import join


R = 6371000.0  # Earth radius in meters

def remove_planar_trend(field):
    """
    Removes a 2D plane (z = ax + by + c) from the data 
    to prevent spectral leakage from large-scale gradients.
    """
    ny, nx = field.shape
    Y, X = np.indices((ny, nx))
    # Flatten for regression
    X_flat = X.ravel()
    Y_flat = Y.ravel()
    Z_flat = field.ravel()
    
    # Fit plane: Z = C0 + C1*X + C2*Y
    # A matrix: [1, x, y]
    A = np.c_[np.ones_like(X_flat), X_flat, Y_flat]
    C, _, _, _ = np.linalg.lstsq(A, Z_flat, rcond=None)
    
    # Evaluate plane
    Z_plane = C[0] + C[1]*X + C[2]*Y
    return field - Z_plane


def compute_psd(
    ssh: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    y_slice: slice = slice(350, 550),
    x_slice: slice = slice(150, 560),
):
    """
    Compute the 1D, radially averaged power spectral density (PSD)
    of a sea surface height (SSH) field.

    This function is a functional version of the logic implemented in `psd.py`.

    **Assumptions**
    - `ssh` is either a 2D array (ny, nx) or a 3D array (..., ny, nx).
    - `lat` is a 1D array of latitude values corresponding to the y-dimension.
    - `lon` is a 1D array of longitude values corresponding to the x-dimension.
    - `ssh` is already expressed in physical units (e.g. meters).

    Parameters
    ----------
    ssh : np.ndarray
        SSH field. Shape (..., ny, nx). Only the last two dimensions are used.
    lat : np.ndarray
        1D latitude array of length ny.
    lon : np.ndarray
        1D longitude array of length nx.
    y_slice : slice, optional
        Slice to apply on the latitude / y-dimension, by default slice(350, 550).
    x_slice : slice, optional
        Slice to apply on the longitude / x-dimension, by default slice(150, 560).

    Returns
    -------
    psd_1d : np.ndarray
        1D radially integrated PSD values.
    bin_centers : np.ndarray
        Centers of the wavenumber bins corresponding to `psd_1d`.
    """

    ssh = np.asarray(ssh)
    lat = np.asarray(lat)
    lon = np.asarray(lon)

    # Apply spatial slices
    ssh = ssh[..., y_slice, x_slice]
    lat_slice = lat[y_slice]
    lon_slice = lon[x_slice]

    # ssh must be strictly 2D after slicing
    if ssh.ndim != 2:
        raise ValueError(
            f"`ssh` must be a 2D array after slicing, but has shape {ssh.shape}"
        )
    ssh_2d = ssh

    ny, nx = ssh_2d.shape

    # Latitude / longitude in radians
    lat0 = np.deg2rad(np.mean(lat_slice))  # Average latitude of the patch
    lat_rad = np.deg2rad(lat_slice)
    lon_rad = np.deg2rad(lon_slice)

    dlat = np.gradient(lat_rad)  # [rad]
    dlon = np.gradient(lon_rad)  # [rad]

    # Grid spacing in km
    dy = R * dlat  # [m], 1D (ny,)
    dx = R * np.cos(lat0) * dlon  # [m], 1D (nx,) using lat0 for small patch
    dx = np.nanmean(dx) / 1000.0  # [km]
    dy = np.nanmean(dy) / 1000.0  # [km]

    # 1. Preprocess: Apply Hanning filter
    wx = np.hanning(nx)
    wy = np.hanning(ny)

    W2 = np.outer(wy, wx)
    PCF_hann = 1.0 / np.mean(W2**2)  # Power Correction Factor for Hanning
    
    detrended_ssh = remove_planar_trend(ssh_2d) # detrended SSH
    windowed_ssh = detrended_ssh * W2 # windowed detrended SSH

    # 2. Compute the 2D FFT and SHIFT it
    # Shift moves the (0,0) frequency to the center of the array
    fft_ssh = np.fft.fftshift(np.fft.fft2(windowed_ssh))

    # 3. Compute the PSD (in 2D)
    # Normalize by N^2 and account for window energy loss
    psd_2d = (dx * dy) * (np.abs(fft_ssh) ** 2) / (nx * ny)
    psd_2d *= PCF_hann  # Power Correction Factor for Hanning

    # 4. Create the Wavenumber Grid
    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
    KX, KY = np.meshgrid(kx, ky)
    K_mag = np.sqrt(KX**2 + KY**2)

    dkx = 1.0 / (nx * dx)   # cycles/km
    dky = 1.0 / (ny * dy)   # cycles/km
    dkA = dkx * dky         # (cycles/km)^2

    # 5. Radial Integration
    k_bins = np.linspace(K_mag.min(), K_mag.max(), num=100)
    bin_centers = 0.5 * (k_bins[1:] + k_bins[:-1])

    psd_1d = []
    for i in range(len(k_bins) - 1):
        mask = (K_mag >= k_bins[i]) & (K_mag < k_bins[i + 1])
        if np.any(mask):
            # Total power in the current ring
            total_bin_power = np.sum(psd_2d[mask]) * dkA
            psd_val = total_bin_power / (k_bins[i + 1] - k_bins[i])
            psd_1d.append(psd_val)
        else:
            psd_1d.append(np.nan)

    psd_1d = np.array(psd_1d)

    return psd_1d, bin_centers


def plot_psd(
    spectra,
    path: str = "",
    filename: str = "psd.png",
    title: str = "Power Spectral Density",
    labels=None,
    add_reference: bool = False,
    anchor_idx: int = 10,
    ref_exponents = (-3, -5),
    ref_colors = ("k--", "r--"),
    linewidth: float = 2
):
    """
    Plot one or more PSD spectra on a log-log plot and save the figure.

    Parameters
    ----------
    spectra : sequence of tuple(np.ndarray, np.ndarray) or np.ndarray
        Either:
        - Iterable of (psd_1d, bin_centers) pairs, e.g.
          [(psd_1d_1, bin_centers_1), (psd_1d_2, bin_centers_2), ...]
        - OR a single psd_1d array (convenience format: if the second positional
          argument is also an array, it will be interpreted as bin_centers)
    path : str, optional
        Directory path where the figure will be written. Default is "".
        Note: In convenience format plot_psd(psd_1d, bin_centers, ...), the
        second argument is bin_centers, not path. Use keyword arguments for
        path and other options in that case.
    filename : str, optional
        Name of the output image file (e.g. "psd.png"). Default is "psd.png".
    title : str, optional
        Plot title. Default is "Power Spectral Density".
    labels : sequence of str, optional
        Legend labels for each spectrum. If not provided or length
        does not match the number of spectra, generic labels
        "spectrum_1", "spectrum_2", ... are used.
    add_reference : bool, optional
        If True, add reference power-law spectra (e.g. k^-3, k^-5)
        using the first spectrum as anchor. Default is False.
    anchor_idx : int, optional
        Index in the first spectrum from which to anchor the
        reference lines. Default is 10.
    ref_exponents : sequence of float, optional
        Exponents for the reference slopes (default: (-3, -5)).
    ref_colors : sequence of str, optional
        Matplotlib line styles/colors for each reference slope
        (default: ("k--", "r--")).
    """

    # Detect convenience call: if spectra is array-like and path is also array-like,
    # treat as plot_psd(psd_1d, bin_centers, ...)
    if hasattr(spectra, 'ndim') or (isinstance(spectra, np.ndarray)):
        if hasattr(path, 'ndim') or isinstance(path, np.ndarray):
            # Convenience format: plot_psd(psd_1d, bin_centers, ...)
            bin_centers = path
            spectra = [(spectra, bin_centers)]
            path = ""  # Reset path to default
    
    # Convert to list and handle single tuple case
    if isinstance(spectra, tuple) and len(spectra) == 2:
        # Single tuple: (psd_1d, bin_centers)
        spectra = [spectra]
    else:
        spectra = list(spectra)
    
    n_spec = len(spectra)

    if labels is None or len(labels) != n_spec:
        labels = [f"spectrum_{i+1}" for i in range(n_spec)]

    plt.figure(figsize=(8, 6))

    for (psd_1d, bin_centers), label in zip(spectra, labels):
        psd_1d = np.asarray(psd_1d)
        bin_centers = np.asarray(bin_centers)
        plt.loglog(bin_centers, psd_1d, label=label, linewidth=linewidth)

    # Optional reference spectra (e.g. k^-3, k^-5) based on first spectrum
    if add_reference and n_spec > 0:
        psd_ref, k_ref_centers = spectra[0]
        psd_ref = np.asarray(psd_ref)
        k_ref_centers = np.asarray(k_ref_centers)

        if 0 <= anchor_idx < len(k_ref_centers):
            k_ref = k_ref_centers[anchor_idx:]
            base_val = psd_ref[anchor_idx]

            for exp, style in zip(ref_exponents, ref_colors):
                ref_psd = base_val * (k_ref / k_ref[0]) ** exp
                label = f"$k^{{{exp}}}$ Slope"
                plt.loglog(k_ref, ref_psd, style, label=label, linewidth=linewidth)

    plt.xlabel("Wavenumber [Km$^{-1}$]")
    plt.ylabel("PSD")
    plt.title(title)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()

    out_path = join(path, filename)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    return out_path


