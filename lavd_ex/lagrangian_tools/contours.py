import numba as nb
import numpy as np
from typing import Tuple
from scipy.interpolate import interp1d
from skimage.feature import peak_local_max
from skimage.measure import find_contours
from scipy.spatial import ConvexHull, convex_hull_plot_2d


@nb.njit
def polygon_area(xy: np.array):
    """
    Area of a polygon defined by the points xy

    Args:
        xy [N,2]: coordinates defining the polygon

    Returns:
        [floats]: area
    """
    x, y = xy[:, 0], xy[:, 1]
    correction = x[-1] * y[0] - y[-1] * x[0]
    main_area = np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])
    return 0.5 * np.abs(main_area + correction)


def peak_in_hull(p, hull, tol=1e-12):
    """
    Test in a point p are in the convex hull (scipy.spatial.ConvexHull)
    """
    hq = hull.equations
    return np.all(hq[:, :-1] @ p + hq[:, -1] <= tol)


# not used now but I want to rethink the algorithm
# and reorder some calculations for speed up
def peaks_in_hull(p, hull, tol=1e-12):
    """
    Test in points p are in the convex hull (scipy.spatial.ConvexHull)
    """
    hq = hull.equations
    return np.all(
        hq[:, :-1] @ p.T + np.repeat(hq[:, -1][None, :], len(p), axis=0).T <= tol, 0
    )


def extract_contours(
    lon: np.array,
    lat: np.array,
    lavd: np.array,
    defTol: float = 0.075,
    number_levels: int = 50,
) -> Tuple[np.array, np.array]:
    """

    Args:
        lon: meridional coordinates of the grid in degrees [nx, ny]
        lat: zonal coordinates of the grid in degrees [nx, ny]
        lavd: Lagrangian averaged vorticity deviation [nx, ny]
        defTol: control the deficiency of the loop closer to 0 means perfectly convex (~circular)
        number_levels: contour levels between [0, peak_lavd_value]

    Returns:
        peaks_xy: centers of extracted vortices [N, 2]
        contours: contours of extracted vortices [N, 2]

    """

    # peaks and data structures
    peaks_xy = peak_local_max(lavd, min_distance=20)
    peaks_value = lavd[peaks_xy[:, 0], peaks_xy[:, 1]]
    contours = np.empty_like(peaks_value, dtype="object")

    # coordinates are converted to indices by `find_contour`
    # this is used to bring back to degrees
    flon = interp1d(np.arange(0, len(lon)), lon)
    flat = interp1d(np.arange(0, len(lat)), lat)

    n = 0  # only for counting
    for j in range(0, len(peaks_xy)):
        print(
            f"{j+1}/{len(peaks_xy)} (Found {n} {'eddies' if n>1 else 'eddy'})", end="\r"
        )
        pxy = peaks_xy[j]
        c_levels = np.linspace(0, peaks_value[j], number_levels)[1:]

        # loop from the largest lowest value to the peaks
        # if we found something we stop because we want to largest
        # contour respecting the criteria
        for c_level in c_levels:
            if contours[j] is None:
                # TODO: calculate only on subregion for better perf                
                # this changes the unit from (lon,lat) to (i,j) indices
                cs = find_contours(lavd, c_level)  
                for c in cs:
                    try:  # prevent error when ConvexHull fails on weirdly shaped contour
                        hull = ConvexHull(c)
                        if peak_in_hull(pxy, hull):
                            areaPoly = polygon_area(c)
                            if (
                                abs(areaPoly - hull.volume) / areaPoly * 100 < defTol
                            ):  # in 2D hull.volume returns area (!)
                                contours[j] = np.column_stack((flon(c[:, 0]), flat(c[:, 1])))
                                n += 1
                                break
                    except:
                        pass

    return peaks_xy, contours
