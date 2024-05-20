# Some functions used to extract the data along given pathways
# -------------------
# Mathilde Jutras
# contact: mjutras@hawaii.edu
# Nov 2023
# -------------------

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import gsw
from scipy.interpolate import interp1d
import shapely.geometry as geom
import math
from scipy.spatial import distance
from matplotlib.colors import LinearSegmentedColormap
import seawater as sw
from scipy.signal import savgol_filter

def pad(arr):

    # Desired shape
    desired_shape = (arr.shape[0], 4500)

    # Calculate the amount of padding needed
    pad_width = ((0, 0), (0, desired_shape[1] - arr.shape[1]))

    # Pad the array with NaN values
    padded_arr = np.pad(arr, pad_width, mode='constant', constant_values=np.nan)

    return padded_arr


def closest_node(node, nodes):
    closest_index = distance.cdist([node], nodes).argmin()
    return closest_index


def closest(alist, value):
    diff = [np.abs(alist-each) for each in value]
    idx = [each.argmin() for each in diff]
    return idx


def dist_latlon(p1, p2) :
    # get the distance in km between two geographical points
    R = 6373.0

    lat1 = math.radians(p1[0]) ; lat2 = math.radians(p2[0])
    lon1 = math.radians(p1[1]) ; lon2 = math.radians(p2[1])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (math.sin(dlat/2))**2 + math.cos(lat1) * math.cos(lat2) * (math.sin(dlon/2))**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    dist = R * c

    return dist


def neutral_density(sal, temp, press, lon, lat):

    SA = gsw.SA_from_SP(sal, press, lon, lat)
    CT = gsw.CT_from_t( SA, temp, press )

    return gsw.rho(SA, CT, press)


def distPathway(x, y, pathway_x, pathway_y) :
# Check what distance along the pathway a datapoint is

# points :    points to check if along the line
# pathway_x : longitude of pathway points
# pathway_y : latitude of pathway points

    # Points from which the 'line' along the pathway is defined
    xl = np.arange(pathway_x[0], pathway_x[-1], 0.1)
    f = interp1d(pathway_x, pathway_y, kind='cubic')
    yl = f(xl)

    theta = np.linspace(0, 2*np.pi, num=len(pathway_x), endpoint=False)
    interp_x = interp1d(theta, pathway_x, kind='cubic', bounds_error=False)
    interp_y = interp1d(theta, pathway_y, kind='cubic', bounds_error=False)

    theta_interp = np.linspace(0, 2*np.pi, num=100, endpoint=False)
    xl = interp_x(theta_interp)
    yl = interp_y(theta_interp)
    # remove nans at the end
    xl = [each for each in xl if np.isnan(each)==False]
    yl = [each for each in yl if np.isnan(each)==False]
    if len(xl) != len(yl):
        print('PROBLEM WHEN REMOVING NANS IN INTERPOLATION.')

    # Get an array of the distance along the transect line
    dist_vec = [0] ; total = 0
    for i in range(len(xl)-1) :
        dumdist = dist_latlon( [xl[i],yl[i]], [xl[i+1],yl[i+1]] )
        total = total + dumdist
        dist_vec.append( total )

    line = []
    for i in range(len(xl)):
        line.append([xl[i],yl[i]])

    # Calculate the distance along the pathway
    dist = []
    for i in range(len(x)) :

        if np.isnan(x[i]) == False and np.isnan(y[i]) == False: # if no nan in lat/lon

            lineg = geom.LineString(line)
            point = geom.Point(x[i],y[i])
            point_on_line = lineg.interpolate(lineg.project(point))

            dist.append( dist_vec[ closest_node( [point_on_line.x, point_on_line.y], line ) ] )

    return np.array(dist)


def mld_dbm_v3(temp_cast, salin_cast, press_cast, sig_theta_threshold):
    # calculate mixed layer depth

    # Sort cast by increasing depth
    depth_cast = sw.dpth(press_cast, -40)
    dep_sort = np.sort(depth_cast)
    ix = np.argsort(depth_cast)
    temp = temp_cast[ix]
    salin = salin_cast[ix]

    # Compute potential temperature and potential density
    theta = sw.ptmp(salin, temp, dep_sort, 0)
    sig_theta = sw.dens0(salin, theta)

    # Find reference depth
    ref_ind = np.argmax(dep_sort > 9)

    # Reference depth is too deep - no shallow depths in cast
    if dep_sort[ref_ind] > 25:
        return np.nan

    # Choose the reference sigma depending on the threshold chosen
    ref_sig_theta = sw.dens0(salin[ref_ind], theta[ref_ind]) + sig_theta_threshold

    # Search for MLD
    if np.sum(~np.isnan(sig_theta)) > 1:  # Not a one-point or all-NaN cast
        # Find mixed layer depth
        not_found = True
        start_ind = ref_ind
        iter_count = 1
        while not_found:
            # Begin search at reference (10 m) index
            # Find next point below reference depth that exceeds criterion
            ml_ind = np.argmax(sig_theta[start_ind:] > ref_sig_theta)
            if len(sig_theta) >= ml_ind + start_ind:  # ml_ind is within the interior of the cast
                if sig_theta[ml_ind + start_ind] > ref_sig_theta:  # Next point also meets criterion, therefore likely not a spike
                    not_found = False
                    ml_ind += start_ind - 1  # Final index
            else:  # Last point in cast
                not_found = False
                ml_ind += start_ind - 1
            # If a spike, start search again at first point after spike
            start_ind = ml_ind + start_ind
            iter_count += 1
            # Break loop if cast is all spikes/no MLD found
            if iter_count > len(sig_theta):
                break
        # If an MLD is found, interpolate to find depth at which density = ref_sig_theta
        # added that not first depth
        if not not_found:
            if not np.any(np.isnan(sig_theta[ml_ind - 1:ml_ind])) and ml_ind > 0:
                mld_out = np.interp(ref_sig_theta, sig_theta[ml_ind - 1:ml_ind], dep_sort[ml_ind - 1:ml_ind])
            else:
                mld_out = np.nan
        else:
            mld_out = np.nan

    else:
        mld_out = np.nan

    return mld_out
