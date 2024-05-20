# Generate masks based on PV, volume and streamlines, that isolate the water masses we are interested in

# ----------------
# Mathilde Jutras
# contact: mjutras@hawaii.edu
# Nov 2023
# ----------------

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd
import csv
from scipy.ndimage import binary_dilation, binary_erosion
from os import listdir
import matplotlib.ticker as mticker
from csv import DictWriter

# ---- Set-up ----
# Choose the water mass
name = 'SASTMW' 


# Load the PV/Volume/streamlines data
path = ### REPLACE WITH PATH TO PV_GLOBAL.NC
ds = xr.open_dataset(path+'PV_global.nc')

# With the global dataset, we don't use PV as a criteria
# Use use the geographical and density layer criteria we identified from looking at field maps
latrange = {'NASTMW':[26,45], 'NASPMW':[40,55], 'NPCMW':[15 ,50], 'SASTMW':[-45,-15], 'NPSTMW':[15,45], 'SPSTMW':[-42,-13], 'IOSTMW':[-40,-27], 'PacAAIW':[-60,-20]}
lonrange = {'NASTMW':[360-80,360], 'NASPMW':[360-60,360-10], 'NPCMW':[150,360-120], 'SASTMW':[360-40,20], 'NPSTMW':[110,360-180], 'SPSTMW':[110,360-100], 'IOSTMW':[20,60], 'PacAAIW':[150,360-70]}
pvlim = {'NASPMW': 2.5e-10, 'NASTMW':2.5e-10, 'NPCMW':2e-10, 'SASTMW':3e-10, 'NPSTMW':8e-10, 'SPSTMW':3e-10, 'IOSTMW':3e-10, 'PacAAIW':3e-10}
pvmax = {'NASPMW': 0.15e-8, 'NASTMW':5e-10, 'NPCMW':3e-10, 'SASTMW':3e-10, 'NPSTMW':4e-10, 'SPSTMW':2e-10, 'IOSTMW':2e-10, 'PacAAIW':2e-10}
denslevplot = {'NASPMW': 27, 'NASTMW':26.7, 'NPCMW':26.1, 'SASTMW':26.4, 'NPSTMW':25, 'SPSTMW':27, 'IOSTMW':27, 'PacAAIW':27.1}

if ('NP' in name) or ('SP' in name) or ('Pac' in name):
    cl = 180
else:
    cl = 0

# geographical mask
if lonrange[name][0] < lonrange[name][1]:
    dsl = ds.sel(lon=slice(lonrange[name][0], lonrange[name][1]), lat=slice(latrange[name][0], latrange[name][1]))
else:
    ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))
    ds = ds.roll(lon=int(len(ds['lon']) / 2), roll_coords=True)
    dsl = ds.sel(lon=slice(lonrange[name][0]-360, lonrange[name][1]), lat=slice(latrange[name][0], latrange[name][1]))
    lonrange[name] = [ lonrange[name][0]-360, lonrange[name][1] ]

# First, identify the density layers we will want to use
fig = plt.figure(figsize=(9,9))
axs = []
axs.append( fig.add_subplot(2,2,1) )
axs.append( fig.add_subplot(2,2,2, sharey=axs[0]) )
# zonal average
zmean = dsl.PV.mean(axis=2)
zvol  = dsl.Volume.sum(axis=2)
im = axs[0].contourf( dsl.lat, dsl.density, zmean, levels=np.linspace(0,pvmax[name],25), extend='max')
axs[0].contour( dsl.lat, dsl.density, zvol, 20, cmap='Greys_r', linewidths=1)
axs[0].set_title('Zonal mean')
# meridional average
mmean = dsl.PV.mean(axis=1)
mvol  = dsl.Volume.sum(axis=1)

axs[1].contourf( dsl.lon, dsl.density, mmean, levels=np.linspace(0,pvmax[name],25), extend='max' )
axs[1].contour( dsl.lon, dsl.density, mvol, 20, cmap='Greys_r', linewidths=1)
axs[1].set_title('Meridional mean')
axs[0].set_ylabel('Density')
axs[1].set_xlabel('Longitude') ; axs[0].set_xlabel('Latitude')
axs[0].invert_yaxis()
if 'NP' in name:
    axs[0].set_ylim([26.5, 24.5])
else:
    axs[0].set_ylim([27.7, 26.])

plt.subplots_adjust(top=0.9, bottom=0.15, left=0.1, right=0.85, wspace=0.1)
cbar_ax = fig.add_axes([0.89, 0.6, 0.02, 0.3])
cb = fig.colorbar(im, cax=cbar_ax, extend='max')
cb.set_label(label = 'PV [(ms)$^{-1}$]', weight='bold', fontsize=14)

ax = fig.add_subplot(2,2,3, projection=ccrs.Robinson(central_longitude=cl))
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1)
gl.ylocator = mticker.FixedLocator(np.arange(-60,60,20))
gl.xlabels_top = False ; gl.ylabels_left = True
#ax.set_extent([lonrange[name][0], lonrange[name][1], latrange[name][0], latrange[name][1]])
ax.coastlines()
im = ax.pcolor(dsl.lon, dsl.lat, dsl.PV.sel(density=denslevplot[name]), transform=ccrs.PlateCarree(), vmin=0, vmax=pvlim[name])
plt.colorbar(im, ax=ax, label='PV', shrink=0.5, orientation='horizontal')
im2 = ax.contour(dsl.lon, dsl.lat, dsl.Streamfunction.sel(density=denslevplot[name]), 10, transform=ccrs.PlateCarree(), cmap='magma', linewidths=0.5)
ax.set_title(denslevplot[name])

ax = fig.add_subplot(2,2,4, projection=ccrs.Robinson(central_longitude=cl))
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1)
gl.ylocator = mticker.FixedLocator(np.arange(-60,60,20))
gl.xlabels_top = False ; gl.ylabels_left = True
ax.coastlines()
im = ax.pcolor(dsl.lon, dsl.lat, dsl.Volume.sel(density=denslevplot[name]), transform=ccrs.PlateCarree(), cmap='Purples', vmin=0, vmax=1.5e12)
plt.colorbar(im, ax=ax, label='Volume', shrink=0.5, orientation='horizontal')
im2 = ax.contour(dsl.lon, dsl.lat, dsl.Streamfunction.sel(density=denslevplot[name]), 10, transform=ccrs.PlateCarree(), cmap='magma', linewidths=0.5)

plt.suptitle(name)
plt.savefig('Figures/PV_vol_global_%s.png'%name, dpi=300)
plt.show()

# --- Plot volume of low PV waters with density
crit = (dsl.PV < pvlim[name])
totvol = dsl.Volume.where(crit).sum(axis=(1,2))
plt.figure(figsize=(5,4))
plt.plot(dsl.density, totvol, '.-')
if 'NP' in name:
    plt.xlim([24.5, 28])
else:
    plt.xlim([25.5,28])
plt.xlabel('$\gamma_n$') ; plt.ylabel('Volume of low PV waters')
plt.title('%s, PV < %.1e'%(name,pvlim[name]))
plt.savefig('Figures/Vol_lowPV_%s.png'%name)
plt.show()
plt.close()

fieldnames = ['Name', 'Layer', 'Streamline limits', 'PV limit']

layers = input('Which density layers to use? ')
if ',' in layers:
    layers = layers.split(',')
    layers = [float(each) for each in layers]
elif layers == '':
    layers = []
else:
    layers = [float(layers)]
# ---

# Identify the streamlines we want to use
for layer in layers:

    ilayer = np.where(np.round(ds['density'],1) == layer)[0][0]
    # geographical mask
    mask0 = ((ds.lon < lonrange[name][1]) & (ds.lon > lonrange[name][0]) & (ds.lat > latrange[name][0]) & (ds.lat < latrange[name][1]))
    # For some water masses, remove low PV waters
    ##mask = mask & (ds.PV[ilayer,:,:] < pvlim[name])

    adjust = 'y'
    while adjust == 'y':

        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree(central_longitude=cl))
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1)
        gl.xlabels_top = False ; gl.ylabels_left = True ; gl.ylabels_right = False
        ax.coastlines()
        #im = ax.pcolor(dsl.lon, dsl.lat, dsl.Volume[ilayer,:,:], transform=ccrs.PlateCarree(), vmin=0, vmax=1.7e12, cmap='Blues')
        im = ax.pcolor(dsl.lon, dsl.lat, dsl.PV[ilayer,:,:], transform=ccrs.PlateCarree(), cmap='Purples', vmin=0, vmax=pvmax[name])
        levs = np.arange(round(np.nanmin(dsl.Streamfunction[ilayer,:,:]),0)+1, round(np.nanmax(dsl.Streamfunction[ilayer,:,:]),0), 0.5)
        im2 = ax.contour(dsl.lon, dsl.lat, dsl.Streamfunction[ilayer,:,:], levs, transform=ccrs.PlateCarree(), cmap='magma')
        plt.colorbar(im2, shrink=0.7, label='Streamlines', ticks=range(int(np.nanmin(dsl.Streamfunction[ilayer,:,:]))+1, int(np.nanmax(dsl.Streamfunction[ilayer,:,:])),1))
        plt.colorbar(im, shrink=0.7, orientation='horizontal', label='PV')#'Volume [m³]')
        ax.set_title(layer)
        plt.show()

        lims = input('Which streamlines to use as limits? ')
        if ',' in lims:
            lims = lims.split(',')
            lims = [float(each) for each in lims]
        elif lims == '':
            break
        else:
            lims = [float(lims)]
        ##vlim = input('Volume lower limit: ')
        vlim = input('PV upper limit: ')
        vlim = float(vlim)

        if len(lims) == 1:
            mask = mask0 & (ds.Streamfunction[ilayer,:,:] > lims[0]) & (ds.PV[ilayer,:,:] < vlim)#(ds.Volume[ilayer,:,:] > vlim )
            mask = mask.values.T
        else:
            mask = mask0 & (ds.Streamfunction[ilayer,:,:] > min(lims)) & (ds.Streamfunction[ilayer,:,:] < max(lims)) & (ds.PV[ilayer,:,:] < vlim)#(ds.Volume[ilayer,:,:] > vlim )
            mask = mask.values.T

        # remove isolated points in the mask
        structuring_element = np.ones((2,2), dtype=int)
        dilated_mask = binary_dilation(mask, structure=structuring_element)
        mask = binary_erosion(dilated_mask, structure=structuring_element)

        fig = plt.figure(figsize=(8,3))
        ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1)
        gl.xlabels_top = False ; gl.ylabels_left = True ; gl.ylabels_right = False
        ax.coastlines()
        im = ax.pcolor(ds.lon, ds.lat, (ds.Volume[ilayer,:,:]*mask), transform=ccrs.PlateCarree(), cmap='Blues')
        ##im = ax.pcolor(ds.lon, ds.lat, (ds.Sal[ilayer,:,:]*mask), transform=ccrs.PlateCarree(), cmap='viridis')
        plt.colorbar(im, label='Volume [m³]', shrink=0.5)
        ax.set_title(layer)
        plt.savefig('masks/mask_figures/mask_%s_%s_dum.png'%(name,layer), dpi=300)
        plt.show()
        plt.close()

        adjust = input('Need an adjustment? (y/n) ')

    # save the used limites to a csv
    tosave = {'Name':name, 'Layer':layer, 'Streamline limits':lims, 'PV limit':vlim}
    with open('wm_def_criteria.csv', 'a') as f_object:
        object = DictWriter(f_object, fieldnames=fieldnames)
        object.writerow(tosave)
        f_object.close()

    # ----
    # Finally, save the masks
    lims = 'save'#'none'
    if lims != 'none':

        # Put back on the original grid if modified longitudes
        if np.nanmin(ds.lon) < 0:
            mask2 = np.zeros(mask.shape)
            mask2[:,180:] = mask[:,:180]
            mask2[:,:180] = mask[:,180:]
            mask = mask2

        try:
            np.savetxt('masks/masks_%s_%s.csv'%(name,layer), mask.astype(int), delimiter=',')

            # Save the criteria used
            crit = [ name, layer, lims, vlim ]

            # then save the new line
            with open('masks/criteria_masks_global.csv', mode='a', newline='') as savefile:
                    csv_writer = csv.writer(savefile, delimiter=';')
                    csv_writer.writerow(crit)
        except:
            print('No mask')

if np.nanmin(ds.lon) > 0:
    print('save grid')
    np.savetxt('masks/masks_global_Lon.csv', ds.lon.values, delimiter=',')
    np.savetxt('masks/masks_global_Lat.csv', ds.lat.values, delimiter=',')
