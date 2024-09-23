# Script to select the BGC Argo and GLODAP data within each water mass, and save it
# Extract:
#   1) The data in wanted density bins along the pathways
#   2) All the data above and below, to make section plots
#   3) The definition of waters above and below along the pathways
# ---------------
# Mathilde Jutras
# contact: mjutras@hawaii.edu
# Nov 2023
# Modified March 2024
# ---------------

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import glob, os
import pandas as pd
from scipy.interpolate import interp1d
import math
from scipy.spatial import distance
from matplotlib.colors import LinearSegmentedColormap
import seawater as sw
from scipy.signal import savgol_filter
from pathway_utils import *

import warnings
warnings.filterwarnings('ignore')

# ------------------
# FUNCTIONS
# ------------------

def apply_QC_flags_argo(ds, svar):

	floatn = ds.PLATFORM_NUMBER.values.astype('float')[0]

	s = svar+'_ADJUSTED'
	sf = s+'_QC'
	flags = ds[sf].values.astype('float')
	var = ds[s].values

	var[ (flags == 3) | (flags == 4) | (flags == 9) ] = np.nan

	# check if no QC on some data
	n = np.where(flags == 0)
	if len(n[0]) > 0:
		print('some no QC in %i %s'%(floatn,svar))

		plt.figure(figsize=(3,4))
		plt.scatter(var, ds.PRES.values, s=1, c='b')
		plt.scatter(var[flags==2], ds.PRES.values[flags==2], s=1, c='r')
		plt.ylabel('Pressure [dbar]') ; plt.xlabel(svar)
		plt.gca().invert_yaxis()
		plt.tight_layout()
		plt.show()
		plt.close()

		user_decision = input('Should we keep this data? (y/n)  ')
		if user_decision == 'N' or user_decision == 'n':
			var[ flags == 0 ] == np.nan
		elif user_decision == 'Y' or user_decision == 'y':
			pass
		else:
			print('Error with the user decision. Using the data')

	# check if probably good data on some data
	n = np.where(flags == 2)
	if len(n[0]) > 0:
		print('some probably good data in %i %s (in red)'%(floatn,svar))

		plt.figure(figsize=(3,4))
		plt.scatter(var, ds.PRES.values, s=1, c='b')
		plt.scatter(var[flags==2], ds.PRES.values[flags==2], s=1, c='r')
		plt.ylabel('Pressure [dbar]') ; plt.xlabel(svar)
		plt.gca().invert_yaxis()
		plt.tight_layout()
		plt.show()
		plt.close()

		user_decision = input('Should we keep this data? (y/n)  ')
		if user_decision == 'N' or user_decision == 'n':
			var[ flags == 2 ] == np.nan
		elif user_decision == 'Y' or user_decision == 'y':
			pass
		else:
			print('Error with the user decision. Using the data')

	# pad to give the same length to all profiles
	var_pad = pad(var)

	return var_pad


def apply_QC_flags_glodap(df, svar):

	sf = 'G2'+svar+'f'
	flags = df[sf].values
	var = df['G2'+svar].values

	var[ (flags == 9) ] = np.nan

	return var

def get_alkalinity(dictin, which):

	msmts = np.array([dictin['Salinity '+which], dictin['Doxy '+which], dictin['Temperature '+which]]).T.astype(np.float64)
	measID = [1,6,7]
	coords = np.array([dictin['Lons '+which], dictin['Lats '+which], dictin['Pressure '+which]]).T.astype(np.float64)
	res = LIAR_matlab(liar_path, sw_path, coords, msmts, measID)
	alk = np.array(res).flatten()
	# then with nitrate when possible
	msmts = np.array([dictin['Salinity '+which], dictin['Nitrate '+which], dictin['Doxy '+which], dictin['Temperature '+which]]).T.astype(np.float64)
	measID = [1,3,6,7]
	res = LIAR_matlab(liar_path, sw_path, coords, msmts, measID)
	alk2 = res.flatten()
	# combine the two
	idx_nitrate = np.where(np.isnan(dictin['Nitrate '+which])==False)
	for i in idx_nitrate:
		alk[i] = alk2[i]

	return list(alk)


def load_data_argo(streamlines, paths_dist, layer, lons_mask, lats_mask, mask, name, dens1, dens2):

	path = ### REPLACE WITH PATH TO ARGO SPROF DATA
	files = os.listdir(path)
	files = [each for each in files if 'Sprof.nc' in each]

	print('Getting Argo data...')

 	# select the data
	wprop={'Lats layer':[], 'Lons layer':[], 'Date layer':[], 'Pressure layer':[], 'Distance layer':[], 'Doxy layer':[], 'Nitrate layer':[], 'Salinity layer':[], 'Temperature layer':[], 'DIC layer':[], 'pH layer':[], 'Lats column':[], 'Lons column':[], 'Date column':[], 'Pressure column':[], 'Distance column':[], 'Doxy column':[], 'Nitrate column':[], 'Salinity column':[], 'Temperature column':[], 'pH column':[], 'DIC column':[] }
	for f in files:

		ds = xr.open_dataset(path+f)

		l = list(ds.keys())
		if any(item in ['DOXY', 'NITRATE', 'PH_IN_SITU'] for item in l):

			lats = np.array(ds.LATITUDE.values)
			lons = ds.LONGITUDE.values
			lons = np.array([each+360. if each < 0. else each for each in lons])
			date = ds.JULD.values
			date = date.astype('datetime64[Y]').astype(float)+1970. + (date - date.astype('datetime64[Y]')).astype('timedelta64[D]').astype(int) / 365.

			# select the profiles that are within the mask
			ilon = closest(lons_mask[0,:], lons) # indices of the corresponding lat/lon in the mask
			ilat = closest(lats_mask[:,0], lats)
			maskl = mask[ilat,ilon]
			imask = [i for i in range(len(maskl)) if maskl[i]==1]

			if len(imask) > 0 :

				lats = lats[imask]
				lons = lons[imask]
				date = date[imask]
				press = ds.PRES.values[imask, :]
				press = pad(press)

				# Associate the data points with one of the streamlines
				dist = []
				strml_lons = [each[0] for each in streamlines]
				for i in range(len(lons)):
					# check if crossing meridian 0
					if ((max(lons) > 300) & (min(lons) < 100)) | ((max(strml_lons) > 300) & (lons[i] < 100)) | ((max(strml_lons) > 300) & (min(strml_lons) < 100)):
						strml = [ [each[0]-360, each[1]] if each[0]>180 else each for each in streamlines ]
						if lons[i] > 180:
							li = lons[i] - 360
						else:
							li = lons[i]
						idx = closest_node([li, lats[i]], strml)
					else:
						idx = closest_node([lons[i], lats[i]], streamlines)
					dist.append( paths_dist[idx] )
				dist = np.array(dist)

				# OXYGEN
				if 'DOXY3' in l:
					doxy1 = apply_QC_flags_argo(ds, 'DOXY')
					doxy2 = apply_QC_flags_argo(ds, 'DOXY2')
					doxy3 = apply_QC_flags_argo(ds, 'DOXY3')

					oxydum = np.array([doxy1, doxy2, doxy3])
					doxy = np.nanmean(oxydum, axis=0)[imask, :]

				elif 'DOXY2' in l:
					doxy1 = apply_QC_flags_argo(ds, 'DOXY')
					doxy2 = apply_QC_flags_argo(ds, 'DOXY2')

					oxydum = np.array([doxy1, doxy2])
					doxy = np.nanmean(oxydum, axis=0) [imask, :]

				elif 'DOXY' in l:
					doxy = apply_QC_flags_argo(ds, 'DOXY')[imask, :]
				else :
					doxy = np.ones(press.shape)*np.nan

				# NITRATE
				if 'NITRATE' in l:
					nitrate = apply_QC_flags_argo(ds, 'NITRATE')[imask, :]
					nitrate = nitrate
				else:
					nitrate = np.ones(press.shape)*np.nan
				# DIC

				if 'DIC' in l:
					dic = ds.DIC.values[imask,:]
					dic = pad(dic)
				else:
					dic = np.ones(press.shape)*np.nan

				# PH
				if 'PH_IN_SITU_TOTAL' in l:
					ph = apply_QC_flags_argo(ds, 'PH_IN_SITU_TOTAL')[imask, :]
					ph = ph
				else:
					ph = np.ones(press.shape)*np.nan

				# S
				if 'PSAL' in l:
					sal = apply_QC_flags_argo(ds, 'PSAL')[imask, :]
					sal = sal
				else:
					sal = np.ones(press.shape)*np.nan

				# T
				if 'TEMP' in l:
					temp = apply_QC_flags_argo(ds, 'TEMP')[imask, :]
					temp = temp
				else:
					temp = np.ones(press.shape)*np.nan

				# remove bad temperature data
				temp = np.where( temp < 0, np.nan, temp )

				# select this data
				doxy_flat = [] ; press_flat = [] ; sal_flat = [] ; temp_flat = [] ; ph_flat = [] ; nitrate_flat = [] ; dic_flat = []
				lons_flat = [] ; lats_flat = [] ; dist_flat = [] ; date_flat = []
				for i in range(sal.shape[0]):
					if np.isnan(sal[i,:]).all() == False:
					    dum = sal[i,:][::-1]
					    inonan = np.where(~np.isnan(dum))[0][0]
					    inonan = len(dum)-inonan-1

					    doxy_flat.extend( doxy[i,0:inonan] )
					    press_flat.extend( press[i,0:inonan] )
					    sal_flat.extend( sal[i,0:inonan] )
					    temp_flat.extend( temp[i,0:inonan] )
					    ph_flat.extend( ph[i,0:inonan] )
					    nitrate_flat.extend( nitrate[i,0:inonan] )
					    dic_flat.extend( dic[i,0:inonan] )

					    lons_flat.extend( [[lons[i]] * inonan][0] )
					    lats_flat.extend( [[lats[i]] * inonan][0] )
					    date_flat.extend( [[date[i]] * inonan][0] )
					    dist_flat.extend( [[dist[i]] * inonan][0] )

				if len(lats_flat) > 0:
					    wprop['Lats column'].extend( lats_flat )
					    wprop['Lons column'].extend( lons_flat )
					    wprop['Date column'].extend( date_flat )
					    wprop['Doxy column'].extend( doxy_flat )
					    wprop['Pressure column'].extend( press_flat )
					    wprop['Salinity column'].extend( sal_flat )
					    wprop['Temperature column'].extend( temp_flat )
					    wprop['pH column'].extend( ph_flat )
					    wprop['Nitrate column'].extend( nitrate_flat )
					    wprop['DIC column'].extend( dic_flat )
					    wprop['Distance column'].extend( dist_flat )

				# --------------------------------------------------------------
				# Put apart the wanted density range
				# Get the MLD
				mlds = []
				[mlds.append( mld_dbm_v3(temp[i,:], sal[i,:], press[i,:], 0.03) ) for i in range(temp.shape[0])]
				# use the deepest mld of the whole float
				max_mld = np.nanmax(mlds)
				if np.isnan(max_mld) == False:
					max_mld_lat = lats[ mlds.index(max_mld) ]
					mld_pres = sw.pres(max_mld, max_mld_lat)
				else:
					mld_pres = 150.

				# and remove within MLD
				#sigma_theta = sw.eos80.pden(sal, temp, press)
				sigma_theta = pad( np.genfromtxt(path+'neutral_densities/nden_'+f[0:7]+'.csv', delimiter=',') )[imask, :]
				idens = np.where( (sigma_theta >= dens1) & (sigma_theta < dens2) & (press > mld_pres))#np.array(mld_pres)[:, np.newaxis]) )

				# select the data points within the wanted density range
				if len(idens[0]) == 0:
					print('Weird, no point in the density range')
				lats = lats[idens[0]] ; lons = lons[idens[0]] ; date = date[idens[0]] ; press = press[idens[0], idens[1]] ; dist = dist[idens[0]]
				doxy = doxy[idens[0], idens[1]] ; sal = sal[idens[0], idens[1]] ; temp = temp[idens[0], idens[1]] ; nitrate = nitrate[idens[0], idens[1]] ; ph = ph[idens[0], idens[1]]

				# select this data
				wprop['Lats layer'].extend( lats )
				wprop['Lons layer'].extend( lons )
				wprop['Date layer'].extend( date )
				wprop['Doxy layer'].extend( doxy )
				wprop['Pressure layer'].extend( press )
				wprop['Salinity layer'].extend( sal )
				wprop['Temperature layer'].extend( temp )
				wprop['pH layer'].extend( ph )
				wprop['Nitrate layer'].extend( nitrate )
				wprop['DIC layer'].extend( dic )
				wprop['Distance layer'].extend( dist )

	if len(lons) == 0:
			print('EMPTY SUBSET')

	return wprop


def load_data_glodap(streamlines, paths_dist, layer, lons_mask, lats_mask, mask, name, dens1, dens2):

	path = ### REPLACE WITH PATH TO GLODAP DATA
	df = pd.read_csv(path+'GLODAPv2.2022_Merged_Master_File.csv')

	print('Getting GLODAP data...')

 	# select the data
	lats = df.G2latitude.values
	lons = df.G2longitude.values
	dates = df.G2year.values + (df.G2month.values-1)/12 + (df.G2day.values-1)/365
	press = df.G2pressure.values
	sal   = apply_QC_flags_glodap(df, 'salinity')
	temp  = df.G2temperature.values
	doxy  = apply_QC_flags_glodap(df, 'oxygen')
	nitrate = apply_QC_flags_glodap(df, 'nitrate')
	ph	= apply_QC_flags_glodap(df, 'phtsinsitutp')
	alk   = apply_QC_flags_glodap(df, 'talk')
	phosp = apply_QC_flags_glodap(df, 'phosphate')
	cruises = df.G2cruise.values
	stations = df.G2station.values
	nden = df.G2gamma.values
	dic = apply_QC_flags_glodap(df, 'tco2')

    # remove bad temperature data
	temp = np.where( temp < 0, np.nan, temp )

	# select the profiles that are within the mask
	lons = np.where( lons < 0, lons+360., lons )
	ilon = closest(lons_mask[0,:], lons) # indices of the corresponding lat/lon in the mask
	ilat = closest(lats_mask[:,0], lats)
	idx = [i for i,v in enumerate(lons) if v<3]
	maskl = mask[ilat,ilon]
	imask = [i for i in range(len(maskl)) if maskl[i]==1]

	lats = lats[imask]; lons = lons[imask]; dates = dates[imask]; press = press[imask] ; nden = nden[imask]
	sal  = sal[imask]; temp = temp[imask]; doxy = doxy[imask]; nitrate = nitrate[imask]; ph   = ph[imask]; alk  = alk[imask]; phosp = phosp[imask]
	cruises = cruises[imask] ; stations = stations[imask] ; dic = dic[imask]

	# Associate the data points with one of the streamlines
	dist = []
	strml_lons = [each[0] for each in streamlines]
	for i in range(len(lons)):
		# check if crossing meridian 0
		if ((max(lons) > 300) & (min(lons) < 100)) | ((max(strml_lons) > 300) & (min(lons) < 100)) | ((max(strml_lons) > 300) & (min(strml_lons) < 100)):
			strml = [ [each[0]-360, each[1]] if each[0]>180 else each for each in streamlines ]
			if lons[i] > 180:
				li = lons[i] - 360
			else:
				li = lons[i]
			idx = closest_node([li, lats[i]], strml)
		else:
			idx = closest_node([lons[i], lats[i]], streamlines)
		dist.append( paths_dist[idx] )
	dist = np.array(dist)

	wprop = {}
	wprop['Lons column'] = lons
	wprop['Lats column'] = lats
	wprop['Date column'] = dates
	wprop['Doxy column'] = doxy
	wprop['Pressure column'] = press
	wprop['Salinity column'] = sal
	wprop['Temperature column'] = temp
	wprop['pH column'] = ph
	wprop['Nitrate column'] = nitrate
	wprop['Distance column'] = dist
	wprop['alk column'] = alk
	wprop['Phosphate column'] = phosp
	wprop['DIC column'] = dic

	# --------------------------------------------------------------
	# Put apart the wanted density range
	# Remove points in the MLD
	list_cruise = list(set(cruises))
	imlds = []
	for cruise in list_cruise:
		station_nums = list(set(stations[ np.where(cruises == cruise)[0] ]))
		mldsl = []
		for stn in station_nums:
			istn_cruise = np.where( (cruises==cruise) & (stations==stn) )[0]
			mldsl.append( mld_dbm_v3( temp[istn_cruise], sal[istn_cruise], press[istn_cruise], 0.03 ) )
		#get the deepest mld over the cruise
		max_mld = np.nanmax(mldsl)
		del mldsl
		imld = np.where( (cruises==cruise) & (press > max_mld) )[0]
		imlds.extend(imld)

	lats = lats[imlds]; lons = lons[imlds]; dates = dates[imlds]; press = press[imlds] ; dist = dist[imlds] ; nden = nden[imlds]
	sal  = sal[imlds]; temp = temp[imlds]; doxy = doxy[imlds]; nitrate = nitrate[imlds]; ph   = ph[imlds]; alk  = alk[imlds]; phosp = phosp[imlds]

	# Select only the wanted density range
	sigma_theta = nden
	idens = list(np.where( (sigma_theta >= dens1) & (sigma_theta < dens2) )[0])
	if len(idens) == 0:
		print('weird, no data in the density range')

	lats = lats[idens] ; lons = lons[idens] ; dates = dates[idens] ; press = press[idens] ; dist = dist[idens]
	doxy = doxy[idens] ; sal = sal[idens] ; temp = temp[idens] ; nitrate = nitrate[idens] ; ph = ph[idens] ; phosp = phosp[idens] ; alk = alk[idens] ; dic = dic[idens]

	wprop['Lons layer'] = list(lons)
	wprop['Lats layer'] = list(lats)
	wprop['Date layer'] = list(dates)
	wprop['Doxy layer'] = list(doxy)
	wprop['Pressure layer'] = list(press)
	wprop['Salinity layer'] = list(sal)
	wprop['Temperature layer'] = list(temp)
	wprop['pH layer'] = list(ph)
	wprop['Nitrate layer'] = list(nitrate)
	wprop['Distance layer'] = list(dist)
	wprop['alk layer'] = list(alk)
	wprop['Phosphate layer'] = list(phosp)
	wprop['DIC layer'] = list(dic)

	return wprop



# ------------------
# SCRIPT
# ------------------

# ---- Set-up ----

# Regions to extract
regions = ['Pac', 'Ind', 'NASTMW', 'NASPMW', 'NPCMW', 'SASTMW', 'NPSTMW', 'PacAAIW']

# paths
path_mask = ### REPLACE WITH PATH CONTAINING THE MASKS
path_pv = ### REPLACE WITH PATH TO PV_GLOBAL.NC
path_save = ### REPLACE WITH WHERE WANT TO SAVE THE DATASET

# -----------------


# these features are used to select the streamlines along which to calculate the distance
num_strml   = {'Ind_26.7':4, 'Ind_26.8':2, 'Pac_27.0':6, 'Pac_27.1':5, 'NASTMW_26.4':6,
				'NASPMW_27.0':3, 'NASPMW_27.1':2, 'NASPMW_27.2':3, 'NPCMW_26.2':5, 'NPCMW_26.3':5,
				'SASTMW_26.6':10, 'NPSTMW_25.3':10}
filt_window = {'Ind_26.8':40, 'Pac_27.0':15, 'NPSTMW_25.2':30}
manual_strml = {'Pac_26.8_x':[[360+each for each in [-152.6, -162.3, -170.4, -175.3, -177.2]]],
				'Pac_26.8_y':[[-45.8, -41.9, -37.4, -33.0, -28.1]],
				'Pac_26.9_x':[[360+each for each in [-136.1, -145.8, -157.1, -169.5, -178.1]]],
				'Pac_26.9_y':[[-49.9, -42.5, -40.6, -36.6, -36.0]],
				'Ind_26.7_x':[[94.7, 83.8, 69.9, 51.8]], 'Ind_26.7_y':[[-38.8, -31.9, -26.8, -23.2]],
				'NPCMW_26.1_x':[[168.1, 179.5, 360-168.8, 360-166.4, 360-172.4, 173.6, 150.2]],
				'NPCMW_26.1_y':[[34.7, 35.2, 33.6, 29.4, 25.0, 22.7, 19.6]],
				'SASTMW_26.5_x':[[360-27.1, 360-10.8, 360-6, 360-12.1, 360-21.1, 360-30.8]],
				'SASTMW_26.5_y':[[-35.9, -35.8, -33, -30, -29.2, -28.5]],
				'SASTMW_26.6_x':[[360-33.4, 360-20.5, 360-4.1, 2, 360-5.7, 360-21.8]],
				'SASTMW_26.6_y':[[-38, -38.1, -36.4, -33.5, -30.5, -28.3]],
				'NPSTMW_25.2_x':[[150.9, 155.4, 158, 155, 143], [150.9, 155.4, 162.5, 158.5, 150.9, 143.1], [150.9, 159.1, 169.6, 161.9]],
				'NPSTMW_25.2_y':[[33.7, 32, 29.9, 26.7, 25.5], [33.7, 32.5, 29.8, 24.7, 24.3, 24.3], 	  [33.5, 31.9, 29.5, 24.6]],
				'NPSTMW_25.3_x':[[149.8, 155.4, 158, 155, 143], [149.8, 155.4, 162.5, 158.5, 150.9, 143.1]],
				'NPSTMW_25.3_y':[[33.7, 32, 29.9, 26.7, 25.5], [33.5, 32.5, 29.8, 24.7, 24.3, 24.3]],
				'NPSTMW_25.4_x':[[151.9, 155.8, 157.7, 153.9, 144.7, 136.1], [151.9, 151.8, 156.1, 167.6, 163, 154.6]],
				'NPSTMW_25.4_y':[[33.8, 32.4, 29.7, 26.5, 24.7, 24.0],     [33.8, 33.4, 32.4, 30.5, 25.5, 23.3]]
    			    			}
ini_lon	 = {'SO_Pac_26.8':202.5 , 'SO_Pac_26.9':223.5, 'SO_Pac_27.0':360-134., 'SO_Pac_27.1':360-110,
			'SO_Ind_26.6':66.3, 'SO_Ind_26.7':91.9, 'SO_Ind_26.8':98.5, 'NASTMW_26.4':360-59.6,
			'NASPMW_27.1':360-21.2, 'NASPMW_27.0':360-18.9, 'NASPMW_27.2':360-20,
			'NPCMW_26.1':360-168, 'NPCMW_26.2':163, 'NPCMW_26.3':163,
			'SASTMW_26.5':360-27.1, 'SASTMW_26.6':360-34.6,
			'NPSTMW_25.2':149, 'NPSTMW_25.3':149, 'NPSTMW_25.4':149,
			'IOSTMW_26.6':59.8, 'PacAAIW_27.2':360-126.1, 'PacAAIW_27.3':360-131.9}
ini_lat	 = {'SO_Pac_26.8':-47.5, 'SO_Pac_26.9':-49.5,  'SO_Pac_27.0':-52,  'SO_Pac_27.1':-52.4,
			'SO_Ind_26.6':-41.6, 'SO_Ind_26.7':-50.7, 'SO_Ind_26.8':-49.4, 'NASTMW_26.4':38.2,
			'NASPMW_27.1':54.8, 'NASPMW_27.0':54.5, 'NASPMW_27.2':54.5,
			'NPCMW_26.1':34.7, 'NPCMW_26.2':34.8, 'NPCMW_26.3':34.8,
			'SASTMW_26.5':-37.9, 'SASTMW_26.6':-39.7,
			'NPSTMW_25.2':35.2, 'NPSTMW_25.3':35.2, 'NPSTMW_25.4':35.2,
			'IOSTMW_26.6':-33.9, 'PacAAIW_27.2':-50.7, 'PacAAIW_27.3':-52.6}
cut = {'SO_Ind_26.6':[0,2], 'SO_Ind_26.7':[0,3], 'SO_Ind_26.8':[0],
		'SO_Pac_27.0':[0,1], 'SO_Pac_27.1':[0,3], 'NASTMW_26.4':[0,2],
		'NASPMW_27.0':[0], 'NPCMW_26.3':[0,1,3],
		'SASTMW_26.5':[0,1,2], 'SASTMW_26.6':[0,3,4], 'NPSTMW_25.3':[0], 'NPSTMW_25.4':[0],
		'IOSTMW_26.6':[5],
		'PacAAIW_27.2':{0:100, 1:100, 2:110, 3:110}, 'PacAAIW_27.3':[0,1]}
inversed_strmlines = ['NASTMW', 'NPCMW', 'NPSTMW']

distbins = {'Pac':1000, 'Ind':500}

files = os.listdir(path_mask+'validated/')

# Loop through all regions and density layers
for reg in regions:
	filesreg = [each for each in files if reg in each]
	layerdens = [float(each[-8:-4]) for each in filesreg]
	layerdens.sort()

	if reg in ['SASTMW']:
		central_lon = 0
	else:
		central_lon = 180

	if reg == 'Pac' or reg == 'Ind':
		sec = 'SO'
	else:
		sec = 'global'

	for layer in layerdens[1:]:
		if sec == 'SO':
			name = 'SO_'+reg+'_%.1f'%layer
			densname = 'Neutral density'
			namesave = reg+'_%.1f'%layer
		else:
			name = reg+'_%.1f'%layer
			densname = 'density'
			namesave = name

		dens0 = layer # lower density bound
		dens1 = dens0+0.1 # higher density bound

		print(' ')
		print('Doing layer', name)


		# 1) Define the pathway based on the mask -----------------------

		wprop = {}

		mask = np.genfromtxt(path_mask+'validated/masks_%s.csv'%name, delimiter=',')
		lons_mask = np.genfromtxt(path_mask+'masks_%s_Lon.csv'%sec, delimiter=',')
		lats_mask = np.genfromtxt(path_mask+'masks_%s_Lat.csv'%sec, delimiter=',')
		if sec == 'global':
			lons_mask = np.repeat(lons_mask[np.newaxis,:], mask.shape[0], axis=0)
			lats_mask = np.repeat(lats_mask[:,np.newaxis], mask.shape[1], axis=1)
		path_data = path_pv+'%s_netcdf/'%sec
		ds = xr.open_dataset(path_data + 'PV_%s.nc'%sec)
		ilayer = np.where( np.round(ds[densname],1) == layer )[0][0]
		strmf = ds.Streamfunction.isel( density=ilayer )
		del ds

		# ---
		# Extract streamlines along which to follow the waters
		if name+'_x' not in manual_strml: # Get from streamlines
			num_strml_l = num_strml[name] if name in num_strml else 8
			filt_window_l = filt_window[name] if name in filt_window else 15
			cut_l = cut[name] if name in cut else None

			strmf_masked = np.ma.masked_where(mask==0, strmf) # apply the mask to the streamlines
			cl = plt.contour(lons_mask, lats_mask, strmf_masked, num_strml_l) # keep only the continuous lines
			plt.close()

			# Extract the streamlines, smooth them, and apply the volume mask
			paths_x = [] ; paths_y = []
			c=0
			for l in cl.collections:
				if len(l.get_paths()) > 0:
					paths = l.get_paths()
					for path in paths:

						# Display and save the full-length closed streamlines we will want to use
						if len(path.vertices[:,0]) > 15 :
							x = np.flip(path.vertices[:,0])
							y = np.flip(path.vertices[:,1])

                            # keep only the main part
							x_pieces = [] ; y_pieces = []
							icut = 0
							iscut = False
							for i in range(len(x)-1):
							    if (abs(x[i+1]-x[i]) > 1.5) or (abs(y[i+1]-y[i]) > 1.5) :
							        iscut = True
							        x_pieces.append( x[icut+1:i] )
							        y_pieces.append( y[icut+1:i] )
							        icut = i
							if iscut:
							        x_pieces.append( x[icut+1:-1] )
							        y_pieces.append( y[icut+1:-1] )
                            # keep only the main piece
							if iscut :
							    imax = np.argmax( [len(each) for each in x_pieces] )
							    x = x_pieces[imax] ; y = y_pieces[imax]

							# Smooth
							if len(x) > 15 :
							    y = savgol_filter(y, np.min( [filt_window_l, len(path.vertices[:,0])] ) , 3)

							    # Manually cut some some streamlines or pieces to avoid for instance weird convergence in the distances
							    if (cut_l != None) :
							        if (isinstance(cut_l,dict)==False) & (c in cut_l):
							            print('Ignore streamline', c)

							        #elif (cut_l != None) & (c == cut_l[0]):
							        #    x = np.concatenate(( x[0:cut[name][1][0]], x[cut[name][1][1]:-1] ))
							        #    y = np.concatenate(( y[0:cut[name][1][0]], y[cut[name][1][1]:-1] ))
							        #    paths_x.append(x[::-1])
							        #    paths_y.append(y[::-1])

							        elif isinstance(cut_l,dict) & (c in cut_l) :
										print('Cut streamline', c)
							            paths_x.append( x[::-1][0:cut_l[c]] )
							            paths_y.append( y[::-1][0:cut_l[c]] )

							        else:
							            paths_x.append(x[::-1])
							            paths_y.append(y[::-1])
							    else:
							        paths_x.append(x[::-1])
							        paths_y.append(y[::-1])

							    c+=1

		else: # when streamlines are too ugly, get from manually defined path
			paths_x = [] ; paths_y = []
			for i in range(len(manual_strml[name+'_x'])):
				theta = np.linspace(0, 2*np.pi, num=len(manual_strml[name+'_x'][i]), endpoint=False)
				if central_lon == 0:
					manual_strml[name+'_x'][i] = [ each-360 if each > 180 else each for each in manual_strml[name+'_x'][i] ]
				interp_x = interp1d(theta, manual_strml[name+'_x'][i], kind='cubic', bounds_error=False)
				interp_y = interp1d(theta, manual_strml[name+'_y'][i], kind='cubic', bounds_error=False)
				theta_interp = np.linspace(0, 2*np.pi, num=50, endpoint=False)
				resx = np.array(interp_x(theta_interp))
				resy = np.array(interp_y(theta_interp))
				dumx = list(resx[~np.isnan(resx)])
				if central_lon == 0:
					paths_x.append( [each+360 if each < 0 else each for each in dumx] )
				else:
					paths_x.append( dumx )
				paths_y.append( list(resy[~np.isnan(resy)]) )

		# Calculate the distance along the streamlines
		paths_dist = []
		for i in range(len(paths_x)):

			if (name[:-5] in inversed_strmlines) & (name+'_x' not in manual_strml): # if need to reverse the order of the streamlines
				paths_x[i] = paths_x[i][::-1]
				paths_y[i] = paths_y[i][::-1]

			if (central_lon == 0) & (min(paths_x[i]) < 100): # correct if cross 0 meridian
				if ini_lon[name] > 180:
					il = ini_lon[name] - 360
				else:
					il = ini_lon[name]
				if paths_x[i][0] > 180:
					px = paths_x[i][0] - 360
				else:
					px = paths_x[i][0]
				dist_strm = [ dist_latlon( [il, ini_lat[name]], [px, paths_y[i][0]] ) ]
				if any(np.array(paths_x[i]) > 180):
					px = [ each-360 if each > 180 else each for each in paths_x[i] ]
				else:
					px = paths_x[i]
				dist_strm.extend( [ dist_latlon([px[j], paths_y[i][j]], [px[j+1], paths_y[i][j+1]]) for j in range(len(px)-1) ] )
			else:
				dist_strm = [ dist_latlon( [ini_lon[name], ini_lat[name]], [paths_x[i][0], paths_y[i][0]] ) ]
				dist_strm.extend( [ dist_latlon([paths_x[i][j], paths_y[i][j]], [paths_x[i][j+1], paths_y[i][j+1]]) for j in range(len(paths_x[i])-1) ] )
			dist_strm = list(np.cumsum(dist_strm))
			paths_dist.extend(dist_strm)

		paths_x = [item for sublist in paths_x for item in sublist]
		paths_y = [item for sublist in paths_y for item in sublist]
		streamlines = [ [paths_x[i], paths_y[i]] for i in range(len(paths_x)) ]

		fig = plt.figure(figsize=(10, 4))
		ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=central_lon))
		ax.pcolor(lons_mask, lats_mask, mask, transform=ccrs.PlateCarree())
		sp = ax.scatter(paths_x, paths_y, c=paths_dist, s=5, transform=ccrs.PlateCarree(), cmap='jet', vmin=0)
		plt.colorbar(sp, label='Distance [km]')
		idxmask = np.where(mask==1)
		londum = lons_mask[0,idxmask[1]]
		latdum = lats_mask[idxmask[0],0]
		if central_lon == 180:
			ax.set_extent([min(londum)-5, max(londum)+5, min(latdum)-5, max(latdum)+5])
		else:
			ax.set_extent([315, 20, min(latdum)-5, max(latdum)+5])
		ax.coastlines()
		ax.set_title('Check if the streamlines seem good')
		plt.savefig('figures/map_streamlines_%s.png'%name,dpi=300)
		plt.show()
		plt.close()

		# Save the pathways
		wprop['Streamlines_x'] = paths_x
		wprop['Streamlines_y'] = paths_y
		wprop['Distance along streamlines'] = paths_dist
		# ---

		# 2) Load the data ----------------------------------------------
		print('Extract the wanted subset')

		wprop_glodap = load_data_glodap(streamlines, paths_dist, layer, lons_mask, lats_mask, mask, name, dens0, dens1)
		wprop_argo = load_data_argo(streamlines, paths_dist, layer, lons_mask, lats_mask, mask, name, dens0, dens1)

		# 3) Get the alkalinity for argo----------------------------------
		print('Get alkalinity and pH for Argo')
		#liar_path = '~/Documents/Postdoc/Dataset/LIARv2'
		#sw_path = '~/matlab_tools/SeaWater'

		for each in ['layer', 'column']:
			alk = get_alkalinity(wprop_argo, each)
			wprop_argo['alk '+each] = alk

		# 4) Combine the datasets -----------------------------------------

		# combine Argo and GLODAP
		for each in ['layer', 'column']:
		    wprop_argo['Phosphate '+each] = list(np.ones(len(wprop_argo['Lons '+each]))*np.nan)
		for var in list(set(wprop_argo.keys())):
			wprop[var] = [*wprop_argo[var], *wprop_glodap[var]]
		del wprop_glodap, wprop_argo

		# plot map
		mycmap = LinearSegmentedColormap.from_list('mycmap', ['w', 'pink'])

		fig = plt.figure(figsize=(8, 7))
		ax = fig.add_subplot(2, 1, 1, projection=ccrs.PlateCarree(central_longitude=central_lon))
		ax.coastlines()
		if central_lon == 180:
			ax.set_extent([min(londum)-5, max(londum)+5, min(latdum)-5, max(latdum)+5])
		else:
			ax.set_extent([315, 20, min(latdum)-5, max(latdum)+5])
		sc = ax.scatter(wprop['Lons layer'], wprop['Lats layer'], c=wprop['Distance layer'], s=1, transform=ccrs.PlateCarree())
		plt.colorbar(sc, label='Distance [km]')
		gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=0.5, color='gray', alpha=0.5)
		gl.xlabels_top = False ; gl.ylabels_left = True ; gl.ylabels_right = False

		ax = fig.add_subplot(2, 1, 2, projection=ccrs.PlateCarree(central_longitude=central_lon))
		ax.coastlines()
		if central_lon == 180:
			ax.set_extent([min(londum)-5, max(londum)+5, min(latdum)-5, max(latdum)+5])
		else:
			ax.set_extent([315, 20, min(latdum)-5, max(latdum)+5])
		sc = ax.scatter(wprop['Lons layer'], wprop['Lats layer'], c=wprop['Doxy layer'], s=1, transform=ccrs.PlateCarree(), cmap='jet')
		plt.colorbar(sc, label='DO [uM]')
		gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=0.5, color='gray', alpha=0.5)
		gl.xlabels_top = False ; gl.ylabels_left = True ; gl.ylabels_right = False

		plt.suptitle('Check if should remove part of mask close to front')
		plt.savefig('figures/%s/map_profiles_within_pathway_%s.png'%(sec,name), dpi=300)
		plt.show()


		#5) Get the surrounding water masses properties ----------------------
		print(' ')
		print('For the surrounding waters...')
		wprop_sur = {}
		wprop_argo_below = load_data_argo(streamlines, paths_dist, layer, lons_mask, lats_mask, mask, name, dens1, dens1+0.1)
		wprop_glodap_below = load_data_glodap(streamlines, paths_dist, layer, lons_mask, lats_mask, mask, name, dens1, dens1+0.1)

		wprop_argo_above = load_data_argo(streamlines, paths_dist, layer, lons_mask, lats_mask, mask, name, dens0-0.1, dens0)
		wprop_glodap_above = load_data_glodap(streamlines, paths_dist, layer, lons_mask, lats_mask, mask, name, dens0-0.1, dens0)

		# Get alkalinity
		alk = get_alkalinity( wprop_argo_above, 'layer' )
		wprop_argo_above['alk layer'] = alk
		alk = get_alkalinity( wprop_argo_below, 'layer' )
		wprop_argo_below['alk layer'] = alk

		wprop_argo_above['Phosphate layer'] = list(np.ones(len(wprop_argo_above['Lons layer']))*np.nan)
		for var in ['Salinity', 'Temperature', 'Doxy', 'Nitrate', 'alk', 'Lons', 'Lats']:
			wprop_sur[var+' layer above'] = [*wprop_argo_above[var+' layer'], *wprop_glodap_above[var+' layer']]

		wprop_argo_below['Phosphate layer'] = list(np.ones(len(wprop_argo_below['Lons layer']))*np.nan)
		for var in ['Salinity', 'Temperature', 'Doxy', 'Nitrate', 'alk', 'Lons', 'Lats']:
			wprop_sur[var+' layer below'] = [*wprop_argo_below[var+' layer'], *wprop_glodap_below[var+' layer']]

		del wprop_glodap_above, wprop_argo_above, wprop_argo_below, wprop_glodap_below

#		# 5A) Get the water mass definition, from means along the pathway ----------------------------------------

		order_d = {'Doxy':'neg', 'Nitrate':'pos', 'Temperature':'pos', 'Salinity':'pos', 'alk':'pos'}
		for sur in ['above', 'below']:

			dist = []
			for i in range(len(wprop_sur['Lons layer '+sur])):
				idx = closest_node([wprop_sur['Lons layer '+sur][i], wprop_sur['Lats layer '+sur][i]], streamlines)
				dist.append( paths_dist[idx] )
			dist = np.array(dist)

			# Average in bins
			if reg in distbins:
				dd=distbins[reg] # distance bins, in km
			else:
				dd=1000
			hist, edges = np.histogram(dist, bins = range(0,int(max(dist)),dd))
			idx = np.digitize(dist, edges)

			for var in ['Doxy', 'Nitrate', 'Temperature', 'Salinity', 'alk'] :

			    wprop[sur+' mean '+var] = []
			    for id in range(1,max(idx)):
			    	idxl = np.where( idx==id )[0]
			    	distsl = np.array(dist)[idxl]
			    	varl = np.array(wprop_sur[var+' layer '+sur])[idxl]
			    	wprop[sur+' mean '+var].append( np.nanmean(varl) )
			wprop['Distance bins'] = edges[1:]


		# 6) Save -------------------------------------------------------

		print('Save')

		coords_layer = dict(
		    Distance_layer = ('x_layer', wprop['Distance layer']),
		    Pressure_layer = ('x_layer', wprop['Pressure layer']),
		    Date_layer = ('x_layer', wprop['Date layer']),
		    Lons_layer = ('x_layer', wprop['Lons layer']),
		    Lats_layer = ('x_layer', wprop['Lats layer']),
		    x_layer    = ('x_layer', range(len(wprop['Lats layer'])))
		    )

		coords_column = dict(
		    Distance_column = ('x_column', wprop['Distance column']),
		    Pressure_column = ('x_column', wprop['Pressure column']),
		    Date_column = ('x_column', wprop['Date column']),
		    Lons_column = ('x_column', wprop['Lons column']),
		    Lats_column = ('x_column', wprop['Lats column']),
		    x_column    = ('x_column', range(len(wprop['Lats column'])))
		    )

		coords_defs = dict(
			Distance_bins = ('x_bins', wprop['Distance bins']),
			x_bins		  = ('x_bins', range(len(wprop['Distance bins']))),
		)

		coords_streamlines = dict(
			x 			  = ('x', range(len(wprop['Streamlines_x']))),
		)

		das = []
		for var in ['Salinity', 'Temperature', 'Nitrate', 'Doxy', 'Phosphate', 'DIC', 'pH', 'alk']:

		    das.append( xr.DataArray(  name = '%s_layer'%var,
		                    data = wprop['%s layer'%var],
		                    dims = ['x_layer'],
		                    coords = coords_layer,
		                    ) )

		    das.append( xr.DataArray(  name = '%s_column'%var,
		                    data = wprop['%s column'%var],
		                    dims = ['x_column'],
		                    coords = coords_column,
		                    ) )

		for var in ['Salinity', 'Temperature', 'Nitrate', 'Doxy', 'alk', 'DIC', 'Oxygen']:

		    for each in ['above', 'below']:
		        das.append( xr.DataArray( name = '%s_%s_mean'%(var,each),
								data = wprop['%s mean %s'%(each,var)],
								dims = ['x_bins'],
								coords = coords_defs,
				 				) )

		das.append( xr.DataArray( name = 'Streamlines_x',
								data = wprop['Streamlines_x'],
								dims = ['x'],
								coords = coords_streamlines,
								) )

		das.append( xr.DataArray( name = 'Streamlines_y',
								data = wprop['Streamlines_y'],
								dims = ['x'],
								coords = coords_streamlines,
								) )

		das.append( xr.DataArray( name = 'Streamlines_distance',
								data = wprop['Distance along streamlines'],
								dims = ['x'],
								coords = coords_streamlines,
								) )

		ds = xr.merge(das)
		ds.attrs['description'] = 'Layer is within the density layer only. Column is with the whole water column. For the above and below waters, line and mean refer to two methods to obtain the defining properties of the waters.'
		ds.to_netcdf(path_save+'%s/Argo_glodap_final_%s.nc'%(sec,namesave), mode='w')
		del ds, das
