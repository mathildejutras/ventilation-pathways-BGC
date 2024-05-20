# Solve the mixing equations along identified pathways, using Monte Carlo approach

# ---------------
# Mathilde Jutras
# contact: mjutras@hawaii.edu
# March 2024
# ---------------

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os
import gsw
from scipy.optimize import curve_fit
import csv
import warnings
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
import matplotlib.ticker as plticker

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --------------------
# FUNCTIONS
# --------------------

def solve_mixing_eqs(sprev, sabove, sbelow, shere, oprev, oabove, obelow, ohere, nprev, nabove, nbelow, nhere, r):
    A = np.array([[ sprev, sabove, sbelow, 0 ],
                  [ oprev, oabove, obelow, 1 ],
                  [ nprev, nabove, nbelow, r ],
                  [ 1, 1, 1, 0 ] ])
    B = np.array([shere, ohere, nhere, 1 ])

    x = np.linalg.solve(A, B)

    return x


def gauss(x, A, sigma, x0):
    y = A*np.exp(-((x-x0)/(2*sigma))**2)
    return y


def find_peak( hist, edgebins ):

    if edgebins[1]-edgebins[0] > 0.8:
        sigma0 = 20 # for delta O
    else:
        sigma0 = 0.1 # for fractions

    # check if there is a clear peak
    sorted = np.sort(hist)
    if ((sorted[-1]-sorted[-2]) > max(hist)/3) | (len( np.where(hist > 0)[0] ) < 3):
        peak = edgebins[ np.where(hist == max(hist))[0][0] ] + (edgebins[1]-edgebins[0])/2
        std = edgebins[1]-edgebins[0]
    else: # else find peak
        try:
            params, cov = curve_fit( gauss, edgebins[:-1], hist, p0=[max(hist),sigma0,edgebins[np.where(hist==max(hist))[0][0]]] )
            y = gauss(edgebins[:-1],params[0],params[1],params[2])
            peak = edgebins[ np.where(y == max(y))[0][0] ] + (edgebins[1]-edgebins[0])/2
            std = abs(params[1])
        except:
            print('Could not fit a gaussian on the distribution. Taking the max instead')
            peak = edgebins[ np.where(hist == max(hist))[0][0] ] + (edgebins[1]-edgebins[0])/2
            std = edgebins[1]-edgebins[0]

    # Plot the first time you run the script, to make sure the peak identification works fine
    #plt.close()
    #plt.plot(edgebins[:-1], hist, '.')
    #try:
    #    plt.plot(edgebins[:-1], y)
    #except:
    #    print('clear peak')
    #plt.plot(peak, max(hist), '*')
    #plt.plot([dum-std, dum-std], [0, max(hist)], 'r')
    #plt.plot([dum+std, dum+std], [0, max(hist)], 'r')
    #plt.title('std = %.2f'%std)
    #plt.show()
    #plt.close()

    return peak, std

def calc_aou(lon, lat, pres, S, T, O2) :
    O2sol = calc_o2sat(lon, lat, pres, S, T)
    AOU = O2sol - O2
    return AOU

def calc_o2sat(lon, lat, pres, S, T) :
    SA = gsw.SA_from_SP(S, pres, lon, lat)
    CT = gsw.CT_from_t( SA, T, pres )
    O2sol = gsw.O2sol(SA, CT, pres, lon, lat)
    return O2sol

# --------------------
# SCRIPT
# --------------------

# ---- Set-up ----

# Choose regions to solve for
regions = ['Pac'] #['IOSTMW', 'NASPMW', 'SASTMW', 'PacAAIW', 'NPCMW', 'Pac', 'Ind', 'NPCMW']

# Monte Carlo set-up
r = -16/154
numiter = 10000

path_data = ### REPLACE WITH PATH WHERE THE DATA SUBSET ARE SAVED

# --------------


variables = ['Doxy', 'Nitrate', 'Temperature']
name_var = {'Doxy':'O$_2$', 'Nitrate':'NO$_3^-$'}
name_wm = {'PacAAIW':'AAIW$_{Pacific}$', 'Pac':'SAMW$_{Pacific}$', 'Ind':'SAMW$_{Indian}$'}

n=0
for reg in regions:
    print(reg)

    # First, load the data and water masses definitions for waters above and below
    if reg == 'Pac' or reg == 'Ind':
        sec = 'SO'
    else:
        sec = 'global'
    path = path_data + '%s/'%sec
    files = os.listdir(path)

    filesreg = [each for each in files if reg in each]
    filesreg = [each for each in filesreg if 'final' in each ]
    layerdens = [float(each[-7:-3]) for each in filesreg]
    layerdens = np.sort(layerdens)
    if len(layerdens) == 0:
        print('ERROR IN THE NAME OF THE WATER MASS, OR FILES MISSING FOR THIS WATER MASS')
        break

    # initialize the output variables
    fabove = {} ; fbelow = {} ; dObgc = {} ; fini = {} ; oprev_list = {} ; nprev_list = {} ; dOmix = {} ; O2sat = {} ; O2sat_std = {} ; dNbgc = {} ; dNmix = {}
    fabove_final = {} ; fbelow_final = {} ; dObgc_final = {} ; dNbgc_final = {} ; fini_final = {} ; dOmix_final = {} ; dNmix_final = {} ; dOtot = {} ; dNtot = {}
    dOmix_final_std = {} ; dNmix_final_std = {} ; dObgc_final_std = {} ; dNbgc_final_std = {} ; dist_wm_layers = {} ; fini_final_std = {} ; fabove_final_std = {} ; fbelow_final_std = {}
    for layer in layerdens:
        fabove[layer] = {} ; fbelow[layer] = {} ; dObgc[layer] = {} ; dNbgc[layer] = {} ; fini[layer] = {} ; dOmix[layer] = {} ; dNmix[layer] = {} ; O2sat[layer] = {} ; O2sat_std[layer] = {}
        fabove_final[layer] = [] ; fbelow_final[layer] = [] ; dObgc_final[layer] = [] ; dNbgc_final[layer] = [] ; fini_final[layer] = [] ; oprev_list[layer] = [] ; nprev_list[layer] = [] ; dOmix_final[layer] = [] ; dNmix_final[layer] = [] ; dOtot[layer] = [] ; dNtot[layer] = []
        dObgc_final_std[layer] = [] ; dNbgc_final_std[layer] = [] ; dOmix_final_std[layer] = [] ; dNmix_final_std[layer] = [] ; fini_final_std[layer] = [] ; fabove_final_std[layer] = [] ; fbelow_final_std[layer] = []

    # initialize the figure
    fig, axs = plt.subplots(2,len(layerdens),figsize=(4*len(layerdens),8))
    if len(axs.shape) == 1:
        axs = axs[:,np.newaxis]
    # run for all layers
    c=0
    idr_all = {}
    for layer in layerdens:

        name = reg+'_'+str(layer)
        print(name)

        # retrieve the surrounding water masses definitions
        ds = xr.open_dataset(path+'Argo_glodap_final_%s_%s.nc'%(reg,layer))
        dist_wm = ds.Distance_bins.values
        dist_wm_layers[layer] = dist_wm

        # split per distance bins
        dists = ds.Distance_layer.values

        # calculate AOU
        AOU = calc_aou(ds.Lons_layer.values, ds.Lats_layer.values, ds.Pressure_layer.values, ds.Salinity_layer.values, ds.Temperature_layer.values, ds.Doxy_layer.values)

        # split into distance bins along the pathways
        mean_per_dist_bin = {'Temperature':[], 'Salinity':[], 'Nitrate':[], 'Doxy':[], 'Lons':[], 'Lats':[], 'Pressure':[], 'AOU':[]}
        std_per_dist_bin  = {'Temperature':[], 'Salinity':[], 'Nitrate':[], 'Doxy':[], 'Lons':[], 'Lats':[], 'Pressure':[], 'AOU':[]}
        k=0
        dd=dist_wm[1]-dist_wm[0]
        hist, edges = np.histogram(dists, bins = range(0,int(max(dists)),dd))
        idx = np.digitize(dists, edges)
        for id in range(1,max(idx)):
            idxl = np.where( idx==id )[0]
            distsl = np.array(dists)[idxl]

            node = []
            for var in ['Temperature', 'Salinity', 'Doxy', 'Nitrate', 'Lons', 'Lats', 'Pressure']:
                ser = ds[var+'_layer'].values[idxl]
                mean = np.nanmean(ser)
                node.append( mean )
                mean_per_dist_bin[var].append( mean )
                std_per_dist_bin[var].append( np.nanstd(ser) )

            mean_per_dist_bin['AOU'].append( np.nanmean(AOU[idxl]) )
            std_per_dist_bin['AOU'].append( np.nanstd(AOU[idxl]) )

        idr = list(range(max(idx)-1))
        idr_all[layer] = idr

        # initial values
        idxi = np.where(dists<min(dists)+dd/2)[0]
        inis = np.nanpercentile(ds.Salinity_layer.values[idxi], 25)
        init = np.nanpercentile(ds.Temperature_layer.values[idxi], 25)
        inin = np.nanpercentile(ds.Nitrate_layer.values[idxi], 25)
        inio = np.nanpercentile(ds.Doxy_layer.values[idxi], 75)
        if ((reg == 'NASPMW') & (layer == 27.2)) | ((reg == 'SASTMW') & (layer == 26.6)):
            inis = mean_per_dist_bin['Salinity'][0]
            init = mean_per_dist_bin['Temperature'][0]
            inin = mean_per_dist_bin['Nitrate'][0]
            inio = mean_per_dist_bin['Doxy'][0]

        print('Mean values per distance bins:', mean_per_dist_bin)
        print('Mean values per distance bins above:', ds['above mean Temperature'].values, ds['above mean Salinity'].values, ds['above mean Nitrate'].values, ds['above mean Doxy'].values)
        print('Mean values per distance bins below:', ds['below mean Temperature'].values, ds['below mean Salinity'].values, ds['below mean Nitrate'].values, ds['below mean Doxy'].values)
        print('For distances', edges)

        #--- Plot mixing diagrams
        x = ds.Temperature_layer.values
        if reg == 'NPCMW':
            x[x<8] = np.nan
        elif reg == 'NPSTMW':
            x[x<15] = np.nan
            x[x>18] = np.nan

        um = '$\mu$mol kg$^{-1}$'
        varunits = {'Temperature':'$^\circ$C', 'Doxy':um, 'Nitrate':um}

        for var in ['Doxy', 'Nitrate']:
            if var == 'Doxy':
                iniv = inio
            elif var == 'Nitrate':
                iniv = inin

            y = ds[var+'_layer'].values
            if (var == 'Doxy') & (reg == 'NPCMW') :
                y[y<100] = np.nan
            elif (var == 'Doxy') & (reg == 'NPSTMW') :
                y[y<150] = np.nan

            if var == 'Nitrate':
                ax = axs[0,c]
                ax.set_title('%.2f kg/m³'%layer)
            else:
                ax = axs[1,c]

            # change over pathway
            idxf = np.where(dists>np.nanmax(dists)-dd)[0]
            fv = np.nanmean(y[idxf])
            if var == 'Doxy':
                do = fv-iniv

            ax.scatter(x, y, s=1, marker = '+', c='lightgrey', zorder=0)#c=dists, vmin=0, vmax=10000, zorder=0)
            sp = ax.scatter(mean_per_dist_bin['Temperature'], mean_per_dist_bin[var], marker='*', c=edges[:-1], vmin=0, vmax=np.nanmax(ds['Distance_layer']), s=300, edgecolor='k', zorder=1)
            ax.scatter([mean_per_dist_bin['Temperature'][i] for i in idr], [mean_per_dist_bin[var][i] for i in idr], marker='*', c=[edges[i] for i in idr], vmin=0, vmax=np.nanmax(ds['Distance_layer']), s=600, edgecolor='k', zorder=1)
            ax.scatter(ds.Temperature_above_mean, ds['%s_above_mean'%var], marker='s', c=dist_wm, edgecolor='k', linewidth=2, label='Waters above', vmin=0, vmax=np.nanmax(ds['Distance_layer']), zorder=3)
            ax.scatter(ds.Temperature_below_mean, ds['%s_below_mean'%var], marker='o', c=dist_wm, edgecolor='k', linewidth=2, label='Waters below', vmin=0, vmax=np.nanmax(ds['Distance_layer']), zorder=3)
            ax.scatter([init, init], [iniv, iniv], marker='*', c='tab:orange', edgecolor='k', s=300, label='Initial')
            if var == 'Nitrate':
                mval = 4 ; symbol = '^'
            elif var == 'Doxy':
                mval = -50 ; symbol = 'v'
            ax.plot( [init, init], [iniv, iniv+mval], '-', c='tab:orange', linewidth=2, label='Bio. activity' )
            ax.scatter( [init], [iniv+mval], marker=symbol, c='tab:orange', linewidth=2)
            if var == 'Nitrate':
                ax.plot( [min(ds.Temperature_below_mean)-0.1, min(ds.Temperature_below_mean)-0.1], [iniv, fv], '-', c='grey', linewidth=4, alpha=0.7, label='$\Delta$N' )
                ax.plot( [min(ds.Temperature_below_mean)-0.15, min(ds.Temperature_below_mean)-0.15], [iniv, iniv-do/154*16], ':', c='grey', linewidth=4, alpha=0.7, label='$\Delta$O$_2$' )
            elif var == 'Doxy':
                ax.plot( [min(ds.Temperature_below_mean)-0.1, min(ds.Temperature_below_mean)-0.1], [iniv, fv], ':', c='grey', linewidth=4, alpha=0.7, label='' )
            ax.set_xlabel('Temperature [' + varunits['Temperature'] + ']')
            ax.set_ylabel(var + ' [' + varunits[var] + ']')
            ax.grid()
            if var == 'Nitrate':
                ax.yaxis.set_major_locator(plticker.MultipleLocator(base=2.5))
            elif var == 'Doxy':
                ax.yaxis.set_major_locator(plticker.MultipleLocator(base=25))
            ax.xaxis.set_major_locator(plticker.MultipleLocator(base=1))

        c+=1

        # Solve mixing equations -----------------------------------------------

        # For each distance bin
        skip = 0
        for i in range(len(idr_all[layer])):
            print(i, '/', idr_all[layer][-1])

            fabove[layer][i] = [] ; fbelow[layer][i] = [] ; dObgc[layer][i] = [] ; fini[layer][i] = [] ; dOmix[layer][i] = []
            dNbgc[layer][i] = [] ; dNmix[layer][i] = []

            if (idr[i] == idr[0]) | ((i==1) & (skip==1)):
                # define previous point
                tprev = init ; nprev = inin ; oprev = inio ; sprev = inis
            else:
                tprev = mean_per_dist_bin['Temperature'][idr[i-1-skip]]
                sprev = mean_per_dist_bin['Salinity'][idr[i-1-skip]]
                nprev = mean_per_dist_bin['Nitrate'][idr[i-1-skip]]
                oprev = mean_per_dist_bin['Doxy'][idr[i-1-skip]]
            oprev_list[layer].append(oprev)
            nprev_list[layer].append(nprev)

            # mixing equations we will solve
            # x0S0 + xaSa + xbSb = Sobs
            # x0O0 + xaOa + xbO0 + dO2 = Oobs
            # x0N0 + xaNa + xbNb - rdO2 = Nobs
            # x0 + xa + xb = 1

            O2sat[layer] = [ calc_o2sat(mean_per_dist_bin['Lons'][j], mean_per_dist_bin['Lats'][j], mean_per_dist_bin['Pressure'][j], mean_per_dist_bin['Salinity'][j], mean_per_dist_bin['Temperature'][j]) for j in idr ]
            O2sat_std[layer] = [ ( calc_o2sat(mean_per_dist_bin['Lons'][j], mean_per_dist_bin['Lats'][j], mean_per_dist_bin['Pressure'][j], mean_per_dist_bin['Salinity'][j], mean_per_dist_bin['Temperature'][j]+std_per_dist_bin['Temperature'][j]) - calc_o2sat(mean_per_dist_bin['Lons'][j], mean_per_dist_bin['Lats'][j], mean_per_dist_bin['Pressure'][j], mean_per_dist_bin['Salinity'][j], mean_per_dist_bin['Temperature'][j]-std_per_dist_bin['Temperature'][j]) )/2 for j in idr ]

            dO = mean_per_dist_bin['Doxy'][idr[i]]-oprev
            dN = mean_per_dist_bin['Nitrate'][idr[i]]-nprev
            if dO < 0: # check if the change in oxygen is negative. Otherwise, we will skip this point
                dOtot[layer].append(dO)
                dNtot[layer].append(dN)

                vars = ['Salinity', 'Temperature']
                finil = [] ; fabovel = [] ; fbelowl = [] ; dObgcl = []

                for iter in range(numiter):

                    finill = [] ; fabovell = [] ; fbelowll = [] ; dObgcll = []
                    for var in vars:

                        if var == 'Salinity':
                            pprev = sprev # physical parameter
                        else:
                            pprev = tprev

                        pert = np.random.normal(loc=0, scale=1, size=4)
                        # Apply the same magnitude of perturbation for S/T, N, O2 for each iteration, since not completely random
                        pert_p = pert*std_per_dist_bin[var][idr[i]]/2
                        pert_o = pert*std_per_dist_bin['Doxy'][idr[i]]/2
                        pert_n = pert*std_per_dist_bin['Nitrate'][idr[i]]/2

                        # check shape perturbation. Uncomment when run for the first time, to check that the perturbation is ok
                        #from scipy.stats import norm
                        #perts = []
                        #for ii in range(10000):
                        #    pertdum = np.random.normal(loc=0, scale=1) * std_per_dist_bin[var][idr[i]]/2
                        #    perts.append(pertdum)
                        #plt.close()
                        #plt.hist(perts, density=True)
                        #x = np.arange(-0.1,0.1,0.001)
                        #plt.title('Std of value = %.3f'%(std_per_dist_bin[var][idr[i]]/2))
                        #plt.plot(x, norm.pdf(x, 0, std_per_dist_bin[var][idr[i]]/2))
                        #plt.xlabel('Value')
                        #plt.savefig('figures/%s/example_distribution_values.png'%sec, dpi=300)

                        x = solve_mixing_eqs( pprev+pert_p[0],  ds['%s_above_mean'%var][idr[i]].values+pert_p[1],   ds['%s_below_mean'%var][idr[i]].values+pert_p[2],   mean_per_dist_bin[var][idr[i]]+pert_p[3],
                                            oprev+pert_o[0],    ds['Doxy_above_mean'][idr[i]].values+pert_o[1],     ds['Doxy_below_mean'][idr[i]].values+pert_o[2],     mean_per_dist_bin['Doxy'][idr[i]]+pert_o[3],
                                            nprev+pert_n[0],    ds['Nitrate_above_mean'][idr[i]].values+pert_n[1],  ds['Nitrate_below_mean'][idr[i]].values+pert_n[2],  mean_per_dist_bin['Nitrate'][idr[i]]+pert_n[3],
                                            r)

                        finill.append( x[0] )
                        fabovell.append( x[1] )
                        fbelowll.append( x[2] )
                        dObgcll.append( x[3] )

                    finil.append(finill) ; fabovel.append(fabovell) ; fbelowl.append(fbelowll) ; dObgcl.append(dObgcll)

                # Save one example solution with the distribution for T and S, to show that similar and can take average
                if len(layerdens) > 1:
                    ll = 1
                else:
                    ll = 0
                if (layer == layerdens[ll]) & (n==0):
                        np.savetxt( 'outputs/S_output_monte_carlo.csv', np.array([np.array(finil)[:,0], np.array(fabovel)[:,0], np.array(fbelowl)[:,0]]), delimiter=',' )
                        np.savetxt( 'outputs/T_output_monte_carlo.csv', np.array([np.array(finil)[:,1], np.array(fabovel)[:,1], np.array(fbelowl)[:,1]]), delimiter=',' )

                # average the outputs for T and S and keep only the data that has a proper solution (fraction > 0)
                for j in range(len(finil)):
                    good = False
                    if (finil[j][0] > 0) & (finil[j][0] < 1) & (finil[j][1] > 0) & (finil[j][1] < 1) & (fabovel[j][0] > 0) & (fabovel[j][0] < 1) & (fabovel[j][1] > 0) & (fabovel[j][1] < 1) & (fbelowl[j][0] > 0) & (fbelowl[j][0] < 1) & (fbelowl[j][1] > 0) & (fbelowl[j][1] < 1) & (dObgcl[j][0] < 0) & (dObgcl[j][1] < 0) :
                        finil_mean = (finil[j][0]+finil[j][1])/2
                        fabovel_mean = (fabovel[j][0]+fabovel[j][1])/2
                        fbelowl_mean = (fbelowl[j][0]+fbelowl[j][1])/2
                        dObgcl_mean = (dObgcl[j][0]+dObgcl[j][1])/2
                        good = True
                    elif (finil[j][0] > 0) & (finil[j][0] < 1) & (fabovel[j][0] > 0) & (fabovel[j][0] < 1) & (fbelowl[j][0] > 0) & (fbelowl[j][0] < 1) & (dObgcl[j][0] < 0) :
                        finil_mean = finil[j][0]
                        fabovel_mean = fabovel[j][0]
                        fbelowl_mean = fbelowl[j][0]
                        dObgcl_mean = dObgcl[j][0]
                        good = True
                    elif (finil[j][1] > 0) & (finil[j][1] < 1) & (fabovel[j][1] > 0) & (fabovel[j][1] < 1) & (fbelowl[j][1] > 0) & (fbelowl[j][1] < 1) & (dObgcl[j][1] < 0) :
                        finil_mean = finil[j][1]
                        fabovel_mean = fabovel[j][1]
                        fbelowl_mean = fbelowl[j][1]
                        dObgcl_mean = dObgcl[j][1]
                        good = True

                    if good: # if we have at least one good solution
                        fini[layer][i].append( finil_mean )
                        fabove[layer][i].append( fabovel_mean )
                        fbelow[layer][i].append( fbelowl_mean )
                        dObgc[layer][i].append( dObgcl_mean )
                        dNbgc[layer][i].append( dObgcl_mean*r )

                        # calculate the contribution of mixing
                        # dOmix = faOa + fbOb - (fa+fb)Oi
                        dOmix[layer][i].append( fabovel_mean * ds['Doxy_above_mean'][idr_all[layer][i]].values + fbelowl_mean * ds['Doxy_below_mean'][idr_all[layer][i]].values - (fabovel_mean+fbelowl_mean)*oprev_list[layer][i] )
                        dNmix[layer][i].append( fabovel_mean * ds['Nitrate_above_mean'][idr_all[layer][i]].values + fbelowl_mean * ds['Nitrate_below_mean'][idr_all[layer][i]].values - (fabovel_mean+fbelowl_mean)*nprev_list[layer][i] )

                # if could not find a solution to the set of equations, keep the previous point for the next calculations
                if (len(dOmix[layer][i]) == 0) :
                    skip = skip+1
                else:
                    skip = 0

                n+=1

            else: # if positive change in oxygen, skip to the next point
                skip = skip+1
                dOtot[layer].append(np.nan)
                dNtot[layer].append(np.nan)


    # finish water mass plot
    axs[0,0].legend(loc='lower right', bbox_to_anchor=(1.15,-0.1))

    fig.subplots_adjust(bottom=0.2, wspace=0.3, hspace=0.2, left=0.1, right=0.98, top=0.95)
    cbar_ax = fig.add_axes([0.45, 0.1, 0.1, 0.02])
    fig.colorbar(sp, cax=cbar_ax, label='Distance [km]', orientation='horizontal')

    plt.savefig('figures/%s/wm_analysis_%s.png'%(sec,reg), dpi=300)
    plt.close()


    # --------------
    # figure with an example of a solution, to illustrate the method
    if sec == 'Pac':
        layer = 26.9 ; i = 0
        f, axs = plt.subplots(1,2,figsize=(6,3))
        bins = np.arange(0,1,0.05)
        axs[0].hist(fini[layer][i], bins=bins, alpha=0.5, label='f$_{0}$')
        axs[0].hist(fabove[layer][i], bins=bins, alpha=0.5, label='f$_{Overlying}$')
        axs[0].hist(fbelow[layer][i], bins=bins, alpha=0.5, label='f$_{Underlying}$')
        axs[0].set_xlim([0,1])
        axs[0].grid()
        axs[0].legend()
        axs[0]
        bins = np.linspace(np.percentile(flist_bgc,1), np.percentile(flist_bgc,99), 20)
        axs[1].hist(dObgc[layer][i], bins=bins, alpha=0.5, label='f$_{ini}$', color='k')
        axs[1].grid()
        axs[0].set_xlabel('Fraction') ; axs[1].set_xlabel('R$_{O_2}$ [$\mu$mol/kg]')
        axs[0].text(0.05, 0.95, '(a)', fontweight='bold', transform=axs[0].transAxes)
        axs[1].text(0.05, 0.95, '(a)', fontweight='bold', transform=axs[1].transAxes)
        plt.tight_layout()
        plt.savefig('figures/%s/monte_carlo_example_solution_Pac_26.9.png'%sec, dpi=300)
        plt.close()

    # --------------------------------------------------------------------------
    # Calculate the percentages this represents
    for layer in layerdens:

        # mask the distance when there is a nan
        dOtot[layer] = [ dOtot[layer][i] if ((np.isnan(dOtot[layer][i])==False) & (np.isnan(dObgc_final[layer][i])==False)) else np.nan for i in range(len(dOtot[layer])) ]
        dObgc_final[layer] = [ dObgc_final[layer][i] if ((np.isnan(dOtot[layer][i])==False) & (np.isnan(dObgc_final[layer][i])==False)) else np.nan for i in range(len(dOtot[layer])) ]
        dOmix_final[layer] = [ dOmix_final[layer][i] if ((np.isnan(dOtot[layer][i])==False) & (np.isnan(dObgc_final[layer][i])==False)) else np.nan for i in range(len(dOtot[layer])) ]

        dOtotsum = np.nancumsum(dOtot[layer])
        dObgcsum = np.nancumsum(dObgc_final[layer])
        dOmixsum = np.nancumsum(dOmix_final[layer])

        dOtotsum = [ dOtotsum[i] if np.isnan(dOtot[layer][i])==False else np.nan for i in range(len(dOtot[layer])) ]
        dObgcsum = [ dObgcsum[i] if np.isnan(dObgc_final[layer][i])==False else np.nan for i in range(len(dOtot[layer])) ]
        dOmixsum = [ dOmixsum[i] if np.isnan(dOmix_final[layer][i])==False else np.nan for i in range(len(dOtot[layer])) ]

        stdObgcsum = [] ; stdOmixsum = []
        for i in range(len(dObgc_final_std[layer])):
            if i==0:
                stdObgcsum.append( dObgc_final_std[layer][i] )
                stdOmixsum.append( dOmix_final_std[layer][i] )
            else:
                stdObgcsum.append( np.sqrt( np.nansum([ dObgc_final_std[layer][i]**2, stdObgcsum[i-1]**2 ]) ) )
                stdOmixsum.append( np.sqrt( np.nansum([ dOmix_final_std[layer][i]**2, stdOmixsum[i-1]**2 ]) ) )

        perc_mix = dOmixsum / np.nancumsum(dOtot[layer]) *100
        perc_bgc = dObgcsum / np.nancumsum(dOtot[layer]) *100
        print('Mixing percentage :', perc_mix)
        print('BGC percentage :', perc_bgc)

        perc_bgc[np.isinf(perc_bgc)] = np.nan
        idxf = (~np.isnan(perc_bgc)).cumsum().argmax()

        # calculate the mixing coefficient
        # kappa = dO2mix * dz / dt / dC/dZ = dO2mix * dz² / (dt * dC)
        k = dOmix_final[layer] * dz^2 / ( age * (O2above - O2) )
        print('kappa = %.1e'%k)

        # width of the bars
        if dist_wm_layers[layer][1]-dist_wm_layers[layer][0] < 1000:
            w = 150
        else:
            w = 300

        # Save the output
        output = {'distance':dist_wm_layers[layer], 'dOtot':dOtot[layer], 'dObgc':dObgc_final[layer], 'dObgc_std':dObgc_final_std[layer], 'dOmix':dOmix_final[layer], 'dOmix_std':dOmix_final_std[layer], 'dNtot':dNtot[layer], 'dNbgc':dNbgc_final[layer], 'dNbgc_std':dNbgc_final_std[layer], 'dNmix':dNmix_final[layer], 'dNmix_std':dNmix_final_std[layer], 'fO':fini_final[layer], 'fO_std':fini_final_std[layer], 'fabove':fabove_final[layer], 'fabove_std':fabove_final_std[layer], 'fbelow':fbelow_final[layer], 'fbelow_std':fbelow_final_std[layer], 'O2sat':O2sat[layer], 'O2sat_std':O2sat_std[layer]}
        with open('outputs/causes_dO_%s_%s_%.1f.csv'%(sec,reg,layer), 'w') as csvfile:
            fieldnames = list(output.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(len(list(output.values())[0])):
                row = {key: output[key][i] for key in fieldnames}
                writer.writerow(row)

    # ------------------
    # plot the proportion of mixing and respiration along the pathway
    files = os.listdir('outputs/')
    files = [each for each in files if reg+'_' in each]
    files = [each for each in files if 'causes' in each]
    files = np.sort(files)

    colors = ['tab:red', 'darkblue', 'tab:green', 'tab:orange']

    list_max = [] ; dist_max = []
    for file in files:
        data = pd.read_csv('outputs/%s'%file)
        list_max.append( np.nanmin(np.nancumsum(data['dObgc']+data['dOmix'])) )
        dist_max.append( np.nanmax(data['distance']) )

    fig = plt.figure(figsize=(4,len(files)+2))
    gs = GridSpec(len(files), 1, height_ratios=list_max)
    c=0
    bs = [] ; perc_along = [] ; std_along = []
    axs = []
    for file in files:
        data = pd.read_csv('outputs/%s'%file)
        layer = float(file[-8:-4])

        if c==0:
            ax = plt.subplot(gs[c])
        else:
            ax = plt.subplot(gs[c], sharex=axs[-1])

        dObgcsum = np.nancumsum(data['dObgc'])
        dOmixsum = np.nancumsum(data['dOmix'])

        dObgcsum = [ dObgcsum[i] if np.isnan(data['dObgc'][i])==False else np.nan for i in range(len(data['dObgc'])) ]
        dOmixsum = [ dOmixsum[i] if np.isnan(data['dOmix'][i])==False else np.nan for i in range(len(data['dObgc'])) ]

        stdObgcsum = [] ; stdOmixsum = []
        for i in range(len(data['dObgc'])):
            if i==0:
                stdObgcsum.append( data['dObgc_std'][i] )
                stdOmixsum.append( data['dOmix_std'][i] )
            else:
                stdObgcsum.append( np.sqrt( np.nansum([ data['dObgc_std'][i]**2, stdObgcsum[i-1]**2 ]) ) )
                stdOmixsum.append( np.sqrt( np.nansum([ data['dOmix_std'][i]**2, stdOmixsum[i-1]**2 ]) ) )

        perc_mix = np.divide(dOmixsum, np.add(dOmixsum, dObgcsum)) *100
        perc_bgc = np.divide(dObgcsum, np.add(dOmixsum, dObgcsum)) *100
        perc_bgc[np.isinf(perc_bgc)] = np.nan
        idxf = (~np.isnan(data['dObgc'])).cumsum().argmax()

        if data['distance'][1]-data['distance'][0] < 1000:
            w = 400
        else:
            w = 850

        ax.bar(data['distance'], dObgcsum, width=w, color=colors[c], label='Respiration')
        ax.errorbar(data['distance'], dObgcsum, yerr=stdObgcsum, capsize=4, c='grey', fmt='_')
        ax.bar(data['distance'], dOmixsum, width=w, bottom=dObgcsum, color=colors[c], alpha=0.3, label='Mixing')
        ax.errorbar(data['distance'], np.add(dOmixsum,dObgcsum), yerr=stdOmixsum, capsize=4, c='grey', fmt='_')
        ax.text(data['distance'].values[idxf]+w/2, -4, '%d%%'%perc_bgc[idxf], fontsize=8)
        ax.text(data['distance'].values[idxf]+w/2, dObgcsum[idxf]-4, '%d%%'%perc_mix[idxf], fontsize=8)
        ax.yaxis.set_major_locator(MultipleLocator(25))
        ax.set_xticks(range(5000,max(dist_max),5000))
        ax.set_axisbelow(True)
        ax.grid(linewidth=0.5)

        ax.text(0.01, 0.02, '%.1f kg/m³'%layer, transform=ax.transAxes)

        c+=1
        axs.append(ax)

        perc_along.append( data['dObgc'].values/(data['dObgc'].values+data['dOmix'].values)*100 )
        std_along.append( np.sqrt( (data['dObgc_std'].values/(data['dObgc'].values+data['dOmix'].values)*100)**2 + (data['dOmix_std'].values/(data['dObgc'].values+data['dOmix'].values)*100)**2 )/2 )

    axs[0].legend(loc='upper right', fontsize=8)
    axs[0].set_xlim([0,max(dist_max)+2000])
    axs[0].set_ylabel('$\Delta$O [$\mu$mol/kg]')
    if (reg in name_wm) & (reg != 'Pac'):
        axs[0].set_title(name_wm[reg])
    else:
        axs[0].set_title(reg)
    axs[c-1].set_xlabel('Distance along pathway [km]')
    plt.subplots_adjust(hspace=0, left=0.2, top=0.95)
    plt.savefig('figures/%s/monte_carlo_percentages_%s_%s_cum.png'%(sec,sec,reg), dpi=300)
    plt.close()


# plot a figure that compares the results when using T and S
resT = np.genfromtxt('outputs/T_output_monte_carlo.csv', delimiter=',')
resS = np.genfromtxt('outputs/S_output_monte_carlo.csv', delimiter=',')
f, axs = plt.subplots(1,2,sharex=True, sharey=True, figsize=(6,3))
bins=np.arange(0,1,0.05)
axs[0].hist(resT[0,:], bins=bins, alpha=0.5, label='f$_{ini}$')
axs[0].hist(resT[1,:], bins=bins, alpha=0.5, label='f$_{above}$')
axs[0].hist(resT[2,:], bins=bins, alpha=0.5, label='f$_{below}$')
axs[1].hist(resS[0,:], bins=bins, alpha=0.5, label='f$_{ini}$')
axs[1].hist(resS[1,:], bins=bins, alpha=0.5, label='f$_{above}$')
axs[1].hist(resS[2,:], bins=bins, alpha=0.5, label='f$_{below}$')
axs[0].set_title('Using T') ; axs[1].set_title('Using S')
axs[0].set_xlim([0,1])
axs[0].grid() ; axs[1].grid()
plt.xlabel('Fraction')
plt.legend()
plt.tight_layout()
plt.savefig('figures/%s/monte_carlo_compare_T_S.png'%sec, dpi=300)
