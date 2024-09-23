# Script to that contains the box model of ventilation
# The box model:
# ----- surface / fixed layer -----||----
#           layer 1
# ------------||--------------------------
#           layer 2
# ------------||--------------------------
#           layer 3
# ------------||--------------------------
# Oxygen, nitrate and temperature evolve in time under
# the effect of respiration, mixing, and advection
# ---------------
# Mathilde Jutras
# contact: mjutras@hawaii.edu
# Sept 2024
# ---------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import math
import matplotlib.patches as patches
from curlyBrace import curlyBrace


def o2sat(sal, temp):
# Adapted from o2satv2b.m by:  Edward T Peltzer, MBARI
# Transferred to Python by Zack Nachod, UH
#
#  CALCULATE OXYGEN CONCENTRATION AT SATURATION
#
#  Source:  Garcia & Gordon (1992).  Oxygen solubility in seawater:
#          Better fitting equations.  L&O 37: 1307-1312.
#
# Input:       S = Salinity (pss-78)
#              T = Temp (deg C)
#
# Output:      Oxygen saturation at one atmosphere (umol/kg).
#
#                        O2 = o2satv2b(S,T).
    A0 =  5.80818
    A1 =  3.20684
    A2 =  4.11890
    A3 =  4.93845
    A4 =  1.01567
    A5 =  1.41575

    B0 = -7.01211e-03
    B1 = -7.25958e-03
    B2 = -7.93334e-03
    B3 = -5.54491e-03

    C0 = -1.32412e-07

    Ts = math.log((298.15 - temp) / (273.15 + temp))

    A = ((((A5 * Ts + A4) * Ts + A3) * Ts + A2) * Ts + A1) * Ts + A0

    B = ((B3 * Ts + B2) * Ts + B1) * Ts + B0

    O2 = math.exp(A + sal * (B + sal * C0))

    return O2


def solve_model(combinations):

    # --------
    # Input: combinations if a list of combinations of the following parameters:
    # 1. O2sat = Fixed oxygen in the top layer, in micromol/kg
    # 2. R = total respiration in the middle (second) layer, in micromol/kg
    # R is an indirect way to define the respiration rate
    # 3. Ra = respiration rate in the first layer
    # 4. Rb = respiration rate in the third layer
    # 5. Ti = initial temperature in the middle layer, in degree celsius
    # 6. Tia = initial temperature in the first layer
    # 7. Tib = initial temperature in the third layer
    # 8. Ni = initial nitrate in the middle layer, in micromol/kg
    # 9. Nia = initial nitrate in the first layer
    # 10. Nib = initial nitrate in the third layer
    # 11. t = total propagation time in the middle layer, in years
    # 12. ta = total propagation time in the first layer, in years
    # 13. tb = total propagation time in the third layer, in years
    # 14. asfc = mixing coefficient in the middle layer, no units
    # 15. aa = mixing coefficient in the first layer
    # 16. ab = mixing coefficient in the third layer
    # 17. Nsat = fixed nitrate in the top layer, in micromol/kg
    # 18. Tsat = fixed temperature in the top layer
    # --------
    # Returns:
    # 1. O2out = Oxygen time series at a chosen set of conditions
    # 2. Nout = Nitrate time series at a chosen set of conditions
    # 3. dOf = Difference between initial and final oxygen
    # 4. Rcolumn = Total integrated respiration in all layers
    # 5. dO2column = Total difference between initial and final oxygen in all layer
    # 6. ratios = ratio in the final change of oxygen to nitrate
    # 7. aouf = Apparent oxygen utilization at the end
    # 8. AOUout = Time series of apparent oxygen utilization
    # --------

    # Stoichiometric ratio of oxygen to nitrate
    r = -16/154

    dOf = [] ; deltanf = []
    dO2column = [] ; Rcolumn = []
    ratios = [] ; aouf = []
    c=0 ; l=len(combinations)
    # For each combination...
    for set in combinations:
        if c%100 == 0:
            print(c,'/',l)

        # Extract each model parameter
        O2sat = set[0] ; Nsat = set[16] ; Tsat = set[17]
        R = set[1] ; Ra = set[2] ; Rb = set[3]
        Oi = o2sat(34.5, set[4]) ; Oia = o2sat(34.5, set[5]) ; Oib = o2sat(34.5, set[6])
        Ni = set[7] ; Nia = set[8] ; Nib = set[9]
        Ti = set[4] ; Tia = set[5] ; Tib = set[6]
        t = set[10] ; ta = set[11] ; tb = set[12]
        asfc = set[13] ; aa = set[14] ; ab = set[15]

        # --- Run the model
        dt = 1
        ts = np.arange(0,t,dt)
        O2 = [] ; O2A = [] ; O2B = []
        O2Aprev = Oia ; O2Bprev = Oib ; O2prev = Oi
        N = [] ; NA = [] ; NB = []
        T = [] ; TA = [] ; TB = []
        NAprev = Nia ; NBprev = Nib ; Nprev = Ni
        TAprev = Tia ; TBprev = Tib ; Tprev = Ti
        age = [] ; ageA = [] ; ageB = []
        Rtotprev = 0 ; RNtotprev = 0
        ageAprev = 0 ; ageBprev = 0 ; ageprev = 0
        ratio = []

        # Iterate through time steps
        for i in range(len(ts)):

            # Calculate oxygen and nitrate after respiration and mixing
            O2.append(  O2prev  - R/t  + aa*(O2Aprev-O2prev) - ab*(O2prev-O2Bprev) )
            O2A.append( O2Aprev - Ra/ta - aa*(O2Aprev-O2prev)  + asfc*(O2sat-O2Aprev) )
            O2B.append( O2Bprev - Rb/tb + ab*(O2prev-O2Bprev) )

            N.append(  Nprev  - R*r/t  + aa*(NAprev-Nprev) - ab*(Nprev-NBprev) )
            NA.append( NAprev - Ra*r/ta - aa*(NAprev-Nprev)  + asfc*(Nsat-NAprev) )
            NB.append( NBprev - Rb*r/tb + ab*(Nprev-NBprev) )

            # Calculate temperature and age mixing
            T.append(  Tprev  + aa*(TAprev-Tprev) - ab*(Tprev-TBprev) )
            TA.append( TAprev - aa*(TAprev-Tprev)  + asfc*(Tsat-TAprev) )
            TB.append( TBprev + ab*(Tprev-TBprev) )

            age.append( ts[i] + aa*(ageAprev-ageprev) - ab*(ageprev-ageBprev) )
            ageB.append( ts[i] + ab*(ageprev-ageBprev) )
            ageA.append( ts[i] - aa*(ageAprev-ageprev) )

            ageAprev = ageA[i] ; ageBprev = ageB[i] ; ageprev = age[i]

            # apparent stoichiometric ratio
            ratio.append( (N[i]-Nprev) / (O2[i]-O2prev) )

            # Save present conditions for next time step
            O2Aprev = O2A[i] ; O2Bprev = O2B[i] ; O2prev = O2[i]
            NAprev = NA[i] ; NBprev = NB[i] ; Nprev = N[i]
            TAprev = TA[i] ; TBprev = TB[i] ; Tprev = T[i]

#        # check that T evolution makes sense -- uncomment the first time run it
#        plt.plot(T, label='middle') ; plt.plot(TA, label='above') ; plt.plot(TB, label='bottom')
#        plt.plot(np.ones(len(T))*Tsat, label='sfc')
#        plt.legend()
#        plt.xlabel('Time') ; plt.ylabel('T')
#        plt.show()

        # Calculate OUR
        dOf.append( O2[-1]-Oi )
        deltan = [ -(N[i] - Ni) for i in range(len(N)) ]
        deltanf.append(deltan[-1])

        # Calculate AOU
        aouf.append( O2[-1] - o2sat(34.5,T[-1]) )

        ratios.append(ratio)

        # integrated over all the columns
        Rcolumn.append( -(R+Ra+Rb) )
        dO2column.append( (O2[i]-Oi) + (O2A[i]-Oia) + (O2B[i]-Oib) )

        # Save the time series for one pre-selected set of condition
        if Tsat == Tout :
            O2out = O2
            Nout = N
            AOUout = [O2[i] - o2sat(34.5,T[i]) for i in range(len(O2)) ]

        c+=1

    return O2out, Nout, dOf, Rcolumn, dO2column, ratios, aouf, AOUout


# ------
# SCRIPT
# ------

# Define a number of combinations, representative of oceanic conditions
# See function for what each input and output represents
Tlayer = 10
Tsat_list = list(range(5,25,1)) # equivalent to O2sat range of 200 to 320
Tsat_list.reverse()
O2sat_list = [o2sat(34.5,T) for T in Tsat_list]
Nsat_list = np.linspace(30,0,len(O2sat_list))
combinations_sat = [[O2sat_list[i], 30, 30, 30, Tlayer, Tlayer-2, Tlayer+2, 20, 22, 18, 100, 100, 100, 0.01, 0.01, 0.01, Nsat_list[i], Tsat_list[i]] for i in range(len(O2sat_list)) ]

# Condition at which we want to output the total time series, to plot it as an example
Tout = 18

O2, NO3, dO, Rcolumn, dO2column, ratios, aou, aouts = solve_model(combinations_sat)


# ---- Plot
f,[ax0,ax] = plt.subplots(2,1,figsize=(4.7,6))

ax0.plot(range(len(O2)), O2, 'grey', label='O$_{2,measured}$', zorder=2, linewidth=3)
ax0.plot(range(len(O2)), [O2[0]+each for each in aouts], '--', c='grey', label='AOU', zorder=2  , linewidth=3)
y = np.linspace(0,-30,len(O2))
ax0.plot(range(len(O2)), O2[0]+y, c='tab:green', label='Respiration', zorder=1, linewidth=2)
ax0.plot(range(len(O2)), O2-y, c='tab:blue', label='Mixing', zorder=1, linewidth=2)
ax0.legend(fontsize=8)
ax0.set_xlabel('Time [years]')
ax0.set_xlim([0,98])
ax0.set_title('Example solution\nO$_{2,surface}$ = %i $\mu$mol kg$^{-1}$'%o2sat(34.5,Tout))
ax0.set_ylabel('O$_2$ [$\mu$mol kg$^{-1}$]')
ax0.plot([95], [min(O2)+1.5], '*', c='k', markersize=12)

curlyBrace(f, ax0, [108, max(O2)], [108, min(O2)], 0.04, str_text='$\Delta O_2$', color='k', lw=1, fontdict={'weight':'bold'}, clip_on=False, int_line_num=1)
curlyBrace(f, ax0, [100, max(O2)], [100, max(O2)-30], 0.04, str_text='$R_{True}$', color='k', lw=1, fontdict={'weight':'bold'}, clip_on=False, int_line_num=1)

# a) Integrated R and dO2
O2sat_list = np.array(O2sat_list)
x = O2sat_list - O2[0]

ax.plot(x, dO, '-k', label='$\Delta O_2$')
ax.plot(x, [-30]*len(dO), '--k', label='R$_{True}$')
ax.set_ylabel('Respiration\n[$\mu$mol kg$^{-1}$]')
ax.legend(fontsize=8)
ax.set_xlim([x[0], x[-1]])
ax.set_ylim([min(dO)-1, max(dO)+1])
ax.yaxis.set_major_locator(plticker.MultipleLocator(base=10))
ax.xaxis.set_major_locator(plticker.MultipleLocator(base=50))
ax.grid(visible=True)

ax.plot([0, 0], [min(dO)-2, max(dO)+2], '-', c='grey', lw=2)
ax.plot([o2sat(34.5,Tout)-o2sat(34.5,Tlayer)], [-36], '*', c='k', markersize=12)

i = 14
ax.fill_between(x[:i], dO[:i], [-30]*len(dO[:i]), color='r', alpha=0.3)
ax.fill_between(x[i-1:], dO[i-1:], [-30]*len(dO[i-1:]), color='b', alpha=0.3)
ax.text(-63, -35.5, '$\Delta O_2$\nunderestimates\nR$_{True}$', fontsize=7.5, c='brown')
ax.text(33, -29.5, '$\Delta O_2$\noverestimates\nR$_{True}$', fontsize=7.5, c='darkblue', horizontalalignment='right')
ax.text(1, -40.5, 'O$_{2,surface}$=\nO$_{2}^{t=0}$', color='grey', fontweight='bold')
ax.fill_betweenx([min(dO)-1, max(dO)+1], [-10, -10], [-60, -60], color='grey', alpha=0.2)
ax.text(-12, -40.5, 'Range of O$_{2,above}$\nfor SAMW$_{Pac}$', c='grey', fontsize=7, horizontalalignment='right')

ax.set_title('Sensitivity to O$_{2,surface}$')
ax.set_xlabel('Difference between surface and layer oxygen,\nO$_{2,surface}$ - O$_2^{t=0}$ [$\mu$mol kg$^{-1}$]')
plt.tight_layout()
plt.savefig('figures/effect_sat_withT.png', dpi=300)
plt.close()
