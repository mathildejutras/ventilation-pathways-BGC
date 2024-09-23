Description of the data in causes_XX.csv

XX provides the abbreviation of the name of the Mode Water (see paper for full name)

Variables:

- distance: distance bin, in km

- mean_per_dist_bin_XX, where XX is Temperature, Salinity, Doxy, Nitrate: Mean value of all the data found within this distance bin.
	Units: 	Temperature -> degC
		Salinity -> psu
		Doxy -> micromol/kg
		Nitrate -> micromol/kg
		
- std_per_dist_bin_XX: Same, but with the standard deviation associated with the mean

- dOtot: Total change in oxygen from the first to the last bin, in micromol/kg

- dObgc: Total change in oxygen from the first to the last bin due to respiration, in micromol/kg

- dObgc_std: Uncertainty on this value

- dOmix: Total change in oxygen from the first to the last bin due to mixing, in micromol/kg

- dNtot, dNbgc, dNbgc_std, dNmix: same, but for nitrate

- fO: Fraction of water advected from the previous distance bin

- fO_std: Uncertainty on this number

- fabove: Fraction of water mixing in from the overlying layer

- fbelow: Fraction of water mixing in from the underlying layer

- O2sat: Oxygen at saturation
