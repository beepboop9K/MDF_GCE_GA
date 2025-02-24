#!/usr/bin/env python3
####################################################
#
# Author: M Joyce
#
# collab: Thomas Trueman, Christian Johnson
#
####################################################
import numpy as np
import glob
import sys
import subprocess
import matplotlib.pyplot as plt

import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.colors import Normalize

from scipy.interpolate import CubicSpline

sys.path.append('../../MESA/work_AGB_mesa-dev/')
import math_functions_lib as mfl

#cmap = mpl.cm.gist_rainbow.reversed()
cmap = mpl.cm.ocean
norm = mpl.colors.Normalize(vmin=1, vmax=375)
m = cm.ScalarMappable(norm=norm, cmap=cmap)

##########################################################
#
# Christian's observations
#
##########################################################
obs_file = 'binned_dist_lat6_0.08dex.dat'

feh, count=np.loadtxt(obs_file, usecols=(0,1), unpack=True)
normalized_count = count/count.max()


##########################################################
#
# simulation data
#
##########################################################
all_runs_file = "MDF_data_375runs.txt"
inf = open(all_runs_file, "r")
x_data = []
y_data = []
i = 0
myDict = {}
for line in inf:
	if line and "Model" not in line:
		p = line.split()
		x = p[0]
		y = p[1]

		x_data.append(float(x))
		y_data.append(float(y))

	elif "Model" in line:
		myDict['model_'+str(i)+'_label'] = line
		myDict["x_model"+str(i)] = np.array(x_data)
		myDict["y_model"+str(i)] = np.array(y_data)
		x_data = []
		y_data = []
		i = i + 1
	else:
		print('bad line encountered')
inf.seek(0)
inf.close()

N_models = i

## feh, normalized_count
placeholder_sigma_array = np.zeros(len(normalized_count)) + 1

scores = []
MDFs = []
model_numbers = []
for k in range(1,i):
	x = myDict["x_model"+str(k)]
	y = myDict["y_model"+str(k)]

	sample_at = 0.0001
	resampled_x = np.arange(x.min(), x.max(), sample_at)
	cs_MDF = CubicSpline(x, y)

	theory_count_array = [cs_MDF(resampled_x)[mfl.find_nearest(resampled_x, feh[i])[1]] for i in range(len(feh)) ]

	wrmse = mfl.compute_wrmse(theory_count_array,  normalized_count, placeholder_sigma_array) ## set all sigmas to 1

	scores.append(wrmse)
	MDFs.append(cs_MDF)
	model_numbers.append(k)


scores = np.array(scores)
MDFs = np.array(MDFs)
model_numbers = np.array(model_numbers)

sorted_scores = np.sort(scores)

outf = open('MDF_stats.dat', "w")
outf.write('#A21    tmax2    tau2    rmse\n')
idx = 0
for i in range(len(sorted_scores)):
	this_one = np.where(scores == sorted_scores[i] )[0]
	best_MDF = MDFs[this_one][0]
	best_model = model_numbers[this_one][0]

	best_x = myDict["x_model"+str(best_model)]
	best_y = myDict["y_model"+str(best_model)]

	model_params_string = myDict['model_'+str(best_model)+'_label'].strip()

	A21_ratio = model_params_string.split('A2/A1:')[1].split('_tmax')[0] ## dimensionless
	tmax2     = model_params_string.split('tmax_2:')[1].split('_tau2')[0] ## Gyr
	tau2      = model_params_string.split('_tau2:')[1]  ## also Gyr?

	if idx < 5:
		plt.plot(best_x, best_y, marker='o', color=m.to_rgba(float(best_model)), markersize=10, alpha=0.5,\
				 linestyle='', label='MDF, '+ model_params_string)
		plt.plot(resampled_x, best_MDF(resampled_x), color=m.to_rgba(float(best_model)), linestyle='-', label = 'cubic spline')
	
	write_str = "%.2f"%float(A21_ratio) +'   '+\
			    "%.2f"%float(tmax2) + '   ' +\
			    "%.2f"%float(tau2) + '   ' +\
			    "%.5f"%float(scores[this_one][0]) 

	outf.write(write_str+'\n')
	#print(model_params_string,'   rms=', scores[this_one][0])
	idx = idx + 1
outf.close()

# ## observations
plt.plot(feh, normalized_count, color='white', markeredgewidth='5', markeredgecolor='red',\
		 markersize=25, marker='^', linestyle='none', label=r'Johnson et al., $b=-6$')


plt.xlim(-2.5,0.7)
plt.xlabel("[Fe/H]", fontsize=25)
plt.ylabel("N stars", fontsize=25)

plt.legend(loc=2)
plt.show()
plt.close()


