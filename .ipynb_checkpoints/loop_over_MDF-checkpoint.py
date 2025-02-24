#!/usr/bin/env python3.8
################################
#
# Author: M Joyce
#
################################
import matplotlib
#import imp
import matplotlib.pyplot as plt
import numpy as np
import re

import sys
#from sklearn import preprocessing
sys.path.append('../')
from NuPyCEE import omega

from string import printable

from matplotlib.lines import *
from matplotlib.patches import *
from NuPyCEE import read_yields
from JINAPyCEE import omega_plus
from NuPyCEE import stellab
from NuPyCEE import sygma
from matplotlib.ticker import AutoMinorLocator, MultipleLocator


f = open('Bensby_Data.tsv')
lines = f.readlines()
Fe_H = []
Fe_H_err = []
age_Joyce=[]
age_Bensby=[]
Si_Fe =[]
Si_Fe_err =[]
Ca_Fe = []
Ca_Fe_err =[]
Mg_Fe = []
Mg_Fe_err =[]
Ti_Fe =[]
Ti_Fe_err =[]

for line in lines[1::]:
    line=line.split()
    
    Fe_H_ind = lines[0].split().index('[Fe/H]')
    Fe_H_err_ind = lines[0].split().index('error_[Fe/H]')
    
    Si_Fe_ind = lines[0].split().index('[Si/Fe]')
    Si_Fe_err_ind = lines[0].split().index('error_[Si/Fe]')
    
    Ca_Fe_ind = lines[0].split().index('[Ca/Fe]')
    Ca_Fe_err_ind = lines[0].split().index('error_[Ca/Fe]')
    
    Mg_Fe_ind = lines[0].split().index('[Mg/Fe]')
    Mg_Fe_err_ind = lines[0].split().index('error_[Mg/Fe]')
    
    Ti_Fe_ind = lines[0].split().index('[Ti/Fe]')
    Ti_Fe_err_ind = lines[0].split().index('error_[Ti/Fe]')
    
    age_Joyce_ind = lines[0].split().index('Joyce_age')
    age_Bensby_ind = lines[0].split().index('Bensby')
    
    age_Joyce.append(float(line[age_Joyce_ind]))
    age_Bensby.append(float(line[age_Bensby_ind]))
    Fe_H.append(float(line[Fe_H_ind]))
    Fe_H_err.append(float(line[Fe_H_err_ind]))
    Si_Fe.append(float(line[Si_Fe_ind]))
    Si_Fe_err.append(float(line[Si_Fe_err_ind]))
    Ca_Fe.append(float(line[Ca_Fe_ind]))
    Ca_Fe_err.append(float(line[Ca_Fe_err_ind]))
    Mg_Fe.append(float(line[Mg_Fe_ind]))
    Mg_Fe_err.append(float(line[Mg_Fe_err_ind]))
    Ti_Fe.append(float(line[Ti_Fe_ind]))
    Ti_Fe_err.append(float(line[Ti_Fe_err_ind]))


# Launch the STELLAB code
s = stellab.stellab()

# Select the galaxy
galaxy = "milky_way"


# ## Create function to make inflow rates that aren't in parallel

def two_inflow_fn(t, exp_inflow):
    if t < exp_inflow[1][1]:
        inflow_rate = (exp_inflow[0][0]*np.exp(-t/exp_inflow[0][2]))
    else:
        inflow_rate = (exp_inflow[0][0]*np.exp(-t/exp_inflow[0][2])+
                           exp_inflow[1][0]*np.exp(-(t-exp_inflow[1][1])/exp_inflow[1][2]))
    return inflow_rate


sys.path.append('/home/mjoyce/bulge_isochrones/GCE_modeling/yield_tables/iniabu/')

iniab_header = 'yield_tables/iniabu/'
bulge_dict={}

A2=-1

###################################
#
# Joyce 10/8/24
#
# loop: change from [1e9] --> array
# sweep over tmax_2
#
###################################
sigma_2_list=[1e9]#,0,1.0] #,0.4,0.6,0.8,1.0,1.2,1.4]   ## setting the ratio to be outrageous in order to suppress the first infall entirely


#for t_2 in tmax_2_list:
## default thick disk assumption is 11 Gyr ago (+/- 3 Gyr) a bunch of stuff fell in at [M/H] = -0.5
## default thin disk assumption is 8 Gyr ago (+/- 5 Gyr) long, slow inflow at solar


###################################
#
# Joyce 10/8/24
#
# loop: change from [5] --> array
# sweep over tmax_2
#
###################################
#tmax_2_list=[5] ## Gyr since age of universe: so 2 means 11 Gyr ago (thick disk argument); 5 means 8 (thin disk argument)  
tmax_2_list= [1]#,5,10]#np.arange(1,10,1) ## in Gyr  #,5,10]#,6,7,8,9,10,11,12]

## if the bulge is being continuously fed by the disk, over all time, this should be  10 Gyr
## default thick disk assumption: 3 Gyr, coming in at [Fe/H] -0.5  if getting fed by thick disk rather than thin disk
## default thin disk: 5
###################################
#
# Joyce 10/8/24
#
# loop: change from [5] --> array
# sweep over infall_2
#
###################################
infall_timescale_list=[0.5, 5, 10] #np.arange(0.5, 10, 0.5) #[0.5]

#'exp_infall':[[-1, 0.1e9, 0.1e9], [A2, t_2*1e9, infall_2*1e9]],

## infall metallicity specified with Christian's files
## [Fe/H] = -0.5; [alpha/Fe] = +0.3 for thick disk
comp_array = ['iniab_output_feh_p000.txt'] #['iniab_output_feh_m050.txt'] 


#['iniab_FeH-1.5_GS98.txt','iniab_solar_GN93.txt','iniab_FeH-1.5_GS98.txt','iniab_solar_GN93.txt']
              #'yield_tables/iniabu/iniab_FeH+0.0_GS98.txt']

## star formation efficiency  -- nu in the manuscript -- how much of gas do you turn into stars -- fudge factor 
## pushing this up will enrich gas faster (probably) -- do want to vary this
sfe_array = [0.02] #np.arange(0.01, 0.03, 0.005) #[0.005, 0.01, 0.02, 0.03] # range 0.005, 0.04

imf_array = ['kroupa']#, 'salpeter', 'chabrier', 'lognormal' ]    
imf_upper_limits = [50] #np.arange(30,100,10) 

## add this in when I feel like it
## huge source of uncertainty!!
sn1a_header = 'yield_tables/'
sn1a_assumptions = ['sn1a_Gronow.txt'] #,\
#                     'sn1a_ivo12_stable_z.txt',\
#                     'sn1a_Leung2018_benchmark.txt',\
#                     'sn1a_shen.txt',\
#                     'sn1a_townsley.txt']


## total mass (?)
mgal_values = [2e9] #np.arange(1e9,2e10, 1e9) #[1e9, 2.5e9, ] # bounded by 3e10 strictly -- doesn't get added to dict above this; below ~1e8 effects are minimal

stellar_yield_assumptions = ['agb_and_massive_stars_K10_LC18_Ravg.txt'] #,\
#                              'agb_and_massive_stars_C15_LC18_R_mix.txt',\
#                              'agb_and_massive_stars_C15_N13_0_0_HNe.txt',\
#                              'agb_and_massive_stars_nugrid_K10.txt']

## Tom thinks this should be varied since it's poorly constrained
nb_array = [1e-3]#, 1e-4, 1e-2] #[1e-4, 5e-3, 1e-3, 5e-2, 1e-2] #default 1e-3; bound by factor of 100 on either side
sn1a_rates =['power_law'] #,'gauss','exp','maoz']



timesteps = 500 #100 #500    


models = 0

for sigma_2 in sigma_2_list: 
    for t_2 in tmax_2_list:
        for infall_2 in infall_timescale_list:
            
            for comp in comp_array: 
                for imf_val in imf_array:
                    #print(comp)
                    for sfe_val in sfe_array:
                        for imf_upper in imf_upper_limits:
                            for sn1a in sn1a_assumptions:
                                for sy in stellar_yield_assumptions:
                                    for mgal in mgal_values:
                                        for nb in nb_array:
                                            for sn1ar in sn1a_rates:
                                                blanka_kwargs = {'special_timesteps':timesteps,
                                                          'twoinfall_sigmas': [1300, sigma_2],
                                                          "galradius":1800,
                                                          'exp_infall':[[-1, 0.1e9, 0.1e9], [A2, t_2*1e9, infall_2*1e9]],
                                                          'tauup': [0.02e9, 0.02e9],
                                                          'mgal':mgal,
                                                          'iniZ':0.0,
                                                          "mass_loading":0.,
                                                          "table":sn1a_header + sy,
                                                          "sfe":sfe_val,
                                                          "imf_type":imf_val,
                                                          'sn1a_table':sn1a_header + sn1a,
                                                          "imf_yields_range":[1,imf_upper],
                                                          "iniabu_table":iniab_header+comp,
                                                          'nb_1a_per_m':nb,
                                                          'sn1a_rate':sn1ar
                                                            }

                                                GCE_model = omega_plus.omega_plus(**blanka_kwargs)
                                                print('parameters accepted')    

                                                m_gas_exp = np.zeros(GCE_model.inner.nb_timesteps+1)
                                                m_locked = np.zeros(GCE_model.inner.nb_timesteps+1)
                                
                                                for i_t in range(GCE_model.inner.nb_timesteps+1):
                                                    m_gas_exp[i_t] = sum(GCE_model.inner.ymgal[i_t])
                                                    m_locked[i_t] = sum(GCE_model.inner.history.m_locked[0:i_t])
                                
                                                #print( (m_locked[-1]+m_gas_exp[-1])/1e10)
                                                if m_locked[-1]+m_gas_exp[-1] < 3e10 and m_locked[-1]+m_gas_exp[-1] > 5e9:   
                                                    mod_string = "Model: "+'A2/A1:'+str(sigma_2/1e10)+'_tmax2:'+str(t_2)+\
                                                    'Gyr_tau2:'+str(infall_2)+'Gyr_FeH'+str(comp.split('_feh_')[1].split('txt')[0] )
                                                   # print(mod_string)
                                                    bulge_dict[mod_string.split('Model: ')[1]]= GCE_model

                                                    #'iniab_output_feh_p000.txt'
                                               
                                                models = models + 1
print('n models: ', models)
print('len(bulge_dict), bulge_dict: ', len(bulge_dict), bulge_dict)
#sys.exit()

plt.figure(figsize=(15,12))

a=len(sigma_2_list)
b=len(sigma_2_list)*len(tmax_2_list)
c=len(tmax_2_list)*len(infall_timescale_list)

colors=['b']*c+['r']*c+['k']*c+['cyan']*c
linestyles=['-']*a+['--']*a+[':']*a+['-']*a+['--']*a+[':']*a + ['-']*a+['--']*a+[':']*a + ['-']*a+['--']*a+[':']*a
markers=['o','^','v','D']*b


plt.figure(figsize=(18,12))
j=0

## pink is the observed metallicity distribution in the bulge
text_file = open('MDF_data.txt', 'w')
    
for key, val in bulge_dict.items():
    text_file.write('Model: '+str(key)+'\n')
    x_gce, y_gce = val.inner.plot_mdf(axis_mdf='[Fe/H]',sigma_gauss=0.1, norm=True, return_x_y=True)#,\
                                     #solar_ab='yield_tables/iniabu/iniab_6.0E-03GN93.txt')
    #print("y_gce: ", y_gce, x_gce)
    
    for x_i, x in enumerate(x_gce):
        temp_string=(x,y_gce[x_i])
#         print(temp_string)
        text_file.write('{0: <8} {1: <8}'.format(*temp_string)+'\n')
    plt.plot(x_gce,y_gce, label=key, 
              color=colors[j], ls=linestyles[j], marker=markers[j],markersize=10, 
             alpha=0.5, zorder=1)
    j+=1

    
plt.xlim(-2, 1) #(-5,1)
plt.tick_params(axis='both', direction='in', length=10)
plt.xlabel(r'$\mathrm{[Fe/H]}$', fontsize=35)
plt.ylabel("Number density", fontsize=35)
plt.ylim(0, 1.05) #(0,1.5)
plt.savefig('MDF_multiple.png')#, bbox_inches='tight')

text_file.close()
plt.close()
