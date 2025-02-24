#!/usr/bin/env python3.8
################################
#
# Author: M Joyce
#
################################
import matplotlib
import imp
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
sigma_2_list=[1e9]    #,0.4,0.6,0.8,1.0,1.2,1.4]   ## setting the ratio to be outrageous in order to suppress the first infall entirely


#for t_2 in tmax_2_list:
## default thick disk assumption is 11 Gyr ago (+/- 3 Gyr) a bunch of stuff fell in at [M/H] = -0.5
## default thin disk assumption is 8 Gyr ago (+/- 5 Gyr) long, slow inflow at solar
tmax_2_list=[5] ## Gyr since age of universe: so 2 means 11 Gyr ago (thick disk argument); 5 means 8 (thin disk argument)  
#tmax_2_list= np.arange(1,10,1) ## in Gyr  #,5,10]#,6,7,8,9,10,11,12]

## if the bulge is being continuously fed by the disk, over all time, this should be  10 Gyr
## default thick disk assumption: 3 Gyr, coming in at [Fe/H] -0.5  if getting fed by thick disk rather than thin disk
## default thin disk: 5
infall_timescale_list=[5] #np.arange(0.5, 10, 0.5) #[0.5] #0.5,1,2,4]

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
                                                    mod_string = "Model: "+'A2/A1:'+str(sigma_2/1e10)+'_tmax_2:'+str(t_2)+\
                                                    'Gyr_tau2:'+str(infall_2)+'Gyr_FeH'+str(comp.split('iniab')[1].split('txt')[0] )
                                                   # print(mod_string)
                                                    bulge_dict[mod_string.split('Model: ')[1]]= GCE_model
                                               
                                                models = models + 1
print('n models: ', models)
print(len(bulge_dict), bulge_dict)


plt.figure(figsize=(15,12))

a=len(sigma_2_list)
b=len(sigma_2_list)*len(tmax_2_list)
c=len(tmax_2_list)*len(infall_timescale_list)

colors=['b']*c+['r']*c+['k']*c+['cyan']*c
linestyles=['-']*a+['--']*a+[':']*a+['-']*a+['--']*a+[':']*a + ['-']*a+['--']*a+[':']*a + ['-']*a+['--']*a+[':']*a
markers=['o','^','v','D']*b


# j=0
# for key,i in bulge_dict.items():
#     # OMEGA+ time array
#     t_op = np.array(i.inner.history.age)/1.0e9
#     # Get the total mass of gas [Msun]
#     m_gas_exp = np.zeros(i.inner.nb_timesteps+1)
#     m_locked  = np.zeros(i.inner.nb_timesteps+1)
#     m_in_gal  = np.zeros(i.inner.nb_timesteps+1)
   
#     for i_t in range(i.inner.nb_timesteps+1):
#         m_gas_exp[i_t] = sum(i.inner.ymgal[i_t])
#         m_locked[i_t] = sum(i.inner.history.m_locked[0:i_t])
#         m_in_gal[i_t] = sum(i.inner.history.m_tot_ISM_t)#[0:i_t])
        
        
# #     plt.plot(t_op, m_locked, 'g', ls=ls[j], linewidth=1.5)
# #     plt.plot(t_op, m_gas_exp, 'b', ls=ls[j], linewidth=1.5)
#     #print("m_gas_exp: ",m_gas_exp)
#     #print("m_locked array: ",m_locked)
#     #print("m_in_gal array: ",m_in_gal)
    
#     plt.plot(t_op, m_locked+m_gas_exp, 
# #              color=colors[j], ls=linestyles[j], marker=markers[j],
#              linewidth=5, label=key,
#            )
    
#     plt.plot(t_op, m_locked+m_gas_exp, 
# #              color=colors[j], ls=linestyles[j], marker=markers[j],
#              linewidth=1.5, label=key,
#            )
#     j+=1
    
# plt.ylabel(r'Mass of Bulge ($M_\odot$)')
# plt.xlabel('Age (Gyr)')    

# plt.axhline(y=2e10, ls=':', lw=2, color='k')
# # plt.legend()
# plt.savefig('Mass_age.png', bbox_inches='tight')
# ("")


# In[9]:


plt.close()
#get_ipython().run_line_magic('matplotlib', 'inline')
# Plot the age-metallicity relations
al_f = 14 # text font
matplotlib.rcParams.update({'font.size': 14.})
f, ax = plt.subplots(2,1, figsize=(5,7), sharex=True)
f.subplots_adjust(hspace=0)


def plot_spectr(omega_run, solar_norm):
    return(omega_run.inner.plot_spectro(solar_norm=solar_norm, return_x_y=True))

time=[]
Fe=[]
# Extract predictions
for key, val in bulge_dict.items():
    time.append(plot_spectr(val, 'Lodders_et_al_2009')[0])
    Fe.append(plot_spectr(val, 'Lodders_et_al_2009')[1])

# # Extract predictions
# o_t_def, o_Fe_H_def = op_dean_power.inner.plot_spectro(solar_norm='Lodders_et_al_2009', return_x_y=True)
# o_t_low, o_Fe_H_low = op_brox_power.inner.plot_spectro(solar_norm='Lodders_et_al_2009', return_x_y=True)
# o_t_med, o_Fe_H_med = op_shen_power.inner.plot_spectro(solar_norm='Lodders_et_al_2009', return_x_y=True)
# o_t_hi, o_Fe_H_hi = op_default.inner.plot_spectro(solar_norm='Lodders_et_al_2009', return_x_y=True)



# Plot age-[Fe/H]
# ===============
# ax[0].plot(13.-(np.array(t)), FeH, 'o', color='cornflowerblue', markersize=3, alpha=0.7)

# i=0
# for key, val in bulge_dict.items():
#     ax[0].plot(np.array(time[i])/1e9, 
#                Fe[i],
# #                colors[i], ls=linestyles[i], 
#                linewidth=1.5, label=key)
#     i+=1

# # ax[0].plot([(13-4.6),(13-4.6)], [-10, 10], ':', color='cornflowerblue', alpha=0.7)
# ax[0].plot([(13-4.6),(13-4.6)], [-10, 10], ':', color='cornflowerblue', alpha=0.7)
# ax[0].plot([-1e10,1e10], [0,0], ':', color='cornflowerblue', alpha=0.7)
# ax[0].set_xlim(-0.3,12.5)
# ax[0].set_ylim(-1., 2.0)

# i=0
# for key, val in bulge_dict.items():
#     ax[1].plot(np.array(val.inner.history.age)/1e9, val.inner.history.metallicity, color='green', marker='o', 
# #                colors[i], ls=linestyles[i],
#                linewidth=1.5)
#     i+=1
           
# # ax[1].plot([(13-4.6),(13-4.6)], [-10, 10], ':', color='cornflowerblue', alpha=0.7)
# ax[1].plot([(13-4.6),(13-4.6)], [-10, 10], ':', color='cornflowerblue', alpha=0.7)
# ax[1].plot([-1e10,1e10], [0.014,0.014], ':', color='cornflowerblue', alpha=0.7)
# ax[1].set_yscale('log')
# ax[1].set_xlim(-0.3,12.5)
# ax[1].set_ylim(1e-3, 4e-1)

# ## questions for Tom -- what is this point and why? IT IS THE SUN -- JAMIE
# ax[1].plot([8.4],[0.014],color='gold', marker='*', markersize=40, alpha=0.3,zorder=10)

# # Labels and visual aspect
# # ax[0].legend(frameon=False, loc='center left', bbox_to_anchor=(1,0.8))
# ax[0].set_ylabel('[Fe/H]', fontsize=al_f)
# ax[1].set_ylabel('Z (mass fraction)', fontsize=al_f)
# ax[1].set_xlabel('Galactic age [Gyr]', fontsize=al_f)

# ## put my age-[Fe/H] relation on here

# # Adjust layout/white spaces
# plt.subplots_adjust(top=0.95)
# plt.subplots_adjust(bottom=0.2)
# plt.subplots_adjust(left=0.2)
# plt.subplots_adjust(right=0.97)

# plt.savefig('calibration2.png',bbox_inches='tight')
# plt.close()


# plt.figure(figsize=(15,12))

# i=0
# for key,val in bulge_dict.items():
#     plt.plot(val.inner.history.age/1e9, np.log10(val.inner.history.sfr_abs), 
# #              colors[i],ls=linestyles[i], marker=markers[i],
#              label=key, 
#              linewidth=1.5,)
#     i+=1
    
# plt.ylabel(r'SFR $[M_{\odot}\;\rm{yr}^{-1}]$')
# plt.xlabel('Age (Gyr)') 
# #plt.legend(loc='upper right')
# plt.savefig('SFR_age.png', bbox_inches='tight')
# plt.xlim(-0.2,12)
# plt.savefig('SFR.png', bbox_inches='tight')

# plt.close()



plt.figure(figsize=(18,12))
j=0

## pink is the observed metallicity distribution in the bulge
text_file = open('MDF_data.txt', 'w')
    
# read in photometric data
#bins = 25
# n,x,_=plt.hist(Fe_H,  bins=25, density=True, histtype=u'step', color='magenta', lw=2,zorder=10, stacked=True)
# n,x,_=plt.hist(Fe_H,bins=10, density=True, histtype=u'step', color='k', lw=2, ls='-')

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


# custom_patches = [Patch(edgecolor='k',facecolor='None', lw=2),
#               Line2D([0], [0], color="b", lw=2, ls=':'),
#               Line2D([0], [0], color="orange", lw=2, ls='--'),
#               Line2D([0], [0], color="green", lw=2, ls='-.'),
#               Line2D([0], [0], color="k", lw=2, ls=':'),
#               Line2D([0], [0], color="b", lw=2, ls='--')]
   
# plt.legend(custom_patches, ['Bensby data']+list([key for key in bulge_dict.keys()]), loc='upper left')
# plt.legend(loc='upper left')
    
plt.xlim(-2.5, 0.5) #(-5,1)
plt.tick_params(axis='both', direction='in', length=10)
plt.xlabel(r'$\mathrm{[Fe/H]}$', fontsize=35)
plt.ylabel("Number density", fontsize=35)
plt.ylim(0, 1.05) #(0,1.5)
plt.savefig('MDF.png')#, bbox_inches='tight')

text_file.close()
plt.close()


plt.figure(figsize=(15,10))
xy_dict={}
i=0
xaxis='age'
yaxis='[Fe/H]'
text_file = open('FeH_age_data.txt', 'w')
for key,val in bulge_dict.items():
    text_file.write('Model: '+str(key)+'\n')
    x_age=[]
    x_gce,y_gce=val.inner.plot_spectro(xaxis=xaxis, yaxis=yaxis, return_x_y=True)
    for x_i,x in enumerate(x_gce):
        temp_string=((x_gce[-1]/1e9)-(x/1e9),y_gce[x_i])
        text_file.write('{0: <8} {1: <8}'.format(*temp_string)+'\n')
    plt.plot(y_gce,(x_gce[-1]/1e9)-np.array(x_gce)/1e9, 
#              color=colors[i], ls=linestyles[i], 
         alpha=1, linewidth=2, label=key, zorder=10)

#     i+=1
    
plt.scatter(Fe_H, age_Joyce, marker='*', s=150, color='blue', label='Joyce et al.')
plt.scatter(Fe_H,age_Bensby,  marker='^', s=150, color='orange', label='Bensby et al.')
plt.xlabel('[Fe/H]')
plt.ylabel('Age (Gyr)')
plt.xlim(-2,1)
#plt.gca().invert_yaxis()
plt.legend(loc=3)
plt.savefig('FeH_vs_age.png', bbox_inches='tight')
text_file.close()
plt.close()


# Figure frame
#get_ipython().run_line_magic('matplotlib', 'inline')
# al_f = 14 # text font
# matplotlib.rcParams.update({'font.size': 16.})
# f, axarr = plt.subplots(2, 2, figsize=(12,8), sharex=True, sharey=True) # [row][col]
# f.subplots_adjust(hspace=0.)
# f.subplots_adjust(wspace=0.)

# axis_dict={}

# axis_dict['[Mg/Fe]']='[Fe/H]'
# axis_dict['[Si/Fe]']='[Fe/H]'
# axis_dict['[Ca/Fe]']='[Fe/H]'
# axis_dict['[Ti/Fe]']='[Fe/H]'

# l=0
# j=0
# for yaxis, xaxis in axis_dict.items():
#     i=0
#     for key,val in bulge_dict.items():
#         x,y=val.inner.plot_spectro(xaxis=xaxis, yaxis=yaxis, return_x_y=True, 
#                                   )

#         axarr[j][l].plot(x, y, 
# #                          color=colors[i], ls=linestyles[i], 
#                  alpha=1, linewidth=2, label=key, zorder=1)
#         txt_lab=yaxis.split('/')[0]
#         txt_lab=txt_lab.split('[')[1]
#         axarr[j][l].text(-1,0.75,txt_lab,backgroundcolor='white',zorder=11,ha='center')
#         axarr[j][l].xaxis.set_minor_locator(MultipleLocator(0.2))
#         axarr[j][l].yaxis.set_minor_locator(MultipleLocator(0.2))
#         axarr[j][l].tick_params(top=True, right=True, direction='in', length=6)
#         axarr[j][l].tick_params(which='minor',right=True, direction='in', length=4)
        

#         i+=1
#     if l<1:
#         l+=1
#     elif l == 1:
#         l=0
#         j+=1
        
# plt.ylim(-0.8,1)
# plt.xlim(-2,1)
# # Observational data
# axarr[0][0].scatter(Fe_H, Mg_Fe, marker='*', color='blue', )
# axarr[1][0].scatter(Fe_H, Ca_Fe, marker='*', color='blue', )
# axarr[0][1].scatter(Fe_H, Si_Fe, marker='*', color='blue', )
# axarr[1][1].scatter(Fe_H, Ti_Fe, marker='*', color='blue', )

# # axarr[0][1].legend(loc=(1.05,0.4))
# f.add_subplot(111, frameon=False)
# plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
# plt.xlabel("[Fe/H]")
# plt.ylabel("[X/Fe]")
# plt.savefig('Fe_H_vs_X_Fe.png', bbox_inches='tight')
