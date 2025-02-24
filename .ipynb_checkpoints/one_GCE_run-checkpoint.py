#!/usr/bin/env python3.8
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import json
import re
import sys
sys.path.append('../')
from NuPyCEE import omega
sys.path.append('/home/mjoyce/bulge_isochrones/GCE_modeling/NuPyCEE/yield_tables/iniabu/')
from string import printable
from matplotlib.lines import *
from matplotlib.patches import *
from NuPyCEE import read_yields
from JINAPyCEE import omega_plus
from NuPyCEE import stellab
from NuPyCEE import sygma
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

sn1a_header = 'yield_tables/'
iniab_header = 'yield_tables/iniabu/'


inlist_name = 'inlist_GCE_one_infall_thin_disk'


parameter_assignments = []
inf = open(inlist_name,'r')
data = inf.readlines()
for line in data:
    if "#" in line:
        pass
    elif "=" in line:
        parameter_assignments.append(line)
    else:
        pass

data_dict = {}
for param in parameter_assignments:
    kwd = param.split('=')[0].strip()
    val = param.split('=')[1].strip()
    #print(kwd,val)
    data_dict[kwd] = val

print(data_dict)
#sys.exit()

###################################################
bulge_dict = {}

OMEGA_kwargs = {'special_timesteps': int(data_dict['timesteps']),
          'twoinfall_sigmas': [1300, float(data_dict['sigma_2'])],
          "galradius":float(data_dict['galradius']),
          'exp_infall':[  [-1, float(data_dict['tmax_1'])*1e9, float(data_dict['infall_1'])*1e9],\
                          [-1, float(data_dict['tmax_2'])*1e9, float(data_dict['infall_2'])*1e9]],
          'tauup': [0.02e9, 0.02e9],
          'mgal':float(data_dict['mgal']),
          'iniZ':0.0,
          "mass_loading":0.,
          "table":sn1a_header +data_dict['sy'],
          "sfe":float(data_dict['sfe_val']),
          "imf_type":data_dict['imf_val'],
          'sn1a_table':sn1a_header + data_dict['sn1a'],
          "imf_yields_range":[1,float(data_dict['imf_upper'])],
          "iniabu_table":iniab_header+data_dict['composition'],
          'nb_1a_per_m':float(data_dict['nb']),
          'sn1a_rate':data_dict['sn1a_form']
           }

GCE_model = omega_plus.omega_plus(**OMEGA_kwargs)
print('parameters accepted')    

m_gas_exp = np.zeros(GCE_model.inner.nb_timesteps+1)
m_locked = np.zeros(GCE_model.inner.nb_timesteps+1)

for i_t in range(GCE_model.inner.nb_timesteps+1):
    m_gas_exp[i_t] = sum(GCE_model.inner.ymgal[i_t])
    m_locked[i_t] = sum(GCE_model.inner.history.m_locked[0:i_t])

#print( (m_locked[-1]+m_gas_exp[-1])/1e10)
if m_locked[-1]+m_gas_exp[-1] < 3e10 and m_locked[-1]+m_gas_exp[-1] > 5e9:   
    
    mod_string = data_dict['model_name']

    # mod_string = "Model: "+'A2/A1:'+str(sigma_2/1e10)+'_tmax_2:'+str(t_2)+\
    # 'Gyr_tau2:'+str(infall_2)+'Gyr_FeH'+str(comp.split('iniab')[1].split('txt')[0] )
    print(mod_string)

    bulge_dict[mod_string]= GCE_model


print(len(bulge_dict), bulge_dict)
#sys.exit()

###############################################
#
# loading Bensby data for comparison
#
###############################################
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



#########################################################
#
# plotting results
#
#########################################################


#########################################################
#
# mass of bulge vs age -- targeting this
#
#########################################################
plt.figure(figsize=(15,12))

a=1 #len(sigma_2_list)
b=1 #len(sigma_2_list)*len(tmax_2_list)
c=1 #len(tmax_2_list)*len(infall_timescale_list)

colors=['b']*c+['r']*c+['k']*c+['cyan']*c
linestyles=['-']*a+['--']*a+[':']*a+['-']*a+['--']*a+[':']*a +\
           ['-']*a+['--']*a+[':']*a + ['-']*a+['--']*a+[':']*a
markers=['.','^','v','D']*b


j=0
for key,i in bulge_dict.items():
    # OMEGA+ time array
    t_op = np.array(i.inner.history.age)/1.0e9
    # Get the total mass of gas [Msun]
    m_gas_exp = np.zeros(i.inner.nb_timesteps+1)
    m_locked  = np.zeros(i.inner.nb_timesteps+1)
    m_in_gal  = np.zeros(i.inner.nb_timesteps+1)
   
    for i_t in range(i.inner.nb_timesteps+1):
        m_gas_exp[i_t] = sum(i.inner.ymgal[i_t])
        m_locked[i_t] = sum(i.inner.history.m_locked[0:i_t])
        m_in_gal[i_t] = sum(i.inner.history.m_tot_ISM_t)#[0:i_t])
        

    plt.plot(t_op, m_locked+m_gas_exp, 
#              color=colors[j], ls=linestyles[j], marker=markers[j],
             linewidth=5, label=key,
           )
    
    plt.plot(t_op, m_locked+m_gas_exp, 
#              color=colors[j], ls=linestyles[j], marker=markers[j],
             linewidth=1.5, label=key,
           )
    j+=1
    
plt.ylabel(r'Mass of Bulge ($M_\odot$)')
plt.xlabel('Age (Gyr)')    

plt.axhline(y=2e10, ls=':', lw=2, color='k')
# plt.legend()
#plt.savefig('Mass_age.png', bbox_inches='tight')
plt.show()
plt.close()



#########################################################
#
# age vs Fe/H
#
#########################################################
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

i=0
for key, val in bulge_dict.items():
    ax[0].plot(np.array(time[i])/1e9, 
               Fe[i],
#                colors[i], ls=linestyles[i], 
               linewidth=1.5, label=key)
    i+=1

# ax[0].plot([(13-4.6),(13-4.6)], [-10, 10], ':', color='cornflowerblue', alpha=0.7)
ax[0].plot([(13-4.6),(13-4.6)], [-10, 10], ':', color='cornflowerblue', alpha=0.7)
ax[0].plot([-1e10,1e10], [0,0], ':', color='cornflowerblue', alpha=0.7)
ax[0].set_xlim(-0.3,12.5)
ax[0].set_ylim(-1., 2.0)

i=0
for key, val in bulge_dict.items():
    ax[1].plot(np.array(val.inner.history.age)/1e9, val.inner.history.metallicity, 
#                colors[i], ls=linestyles[i],
               linewidth=1.5)
    i+=1
           
# ax[1].plot([(13-4.6),(13-4.6)], [-10, 10], ':', color='cornflowerblue', alpha=0.7)
ax[1].plot([(13-4.6),(13-4.6)], [-10, 10], ':', color='cornflowerblue', alpha=0.7)
ax[1].plot([-1e10,1e10], [0.014,0.014], ':', color='cornflowerblue', alpha=0.7)
ax[1].set_yscale('log')
ax[1].set_xlim(-0.3,12.5)
ax[1].set_ylim(1e-3, 4e-1)
ax[1].scatter([8.4],[0.014],color='k',zorder=10)

# Labels and visual aspect
# ax[0].legend(frameon=False, loc='center left', bbox_to_anchor=(1,0.8))
ax[0].set_ylabel('[Fe/H]', fontsize=al_f)
ax[1].set_ylabel('Z (mass fraction)', fontsize=al_f)
ax[1].set_xlabel('Galactic age [Gyr]', fontsize=al_f)

# Adjust layout/white spaces
plt.subplots_adjust(top=0.95)
plt.subplots_adjust(bottom=0.2)
plt.subplots_adjust(left=0.2)
plt.subplots_adjust(right=0.97)

#plt.savefig('calibration2.png',bbox_inches='tight')
plt.show()
plt.close()



#########################################################
#
# SFR vs age
#
#########################################################
plt.figure(figsize=(15,12))

i=0
for key,val in bulge_dict.items():
    plt.plot(val.inner.history.age/1e9, val.inner.history.sfr_abs, 
#              colors[i],ls=linestyles[i], marker=markers[i],
             label=key, 
             linewidth=1.5,)
    i+=1
    
plt.ylabel(r'SFR $[M_{\odot}\;\rm{yr}^{-1}]$')
plt.xlabel('Age (Gyr)') 
#plt.legend(loc='upper right')
plt.savefig('SFR_age.png', bbox_inches='tight')
plt.xlim(-0.2,12)
#plt.savefig('SFR.png', bbox_inches='tight')
plt.show()
plt.close()


#########################################################
#
# MDF
#
#########################################################
plt.figure(figsize=(18,12))
j=0

text_file = open('MDF_data.txt', 'w')
    
# read in photometric data
n,x,_=plt.hist(Fe_H,bins=25, density=True, histtype=u'step', color='magenta', lw=2,zorder=10, stacked=True)
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
#              color=colors[j], ls=linestyles[j], marker=markers[j], 
             alpha=0.5, zorder=1)
    j+=1


    
plt.xlim(-5,1)
plt.tick_params(axis='both', direction='in', length=5)
plt.xlabel(r'$\mathrm{[Fe/H]}$')
plt.ylabel("Number density" )
plt.ylim(0,1.5)
# plt.savefig('MDF.png', bbox_inches='tight')
text_file.close()
plt.close()

#########################################################
#
# Age vs [Fe/H]; Bensby vs Joyce
#
#########################################################
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
    
plt.scatter(Fe_H, age_Joyce, marker='*', s=10, color='blue', label='Joyce et al.')
plt.scatter(Fe_H,age_Bensby,  marker='^', s=10, color='orange', label='Bensby et al.')
plt.xlabel('[Fe/H]')
plt.ylabel('Age (Gyr)')
plt.xlim(-2,1)
#plt.gca().invert_yaxis()
plt.legend(loc=3)
#plt.savefig('Fe_H_vs_age.png', bbox_inches='tight')
plt.show()
text_file.close()
plt.close()


#########################################################
#
# evolution of individual abundances
#
#########################################################
al_f = 14 
matplotlib.rcParams.update({'font.size': 16.})
f, axarr = plt.subplots(2, 2, figsize=(12,8), sharex=True, sharey=True) # [row][col]
f.subplots_adjust(hspace=0.)
f.subplots_adjust(wspace=0.)

axis_dict={}

axis_dict['[Mg/Fe]']='[Fe/H]'
axis_dict['[Si/Fe]']='[Fe/H]'
axis_dict['[Ca/Fe]']='[Fe/H]'
axis_dict['[Ti/Fe]']='[Fe/H]'

l=0
j=0
for yaxis, xaxis in axis_dict.items():
    i=0
    for key,val in bulge_dict.items():
        x,y=val.inner.plot_spectro(xaxis=xaxis, yaxis=yaxis, return_x_y=True, 
                                  )

        axarr[j][l].plot(x, y, 
#                          color=colors[i], ls=linestyles[i], 
                 alpha=1, linewidth=2, label=key, zorder=1)
        txt_lab=yaxis.split('/')[0]
        txt_lab=txt_lab.split('[')[1]
        axarr[j][l].text(-1,0.75,txt_lab,backgroundcolor='white',zorder=11,ha='center')
        axarr[j][l].xaxis.set_minor_locator(MultipleLocator(0.2))
        axarr[j][l].yaxis.set_minor_locator(MultipleLocator(0.2))
        axarr[j][l].tick_params(top=True, right=True, direction='in', length=6)
        axarr[j][l].tick_params(which='minor',right=True, direction='in', length=4)
        

        i+=1
    if l<1:
        l+=1
    elif l == 1:
        l=0
        j+=1
        
plt.ylim(-0.8,1)
plt.xlim(-2,1)
# Observational data
axarr[0][0].scatter(Fe_H, Mg_Fe, marker='*', color='blue', )
axarr[1][0].scatter(Fe_H, Ca_Fe, marker='*', color='blue', )
axarr[0][1].scatter(Fe_H, Si_Fe, marker='*', color='blue', )
axarr[1][1].scatter(Fe_H, Ti_Fe, marker='*', color='blue', )

f.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel("[Fe/H]")
plt.ylabel("[X/Fe]")
#plt.savefig('Fe_H_vs_X_Fe.png', bbox_inches='tight')
plt.show()
plt.close()





