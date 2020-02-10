#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 12:20:31 2018

@author: fiercenatora
"""

from scipy.interpolate import interp2d
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm    
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import PyMieScatt
from matplotlib.collections import PolyCollection
from matplotlib import cm
import matplotlib as mpl
from helper import *
#from Bayes_model_fitting import *
#from gaussian_kde_weighted import *
from scipy.stats import gaussian_kde
# =============================================================================
# LOAD TOOLBOX
# =============================================================================

wavelength = 532*1e-9

h = 0.03
confidence_interval = [0.25,0.75]

Rbc_grid = np.linspace(0,20,100)
frac_less_than_grid = np.linspace(0,1,1001)
Eabs_grid = np.linspace(0,4,200)


col_lab_part = 'C0' 
col_cs_part = [230/255,184/255,0/255]
col_lab_uniform = 'C3'
col_cs_uniform = 'k'

####################################################################################
#
#  Read in Fontana data --> wavelength = 532 nm
#
####################################################################################
    
observational_dat = np.genfromtxt('data/measurements/FontanaData-090817.txt',delimiter='\t',skip_header=1);

from datetime import datetime
july4_2015 = (datetime(2015, 7, 4, 0, 0)-datetime(1904, 1, 1, 0, 0)).total_seconds()
july7_2015 = (datetime(2015, 7, 7, 0, 0)-datetime(1904, 1, 1, 0, 0)).total_seconds()

these, = np.nonzero((sum(np.vstack([observational_dat[:,1]<july4_2015, observational_dat[:,2]>july7_2015]),0)>0) & 
                    (observational_dat[:,15]>0)&~np.isnan(observational_dat[:,6]))

BC_observed = observational_dat[these,7]
Rbc_observed = observational_dat[these,15]
if wavelength == 405*1e-9:
    MAC_observed = observational_dat[these,3]
    Eabs_observed = observational_dat[these,5]    
else:
    MAC_observed = observational_dat[these,4]
    Eabs_observed = observational_dat[these,6]

from scipy.signal import savgol_filter
from scipy.signal import spline_filter
ci_vals = [0.01,0.05,0.25]

Eabs_ci_observed = np.zeros([len(Rbc_grid),len(ci_vals),2])
Eabs_grid_observed = np.zeros([len(Rbc_grid),len(frac_less_than_grid)])
dNdE_observed =  np.zeros([len(Rbc_grid),len(Eabs_grid)])

thresh_std_low = 4
thresh_std_high = 10

dRbc = Rbc_grid[1] - Rbc_grid[0]
for rr in range(len(Rbc_grid)):
    these = sum(np.vstack([Rbc_observed>=(Rbc_grid[rr]-dRbc/2),Rbc_observed<(Rbc_grid[rr]+dRbc/2)]))>1
    if sum(these)>0:
        idx = np.argsort(Eabs_observed[these])
        these_wi = BC_observed[these]
        these_Ei = Eabs_observed[these]
        idx = np.argsort(these_Ei)
        frac_less_than = np.cumsum(these_wi[idx])/sum(these_wi)
        Eabs_grid_observed[rr,:] = np.interp(frac_less_than_grid,frac_less_than,these_Ei[idx])
        for cc in range(len(ci_vals)):
            confidence_interval = [ci_vals[cc],1-ci_vals[cc]]
            for ii in range(len(confidence_interval)):
                these = abs(frac_less_than_grid - confidence_interval[ii])==min(abs(frac_less_than_grid - confidence_interval[ii]))
                Eabs_ci_observed[rr,cc,ii] = Eabs_grid_observed[rr,these]
        frac_interp = np.interp(Eabs_grid,Eabs_grid_observed[rr,:],frac_less_than_grid)
        dNdE_observed[rr,:] = savgol_filter(frac_interp,21,3,deriv=1)
        dNdE_observed[rr,dNdE_observed[rr,:]<0] = 0        
    else:
        Eabs_grid_observed[rr,:] = float('nan')
        Eabs_ci_observed[rr,:,:] = float('nan')
p_EgivenR_observed = savgol_filter(dNdE_observed,21,3,axis=0)


####################################################################################
#
#   Read in PartMC data for Bayes factor --> wavelength = 532 nm
#
####################################################################################
tt_max = 218
wavelength = 532*1e-9

all_tt_i = np.loadtxt('data/partmc_std_dev/all_tt_i.txt')
all_rr_i = np.loadtxt('data/partmc_std_dev/all_rr_i.txt')

all_Ri = np.loadtxt('data/partmc_bulk/all_Ri.txt')
all_wi = np.loadtxt('data/partmc_bulk/all_wi.txt')

all_Ei = np.loadtxt('data/partmc_bulk/all_Ei.txt')
dNdE_lab =  np.zeros([len(Rbc_grid),len(Eabs_grid)])
for rr in range(len(Rbc_grid)):
    these = sum(np.vstack(
            [all_Ri>=(Rbc_grid[rr]-dRbc/2),
             all_Ri<(Rbc_grid[rr]+dRbc/2),
             all_tt_i<tt_max]))>2
    if sum(these)>0:
        idx = np.argsort(all_Ei[these])
        these_wi = all_wi[these]
        these_Ei = all_Ei[these]
        idx = np.argsort(these_Ei)
        frac_less_than = np.cumsum(these_wi[idx])/sum(these_wi)
        Eabs_grid_lab = np.interp(frac_less_than_grid,frac_less_than,these_Ei[idx])
        frac_interp = np.interp(Eabs_grid,Eabs_grid_lab,frac_less_than_grid)
        dNdE_lab[rr,:] = savgol_filter(frac_interp,21,3,deriv=1)
        dNdE_lab[rr,dNdE_lab[rr,:]<0] = 0        
    else:
        dNdE_lab[rr,:] = float('nan')

p_EgivenR_lab = savgol_filter(dNdE_lab,21,3,axis=0)

all_Ei = np.loadtxt('data/partmc_bulk/all_Ei_lab_uniform.txt')
dNdE_lab_uniform =  np.zeros([len(Rbc_grid),len(Eabs_grid)])
for rr in range(len(Rbc_grid)):
    these = sum(np.vstack(
            [all_Ri>=(Rbc_grid[rr]-dRbc/2),
             all_Ri<(Rbc_grid[rr]+dRbc/2),
             all_tt_i<tt_max]))>2
    if sum(these)>0:
        idx = np.argsort(all_Ei[these])
        these_wi = all_wi[these]
        these_Ei = all_Ei[these]
        idx = np.argsort(these_Ei)
        frac_less_than = np.cumsum(these_wi[idx])/sum(these_wi)
        Eabs_grid_lab_uniform = np.interp(frac_less_than_grid,frac_less_than,these_Ei[idx])
        frac_interp = np.interp(Eabs_grid,Eabs_grid_lab_uniform,frac_less_than_grid)
        dNdE_lab_uniform[rr,:] = savgol_filter(frac_interp,21,3,deriv=1)
        dNdE_lab_uniform[rr,dNdE_lab_uniform[rr,:]<0] = 0        
    else:
        dNdE_lab_uniform[rr,:] = float('nan')        

p_EgivenR_lab_uniform = savgol_filter(dNdE_lab_uniform,21,3,axis=0)

all_Ei = np.loadtxt('data/partmc_bulk/all_Ei_cs.txt')
dNdE_cs =  np.zeros([len(Rbc_grid),len(Eabs_grid)])
for rr in range(len(Rbc_grid)):
    these = sum(np.vstack([all_Ri>=(Rbc_grid[rr]-dRbc/2),all_Ri<(Rbc_grid[rr]+dRbc/2)]))>1
    if sum(these)>0:
        idx = np.argsort(all_Ei[these])
        these_wi = all_wi[these]
        these_Ei = all_Ei[these]
        idx = np.argsort(these_Ei)
        frac_less_than = np.cumsum(these_wi[idx])/sum(these_wi)
        Eabs_grid_cs = np.interp(frac_less_than_grid,frac_less_than,these_Ei[idx])
        frac_interp = np.interp(Eabs_grid,Eabs_grid_cs,frac_less_than_grid)
        dNdE_cs[rr,:] = savgol_filter(frac_interp,21,3,deriv=1)
        dNdE_cs[rr,dNdE_cs[rr,:]<0] = 0 
    else:
        dNdE_cs[rr,:] = float('nan')

p_EgivenR_cs = savgol_filter(dNdE_cs,21,3,axis=0)

all_Ei = np.loadtxt('data/partmc_bulk/all_Ei_cs_uniform.txt')
dNdE_cs_uniform =  np.zeros([len(Rbc_grid),len(Eabs_grid)])
for rr in range(len(Rbc_grid)):
    these = sum(np.vstack(
            [all_Ri>=(Rbc_grid[rr]-dRbc/2),
             all_Ri<(Rbc_grid[rr]+dRbc/2),
             all_tt_i<tt_max]))>2
    if sum(these)>0:
        idx = np.argsort(all_Ei[these])
        these_wi = all_wi[these]
        these_Ei = all_Ei[these]
        idx = np.argsort(these_Ei)
        frac_less_than = np.cumsum(these_wi[idx])/sum(these_wi)
        Eabs_grid_cs_uniform = np.interp(frac_less_than_grid,frac_less_than,these_Ei[idx])        
        frac_interp = np.interp(Eabs_grid,Eabs_grid_cs_uniform,frac_less_than_grid)
        dNdE_cs_uniform[rr,:] = savgol_filter(frac_interp,21,3,deriv=1)
        dNdE_cs_uniform[rr,dNdE_cs_uniform[rr,:]<0] = 0
    else:
        dNdE_cs_uniform[rr,:] = float('nan')        

p_EgivenR_cs_uniform = savgol_filter(dNdE_cs_uniform,21,3,axis=0)

p_EgivenR_lab[p_EgivenR_lab<0] = 0
p_EgivenR_lab_uniform[p_EgivenR_lab_uniform<0] = 0
p_EgivenR_cs[p_EgivenR_cs<0] = 0
p_EgivenR_cs_uniform[p_EgivenR_cs_uniform<0] = 0

best_guess_lab = np.zeros(Rbc_grid.shape)
best_guess_lab_uniform = np.zeros(Rbc_grid.shape)
best_guess_cs = np.zeros(Rbc_grid.shape)
best_guess_cs_uniform = np.zeros(Rbc_grid.shape)

Eabs_ci_lab = np.zeros([len(Rbc_grid),len(ci_vals),2])
Eabs_ci_lab_uniform = np.zeros([len(Rbc_grid),len(ci_vals),2])
Eabs_ci_cs = np.zeros([len(Rbc_grid),len(ci_vals),2])
Eabs_ci_cs_uniform = np.zeros([len(Rbc_grid),len(ci_vals),2])
for rr in range(len(Rbc_grid)):
    best_guess_lab[rr] = sum(p_EgivenR_lab[rr,:]*Eabs_grid)/sum(p_EgivenR_lab[rr,:])
    best_guess_lab_uniform[rr] = sum(p_EgivenR_lab_uniform[rr,:]*Eabs_grid)/sum(p_EgivenR_lab_uniform[rr,:])
    best_guess_cs[rr] = sum(p_EgivenR_cs[rr,:]*Eabs_grid)/sum(p_EgivenR_cs[rr,:])
    best_guess_cs_uniform[rr] = sum(p_EgivenR_cs_uniform[rr,:]*Eabs_grid)/sum(p_EgivenR_cs_uniform[rr,:])
    for cc in range(len(ci_vals)):
        confidence_interval = [ci_vals[cc],1-ci_vals[cc]]
        for ii in range(len(confidence_interval)):
            Eabs_ci_lab[rr,cc,ii] = np.interp(
                    confidence_interval[ii],np.cumsum(p_EgivenR_lab[rr,:])/sum(p_EgivenR_lab[rr,:]),Eabs_grid)
            Eabs_ci_lab_uniform[rr,cc,ii] = np.interp(
                    confidence_interval[ii],np.cumsum(p_EgivenR_lab_uniform[rr,:])/sum(p_EgivenR_lab_uniform[rr,:]),Eabs_grid)
            Eabs_ci_cs[rr,cc,ii] = np.interp(
                    confidence_interval[ii],np.cumsum(p_EgivenR_cs[rr,:])/sum(p_EgivenR_cs[rr,:]),Eabs_grid)
            Eabs_ci_cs_uniform[rr,cc,ii] = np.interp(
                    confidence_interval[ii],np.cumsum(p_EgivenR_cs_uniform[rr,:])/sum(p_EgivenR_cs_uniform[rr,:]),Eabs_grid)

cc = 2

best_guess_lab[0] = 1
best_guess_lab_uniform[0] = 1
best_guess_cs[0] = 1
best_guess_cs_uniform[0] = 1

these = Rbc_grid>7.5
best_guess_lab[these] = savgol_filter(best_guess_lab[these],15,1)
best_guess_lab_uniform[these] = savgol_filter(best_guess_lab_uniform[these],15,1)
best_guess_cs[these] = savgol_filter(best_guess_cs[these],15,1)
best_guess_cs_uniform[these] = savgol_filter(best_guess_cs_uniform[these],15,1)

these = Rbc_grid > -1
best_guess_lab[these] = savgol_filter(best_guess_lab[these],3,1)
best_guess_lab_uniform[these] = savgol_filter(best_guess_lab_uniform[these],3,1)
best_guess_cs[these] = savgol_filter(best_guess_cs[these],3,1)
best_guess_cs_uniform[these] = savgol_filter(best_guess_cs_uniform[these],3,1)

all_std_i =np.loadtxt('data/partmc_std_dev/all_log10std_mass_i.txt')
#all_Ei = np.loadtxt('data/partmc_std_dev/all_Ei.txt')
all_Ei = np.loadtxt('data/partmc_bulk/all_Ei.txt')
all_Ei_lab_uniform = np.loadtxt('data/partmc_bulk/all_Ei_lab_uniform.txt')
all_Ei_cs_uniform = np.loadtxt('data/partmc_bulk/all_Ei_lab_uniform.txt')
all_Ei_cs = np.loadtxt('data/partmc_bulk/all_Ei_lab_uniform.txt')


#
Rbc_lims = [
        [1.8,2.2],
        [3.8,4.2],
        [5.8,6.2],
        [7.8,6.2],
        [9.8,10.2],        
        ]

Rbc_grid_coarse = np.linspace(1,12,6)

std_grid = np.linspace(0.4,1.25,50)
std_mids = np.linspace(std_grid.min()+(std_grid[1]-std_grid[0])/2,std_grid.max()-(std_grid[1]-std_grid[0])/2,len(std_grid)-1)

binned_by_std = list(np.zeros(len(Rbc_lims)))
for rr in range(0,len(Rbc_lims)):
    binned_by_std[rr] = list(np.zeros(len(std_grid)))
    idx = np.sum(np.vstack(
        [all_Ri>Rbc_lims[rr][0],
         all_Ri<Rbc_lims[rr][1]]),axis=0)>1    
    binned_by_std[rr][0] = all_Ei_lab_uniform[idx]
    for ss in range(1,len(std_grid)):
        idx = np.sum(np.vstack(
            [all_Ri>Rbc_lims[rr][0],
             all_Ri<Rbc_lims[rr][1],
             all_std_i>std_grid[ss-1],
             all_std_i<std_grid[ss]]),axis=0)>3
        binned_by_std[rr][ss] = all_Ei[idx]

h_s = 0.1
h_R = 0.3

these_Rbc = [2.5,4.0,5.5]
Eabs_best_ss = np.zeros([len(these_Rbc),len(std_mids)])
Rbc_lims = list([np.zeros(2),np.zeros(2),np.zeros(2)])
Eabs_ci_ss = np.zeros([len(these_Rbc),len(std_mids)+1,len(ci_vals),2])

for rr in range(len(these_Rbc)):
    these = sum(np.vstack(
        [all_Ri>=(these_Rbc[rr]-dRbc*5),
         all_Ri<(these_Rbc[rr]+dRbc*5)]))>1    
    Rbc_lims[rr][0] = these_Rbc[rr]-dRbc
    Rbc_lims[rr][1] = these_Rbc[rr]+dRbc   
    if sum(these)>0:
        for ss in range(len(std_mids)):
            w_i = gaussian_kernel(these_Rbc[rr],all_Ri[these],h_R)*gaussian_kernel(std_mids[ss],all_std_i[these],h_s)
            idx = np.argsort(all_Ei[these])
            these_wi = w_i*all_wi[these]
            these_Ei = all_Ei[these]
                    
            frac_less_than = np.cumsum(these_wi[idx])/sum(these_wi)
            dNdE = savgol_filter(frac_less_than,5,1,deriv=1)
            Eabs_best_ss[rr,ss] = sum(dNdE*these_Ei[idx])/sum(dNdE)
            for cc in range(len(ci_vals)):
                confidence_interval = [ci_vals[cc],1-ci_vals[cc]]
                for ii in range(len(confidence_interval)):
                    Eabs_ci_ss[rr,ss+1,cc,ii] = np.interp(confidence_interval[ii],frac_less_than,these_Ei[idx])            
    else:
        Eabs_best_ss[rr,:] = float('nan')
        
    these = sum(np.vstack([
            all_Ri>=(these_Rbc[rr]-h_R*5),
            all_Ri<(these_Rbc[rr]+h_R*5),
            all_tt_i<tt_max]))>2
    w_i = gaussian_kernel(these_Rbc[rr],all_Ri[these],h_R)
    these_wi = w_i*all_wi[these]
    these_Ei = all_Ei_lab_uniform[these]
    idx = np.argsort(these_Ei)
    frac_less_than = np.cumsum(these_wi[idx])/sum(these_wi)
    for cc in range(len(ci_vals)):
        confidence_interval = [ci_vals[cc],1-ci_vals[cc]]
        for ii in range(len(confidence_interval)):
            Eabs_ci_ss[rr,0,cc,ii] = np.interp(confidence_interval[ii],frac_less_than,these_Ei[idx])
Eabs_best_ss2 = np.zeros(Eabs_best_ss.shape)
window = 25
for rr in range(len(these_Rbc)):
    Eabs_best_ss2[rr,:] = savgol_filter(Eabs_best_ss[rr,:],window,1)
    
####################################################################################
#
#  Compute Bayes factor
#
####################################################################################


vals_M1 = 0.
vals_M2_cs = 0.
vals_M2_lab = 0.
vals_M2_cs_uniform = 0.
vals_M2_lab_uniform = 0.
f = interp2d(Rbc_grid, Eabs_grid, np.transpose(p_EgivenR_cs_uniform))
for ii in range(len(Rbc_observed)):
    vals_M1 += BC_observed[ii]*f(Rbc_observed[ii],Eabs_observed[ii])
    
f = interp2d(Rbc_grid, Eabs_grid, np.transpose(p_EgivenR_cs))
for ii in range(len(Rbc_observed)):
    vals_M2_cs += BC_observed[ii]*f(Rbc_observed[ii],Eabs_observed[ii])

f = interp2d(Rbc_grid, Eabs_grid, np.transpose(p_EgivenR_lab_uniform))
for ii in range(len(Rbc_observed)):
    vals_M2_lab_uniform += BC_observed[ii]*f(Rbc_observed[ii],Eabs_observed[ii])

f = interp2d(Rbc_grid, Eabs_grid, np.transpose(p_EgivenR_lab))
for ii in range(len(Rbc_observed)):
    vals_M2_lab += BC_observed[ii]*f(Rbc_observed[ii],Eabs_observed[ii])

B_cs_diverse = vals_M2_cs/vals_M1
B_lab_uniform = vals_M2_lab_uniform/vals_M1
B_lab_diverse = vals_M2_lab/vals_M1

vals_vsR_M1 = np.zeros(len(Rbc_grid))
vals_vsR_M2_cs = np.zeros(len(Rbc_grid))
vals_vsR_M2_lab_uniform = np.zeros(len(Rbc_grid))
vals_vsR_M2_lab = np.zeros(len(Rbc_grid))
just_weights = np.zeros(len(Rbc_grid))
h = 0.2
f = interp2d(Rbc_grid, Eabs_grid, np.transpose(p_EgivenR_cs_uniform))
for rr in range(len(Rbc_grid)):
    for ii in range(len(Rbc_observed)):
        vals_vsR_M1[rr] += BC_observed[ii]*gaussian_kernel_scalar(Rbc_grid[rr], Rbc_observed[ii],h)*f(Rbc_observed[ii],Eabs_observed[ii])
        
f = interp2d(Rbc_grid, Eabs_grid, np.transpose(p_EgivenR_cs))
for rr in range(len(Rbc_grid)):
    for ii in range(len(Rbc_observed)):
        vals_vsR_M2_cs[rr] += BC_observed[ii]*gaussian_kernel_scalar(Rbc_grid[rr], Rbc_observed[ii],h)*f(Rbc_observed[ii],Eabs_observed[ii])

f = interp2d(Rbc_grid, Eabs_grid, np.transpose(p_EgivenR_lab_uniform))
for rr in range(len(Rbc_grid)):
    for ii in range(len(Rbc_observed)):
        vals_vsR_M2_lab_uniform[rr] += BC_observed[ii]*gaussian_kernel_scalar(Rbc_grid[rr], Rbc_observed[ii],h)*f(Rbc_observed[ii],Eabs_observed[ii])

f = interp2d(Rbc_grid, Eabs_grid, np.transpose(p_EgivenR_lab))
for rr in range(len(Rbc_grid)):
    for ii in range(len(Rbc_observed)):
        vals_vsR_M2_lab[rr] += BC_observed[ii]*gaussian_kernel_scalar(Rbc_grid[rr], Rbc_observed[ii],h)*f(Rbc_observed[ii],Eabs_observed[ii])

for rr in range(len(Rbc_grid)):
    for ii in range(len(Rbc_observed)):
        just_weights[rr] += BC_observed[ii]*gaussian_kernel_scalar(Rbc_grid[rr], Rbc_observed[ii],h)

window = 35
BvsR_cs_diverse = savgol_filter(vals_vsR_M2_cs,window,1)/savgol_filter(vals_vsR_M1,window,1)
BvsR_lab_uniform = savgol_filter(vals_vsR_M2_lab_uniform,window,1)/savgol_filter(vals_vsR_M1,window,1)
BvsR_lab_diverse = savgol_filter(vals_vsR_M2_lab,window,1)/savgol_filter(vals_vsR_M1,window,1)

####################################################################################
#
# data for boxplot
#
####################################################################################

vals = np.loadtxt('data/measurements/liu2017_2c-3.csv',delimiter=',')
not_empty = np.zeros(len(Rbc_grid_coarse))>0
Rbc_grid_coarse = np.linspace(0,12,21)
Rbc_mids_coarse = 0.5*(Rbc_grid_coarse[range(len(Rbc_grid_coarse)-1)] + Rbc_grid_coarse[range(1,len(Rbc_grid_coarse))])
dat_chamber = list(np.zeros(len(Rbc_grid_coarse)-1))
for rr in range(1,len(Rbc_grid_coarse)):
    idx, = np.nonzero((vals[:,0]>Rbc_grid_coarse[rr-1]) & (vals[:,0]<=Rbc_grid_coarse[rr]))
    dat_chamber[rr-1] = vals[idx,1]
    

from scipy.stats import norm
vals = np.loadtxt('data/measurements/liu2017_1a_lab-diesel_1fg.csv',delimiter=',')
mu_Rbc_lab,std_Rbc_lab = norm.fit(np.log10(vals[:,0]),weights=vals[:,1])
vals = np.loadtxt('data/measurements/liu2017_1a_SF_1fg.csv',delimiter=',')
mu_Rbc_SF,std_Rbc_SF = norm.fit(np.log10(vals[:,0]),weights=vals[:,1])


Rbc_grid_coarse = np.linspace(0,12,21)
Rbc_mids_coarse = 0.5*(Rbc_grid_coarse[range(len(Rbc_grid_coarse)-1)] + Rbc_grid_coarse[range(1,len(Rbc_grid_coarse))])

Eabs_dat = list(np.zeros(len(Rbc_grid_coarse)-1))
for rr in range(1,len(Rbc_grid_coarse)):
    idx, = np.nonzero((Rbc_observed>Rbc_grid_coarse[rr-1]) & (Rbc_observed<=Rbc_grid_coarse[rr]))
    Eabs_dat[rr-1] = Eabs_observed[idx]    


    
####################################################################################
#
#  Make figure -- Eabs vs. std for different R
#
####################################################################################

cols = plt.cm.viridis([0,0.5,1])
#fig = plt.figure(num=None, figsize=(9.6,5.4), facecolor='w', edgecolor='k')
#fig = plt.figure(num=None, figsize=(8.,4.5), facecolor='w', edgecolor='k')
fig = plt.figure(num=None, figsize=(9.5,6.5), facecolor='w', edgecolor='k')
#fig = plt.figure(num=None, figsize=(11,5.85), facecolor='w', edgecolor='k')
ax1 = plt.subplot2grid((10,10), (2, 0), rowspan=4, colspan=6, fig=fig)
ax2 = plt.subplot2grid((10,10), (2, 6), rowspan=4, colspan=2, fig=fig)
ax3 = plt.subplot2grid((10,10), (7, 0), rowspan=2, colspan=2, fig=fig)
ax4 = plt.subplot2grid((10,10), (7, 2), rowspan=2, colspan=2, fig=fig)
ax5 = plt.subplot2grid((10,10), (7, 4), rowspan=2, colspan=2, fig=fig)
ax6 = plt.subplot2grid((10,10), (7, 6), rowspan=3, colspan=3, fig=fig)



l, b, w, h = ax2.get_position().bounds
dh_frac = 0.2
ax2.set_position([l, b+h*dh_frac, w, h*(1-dh_frac)])

medianprops = dict(linestyle='-', linewidth=1.5, color='k')
meanprops = dict(markerfacecolor='k',markeredgecolor='k')

ax1.plot(Rbc_grid,np.ones(len(Rbc_grid)),color='k',linewidth=1)


col = 'b'
window = 9
medianprops2 = dict(linestyle='-', linewidth=1.5, color=col)
meanprops2 = dict(markerfacecolor=col,markeredgecolor=col)
idx0 = 2

cc = 2
f1 = ax1.fill_between(Rbc_grid,savgol_filter(Eabs_ci_cs_uniform[:,cc,0],window,1),savgol_filter(Eabs_ci_cs_uniform[:,cc,1],window,1),alpha=0.08,color=col_cs_uniform)
l1, = ax1.plot(Rbc_grid,best_guess_cs_uniform,color=col_cs_uniform,linestyle='--')
f2 = ax1.fill_between(Rbc_grid,savgol_filter(Eabs_ci_cs[:,cc,0],window,1),savgol_filter(Eabs_ci_cs[:,cc,1],window,1),alpha=0.5,color=col_cs_part,zorder=1); 
l2, = ax1.plot(Rbc_grid,best_guess_cs,color=col_cs_part,zorder=1); 
f3 = ax1.fill_between(Rbc_grid,savgol_filter(Eabs_ci_lab_uniform[:,cc,0],window,1),savgol_filter(Eabs_ci_lab_uniform[:,cc,1],window,1),alpha=0.5,color=col_lab_uniform,zorder=2); 
l3, = ax1.plot(Rbc_grid,best_guess_lab_uniform,color=col_lab_uniform,zorder=3); 

cc = 2
f4 = ax1.fill_between(Rbc_grid,savgol_filter(Eabs_ci_lab[:,cc,0],window,1),savgol_filter(Eabs_ci_lab[:,cc,1],window,1),alpha=0.5,color=col_lab_part); 
l4, = ax1.plot(Rbc_grid,best_guess_lab,color=col_lab_part); 


h_obs2 = ax1.boxplot(Eabs_dat,positions=Rbc_mids_coarse,
                 showfliers=False,whis=[5,95],widths=0.3,#showmeans=True,
                 medianprops=medianprops,meanprops=meanprops);

ind = np.arange(4, 1, -1); 
pm, pc, pn = ax2.barh(ind,np.array([B_lab_uniform,B_cs_diverse,B_lab_diverse])[:,0],alpha=0.5); 
pm.set_facecolor(col_lab_uniform)
pc.set_facecolor(col_cs_part)#[255/255,204/255,0/255])
pn.set_facecolor(col_lab_part)

lg1a = ax1.legend([f1,f4,f3,f2,h_obs2['boxes'][0]],['default model', 'best model', 'deviation from core-shell only', 'heterogeneity in $R_{\mathrm{BC}}$ only','\nobservations of\nBC in urban outflow',],bbox_to_anchor=(0,1.02),loc="lower left",edgecolor=None,ncol=3)

ax1.add_artist(lg1a)

window = 21; rr = 0; cc = 2;
Eabs_lb = savgol_filter(Eabs_ci_ss[rr,range(1,len(std_grid)),cc,0],window,1)
Eabs_ub = savgol_filter(Eabs_ci_ss[rr,range(1,len(std_grid)),cc,1],window,1)
Eabs_best = savgol_filter(Eabs_ci_ss[rr,range(1,len(std_grid)),cc,0],window,1)


Eabs_lb_uniform = Eabs_ci_ss[rr,0,cc,0]*np.ones(len(std_mids)+1)
Eabs_ub_uniform = Eabs_ci_ss[rr,0,cc,1]*np.ones(len(std_mids)+1)
Eabs_best_uniform = savgol_filter(Eabs_ci_ss[rr,range(1,len(std_grid)),cc,0],window,1)


ax3.fill_between([std_mids.min(),std_mids.max()],-2*np.ones(2),3*np.ones(2),color=['C2'],alpha=0.03)
ax3.plot([0.0,0.0],[-1.0,3.0],color=col_lab_uniform,linewidth=2.0,alpha=0.5)
f5a = ax3.fill_between(std_mids,Eabs_lb,Eabs_ub,alpha=0.5,color=col_lab_part);
f5b = ax3.fill_between(np.linspace(0,std_mids.min(),10),
                 np.interp(np.linspace(0,std_mids.min(),10),np.append([0],std_mids),np.append([Eabs_lb_uniform[0]],Eabs_lb)),
                 np.interp(np.linspace(0,std_mids.min(),10),np.append([0],std_mids),np.append([Eabs_ub_uniform[0]],Eabs_ub)),alpha=0.2,color=col_lab_part);


f5a = ax3.plot(std_mids,Eabs_best,color=col_lab_part);
f5b = ax3.plot(std_mids,Eabs_best,color=col_lab_part);

f5b = ax3.fill_between(np.linspace(0,std_mids.min(),10),
                 np.interp(np.linspace(0,std_mids.min(),10),np.append([0],std_mids),np.append([Eabs_lb_uniform[0]],Eabs_lb)),
                 np.interp(np.linspace(0,std_mids.min(),10),np.append([0],std_mids),np.append([Eabs_ub_uniform[0]],Eabs_ub)),alpha=0.2,color=col_lab_part);

                       
window = 21; rr = 1; cc = 2;  
Eabs_lb = savgol_filter(Eabs_ci_ss[rr,range(1,len(std_grid)),cc,0],window,1)
Eabs_ub = savgol_filter(Eabs_ci_ss[rr,range(1,len(std_grid)),cc,1],window,1)

Eabs_lb_uniform = Eabs_ci_ss[rr,0,cc,0]*np.ones(len(std_mids)+1)
Eabs_ub_uniform = Eabs_ci_ss[rr,0,cc,1]*np.ones(len(std_mids)+1)

ax4.fill_between([std_mids.min(),std_mids.max()],-2*np.ones(2),3*np.ones(2),color=[col_lab_part],alpha=0.03)
ax4.plot([0.0,0.0],[-1.0,3.0],color=col_lab_uniform,linewidth=2.0,alpha=0.5)

f6a = ax4.fill_between(std_mids,Eabs_lb,Eabs_ub,alpha=0.5,color=col_lab_part);
f6b = ax4.fill_between(np.linspace(0,std_mids.min(),10),
                 np.interp(np.linspace(0,std_mids.min(),10),np.append([0],std_mids),np.append([Eabs_lb_uniform[0]],Eabs_lb)),
                 np.interp(np.linspace(0,std_mids.min(),10),np.append([0],std_mids),np.append([Eabs_ub_uniform[0]],Eabs_ub)),alpha=0.2,color=col_lab_part);

window = 21; rr = 2; cc = 2;  
Eabs_lb = savgol_filter(Eabs_ci_ss[rr,range(1,len(std_grid)),cc,0],window,1)
Eabs_ub = savgol_filter(Eabs_ci_ss[rr,range(1,len(std_grid)),cc,1],window,1)

Eabs_lb_uniform = Eabs_ci_ss[rr,0,cc,0]*np.ones(len(std_mids)+1)
Eabs_ub_uniform = Eabs_ci_ss[rr,0,cc,1]*np.ones(len(std_mids)+1)

ax5.fill_between([std_mids.min(),std_mids.max()],-2*np.ones(2),3*np.ones(2),color=[col_lab_part],alpha=0.03)
ax5.plot([0.0,0.0],[-1.0,3.0],color=col_lab_uniform,linewidth=2.0,alpha=0.5)

f7a = ax5.fill_between(std_mids,Eabs_lb,Eabs_ub,alpha=0.5,color=col_lab_part);
f7b = ax5.fill_between(np.linspace(0,std_mids.min(),10),
                 np.interp(np.linspace(0,std_mids.min(),10),np.append([0],std_mids),np.append([Eabs_lb_uniform[0]],Eabs_lb)),
                 np.interp(np.linspace(0,std_mids.min(),10),np.append([0],std_mids),np.append([Eabs_ub_uniform[0]],Eabs_ub)),alpha=0.2,color=col_lab_part);

scale_dRbc = 0.5
alpha_val = 1
rr = 0
other_vals = np.loadtxt('data/measurements/peng2016_3c-220nm.csv',delimiter=',')
D0 = 220
these = sum(np.vstack([(other_vals[:,0]*D0+D0)**3/D0**3<Rbc_lims[rr][1]+scale_dRbc*dRbc,(other_vals[:,0]*D0+D0)**3/D0**3>=Rbc_lims[rr][0]-scale_dRbc*dRbc]))>1
other_vals2 = np.loadtxt('data/measurements/peng2016_3c-150nm.csv',delimiter=',')
D0 = 150
these2 = sum(np.vstack([(other_vals2[:,0]*D0+D0)**3/D0**3<Rbc_lims[rr][1]+scale_dRbc*dRbc,(other_vals2[:,0]*D0+D0)**3/D0**3>=Rbc_lims[rr][0]-scale_dRbc*dRbc]))>1

wvl= 532e-9
dat = np.loadtxt('data/measurements/fig1_BC4data.csv',delimiter=',',skiprows=1)
Rbc_vol_lab = dat[:,0]
core_dia_lab =dat[:,5]
if wvl == 532e-9:
    Eabs_lab = dat[:,2]
elif wvl == 632e-9:
    Eabs_lab = dat[:,4]
elif wvl == 405e-9:
    Eabs_lab = dat[:,3]

these3 = sum(np.vstack([dat[:,0]<Rbc_lims[rr][1]+scale_dRbc*dRbc,dat[:,0]>=Rbc_lims[rr][0]-scale_dRbc*dRbc]))>1

these3 = np.where((Rbc_vol_lab<Rbc_lims[rr][1]) & (Rbc_vol_lab>=Rbc_lims[rr][0]) & (~np.isnan(Eabs_lab)))
ax3.errorbar(0,
             np.mean(Eabs_lab[these3]),np.std(Eabs_lab[these3]),
             fmt='ok',alpha=alpha_val)

dat = np.loadtxt('data/measurements/fig1a4_1fg.csv',delimiter=',')
Rbc_1fg,n_1fg = dat[:,0],dat[:,1]
dat = np.loadtxt('data/measurements/fig1a4_2fg.csv',delimiter=',')
Rbc_2fg,n_2fg = dat[:,0],dat[:,1]
dat = np.loadtxt('data/measurements/fig1a4_3fg.csv',delimiter=',')
Rbc_3fg,n_3fg = dat[:,0],dat[:,1]
dat = np.loadtxt('data/measurements/fig1a4_5fg.csv',delimiter=',')
Rbc_5fg,n_5fg = dat[:,0],dat[:,1]

mean_chamber = (
        sum(np.log10(Rbc_1fg)*n_1fg) + sum(np.log10(Rbc_2fg)*n_2fg) + sum(np.log10(Rbc_3fg)*n_3fg)+ sum(np.log10(Rbc_5fg)*n_5fg))/(sum(n_1fg) + sum(n_2fg) + sum(n_3fg) + sum(n_5fg))
std_chamber = np.round(np.sqrt(((sum(n_1fg*(np.log10(Rbc_1fg)-mean_chamber)**2) + 
 sum(n_2fg*(np.log10(Rbc_2fg)-mean_chamber)**2) + 
 sum(n_3fg*(np.log10(Rbc_3fg)-mean_chamber)**2) + 
 sum(n_5fg*(np.log10(Rbc_5fg)-mean_chamber)**2))/(sum(n_1fg) + sum(n_2fg) + sum(n_3fg) + sum(n_5fg)))),1)

vals = np.loadtxt('data/measurements/liu2017_2c-3.csv',delimiter=',')
these = sum(np.vstack([vals[:,0]<Rbc_lims[rr][1],vals[:,0]>=Rbc_lims[rr][0]]))>1
#ax3.errorbar(std_chamber, np.mean(vals[these,1]), np.std(vals[these,1]), fmt='*k',alpha=alpha_val)
h_lab2 = ax3.scatter(std_chamber, np.mean(vals[these,1]), 80, 'k', marker='*',alpha=alpha_val)


dat = np.loadtxt('data/measurements/fig1a3_1fg.csv',delimiter=',')
Rbc_1fg,n_1fg = dat[:,0],dat[:,1]
dat = np.loadtxt('data/measurements/fig1a3_2fg.csv',delimiter=',')
Rbc_2fg,n_2fg = dat[:,0],dat[:,1]
dat = np.loadtxt('data/measurements/fig1a3_5fg.csv',delimiter=',')
Rbc_5fg,n_5fg = dat[:,0],dat[:,1]

mean_ambient = (
        sum(np.log10(Rbc_1fg)*n_1fg) + sum(np.log10(Rbc_2fg)*n_2fg) + sum(np.log10(Rbc_5fg)*n_5fg))/(sum(n_1fg) + sum(n_2fg) + sum(n_5fg))
std_ambient = np.round(np.sqrt(((
        sum(n_1fg*(np.log10(Rbc_1fg)-mean_ambient)**2) + 
        sum(n_2fg*(np.log10(Rbc_2fg)-mean_ambient)**2) + 
        sum(n_5fg*(np.log10(Rbc_5fg)-mean_ambient)**2))/(sum(n_1fg) + sum(n_2fg) + sum(n_5fg)))),1)


these = sum(np.vstack([Rbc_observed<Rbc_lims[rr][1],Rbc_observed>=Rbc_lims[rr][0]]))>1
ax3.boxplot(Eabs_observed[these],positions=[std_ambient], showfliers=False,whis=[5,95],widths=0.15,#showmeans=True,
            medianprops=medianprops,meanprops=meanprops,zorder=10);
            
             
rr = 1
these3 = np.where((Rbc_vol_lab<Rbc_lims[rr][1]) & (Rbc_vol_lab>=Rbc_lims[rr][0]) & (~np.isnan(Eabs_lab)))
ax4.errorbar(0,
             np.mean(Eabs_lab[these3]),np.std(Eabs_lab[these3]),
             fmt='ok',alpha=alpha_val)
vals = np.loadtxt('data/measurements/liu2017_2c-3.csv',delimiter=',')
these = sum(np.vstack([vals[:,0]<Rbc_lims[rr][1],vals[:,0]>=Rbc_lims[rr][0]]))>1
#ax4.errorbar(std_chamber, np.mean(vals[these,1]), np.std(vals[these,1]), fmt='*k',alpha=alpha_val)
ax4.scatter(std_chamber, np.mean(vals[these,1]), 80, 'k', marker='*',alpha=alpha_val)

these = sum(np.vstack([Rbc_observed<Rbc_lims[rr][1],Rbc_observed>=Rbc_lims[rr][0]]))>1
ax4.boxplot(Eabs_observed[these],positions=[std_ambient], showfliers=False,whis=[5,95],widths=0.15,
            medianprops=medianprops,meanprops=meanprops);
            
rr = 2            
these3 = np.where((Rbc_vol_lab<Rbc_lims[rr][1]) & (Rbc_vol_lab>=Rbc_lims[rr][0]) & (~np.isnan(Eabs_lab)))
h_lab1 = ax5.errorbar(0,
             np.mean(Eabs_lab[these3]),np.std(Eabs_lab[these3]),
             fmt='ok',alpha=alpha_val)

vals = np.loadtxt('data/measurements/liu2017_2c-3.csv',delimiter=',')
these = sum(np.vstack([vals[:,0]<Rbc_lims[rr][1],vals[:,0]>=Rbc_lims[rr][0]]))>1
#h_lab2 = ax5.errorbar(std_chamber, np.mean(vals[these,1]), np.std(vals[these,1]), fmt='*k',alpha=alpha_val)
ax5.scatter(std_chamber, np.mean(vals[these,1]), 80, 'k', marker='*',alpha=alpha_val)


these = sum(np.vstack([Rbc_observed<Rbc_lims[rr][1],Rbc_observed>=Rbc_lims[rr][0]]))>1
h_obs = ax5.boxplot(Eabs_observed[these],positions=[std_ambient], showfliers=False,whis=[5,95],widths=0.15,#showmeans=True,
            medianprops=medianprops,meanprops=meanprops);
            
ax3.set_xlim([-0.1,0.9])
ax4.set_xlim([-0.1,0.9])
ax5.set_xlim([-0.1,0.9])

ax3.set_ylim([0.75,1.9])
ax4.set_ylim([0.75,1.9])
ax5.set_ylim([0.75,1.9])


xticks = np.linspace(0,12,7)
ax1.set_xticks(xticks)

xtick_labels = [ "%d" % int(float(l)) for l in xticks]
ax1.set_xticklabels(xtick_labels)

ylims = ax2.get_ylim()
ax1.set_ylim([0.75,1.85])
ax1.set_xlim([0.5,12.5])

ax3.set_yticks([1,1.4,1.8])
ax4.set_yticks([1,1.4,1.8])
ax5.set_yticks([1,1.4,1.8])

ax4.set_yticklabels('')
ax5.set_yticklabels('')
xticks = [0,0.4,0.8]
ax3.set_xticks(xticks)
ax3.set_xticklabels(xticks)
ax4.set_xticks(xticks)
ax4.set_xticklabels(xticks)
ax5.set_xticks(xticks)
ax5.set_xticklabels(xticks)

ax2.set_xscale('log')
ax2.set_xlim([0.5,200])
ax2.set_xticks([1.,10.,100.])
ax2.set_yticklabels('')

ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

ax6.set_xticklabels('')
ax6.set_yticklabels('')
ax6.set_yticks([])
ax6.set_xticks([])
ax6.spines['left'].set_visible(False)
ax6.spines['right'].set_visible(False)
ax6.spines['top'].set_visible(False)
ax6.spines['bottom'].set_visible(False)

ax1.set_xlabel('population-averaged $R_{\mathrm{BC}}$')
ax1.set_ylabel('$E_{\mathrm{abs}}$')
ax2.set_xlabel('Bayes Factor')
ax3.set_ylabel('$E_{\mathrm{abs}}$')
ax4.set_xlabel('variability in per-particle $R_\mathrm{BC}$')
ax2.text(10**(np.log10(B_lab_uniform)+0.1),ind[0],'including deviation\nfrom core-shell only',verticalalignment='center')
ax2.text(10**(np.log10(B_cs_diverse)+0.1),ind[1],'including heterogeneity\nin $R_{\mathrm{BC}}$ only',verticalalignment='center')
ax2.text(10**(np.log10(B_lab_diverse)+0.1),ind[2],'including deviation\nfrom core-shell and\nheterogeneity in $R_{\mathrm{BC}}$',verticalalignment='center')
lg2 = ax6.legend([h_lab1,h_lab2,h_obs["boxes"][0],f7a],['lab, monodisperse BC$^\mathrm{i}$','lab, chamber studies$^\mathrm{ii}$','observations of BC\nin urban outflow$^\mathrm{iii}$', 'model (this study)'],bbox_to_anchor = (0.,1.05), loc='upper left')
ax6.text(0.03,0.39,'(i) BC4, Peng et al., 2016 (ii) Liu et\nal., 2017, (iii) Cappa et al., 2019',fontsize=8,verticalalignment='top')
fig.subplots_adjust(right=0.9,left=0.1,top=0.99,bottom=0.03,hspace=0.25,wspace=0.35)
fig.savefig('../figs/fig2.png',format='png', dpi=1000)