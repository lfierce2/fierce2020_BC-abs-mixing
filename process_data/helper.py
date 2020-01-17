#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 09:49:28 2020

@author: fiercenator
"""

def get_Eabs_model_params(Rbc_lab, Eabs_cs, Eabs_lab, param_type, params_range, N, N_burn):
    params_old = np.zeros(len(params_range))
    
    for pp in range(len(params_range)):
        params_old[pp] = np.random.uniform(low=params_range[pp][0],high=params_range[pp][1])
        
    params_list = np.zeros([N,len(params_old)])
    for ii in range(N + N_burn):
        likelihood_old = evaluate_likelihood(Rbc_lab, Eabs_cs, Eabs_lab, param_type, params_old)
        params_new = get_params(param_type, params_old, params_range)
        likelihood_new = evaluate_likelihood(Rbc_lab, Eabs_cs, Eabs_lab, param_type, params_new)
        u = np.random.uniform(low=0,high=1)
        if u < (likelihood_new/likelihood_old):
            params_old = params_new
        if ii >= N_burn:
            params_list[ii-N_burn,:] = params_old
    return params_list

def evaluate_likelihood(Rbc_lab, Eabs_cs, Eabs_lab, param_type, params):
    Eabs_param = get_Eabs_param(Rbc_lab, Eabs_cs, param_type, params)
    not_nan, = np.where(~np.isnan(Eabs_lab + Eabs_param))    
    sigE = params[len(params)-1]
    p = np.prod(norm.pdf(Eabs_lab[not_nan],loc=Eabs_param[not_nan],scale=sigE))
    return p

def get_params(param_type, params_old, params_range):
    params_new = np.zeros(np.shape(params_old))
    scaling_factor = 1./30
    if (param_type == 'Liu2017') or (param_type == 'Liu2017_fmax'):
        for pp in range(len(params_range)):
            if pp == 1:
                params_new[pp] = np.random.normal(loc=params_old[pp],scale=(params_range[pp][1]-params_range[pp][0])*scaling_factor)
                while params_new[pp]<params_new[0] or params_new[pp]>params_range[pp][1]:
                    params_new[pp] = np.random.normal(loc=params_old[pp],scale=(params_range[pp][1]-params_range[pp][0])*scaling_factor)
            else:
                params_new[pp] = np.random.normal(loc=params_old[pp],scale=(params_range[pp][1]-params_range[pp][0])*scaling_factor)
                while params_new[pp]<params_range[pp][0] or params_new[pp]>params_range[pp][1]:
                    params_new[pp] = np.random.normal(loc=params_old[pp],scale=(params_range[pp][1]-params_range[pp][0])*scaling_factor)
    else:
        for pp in range(len(params_range)):
            params_new[pp] = np.random.normal(loc=params_old[pp],scale=(params_range[pp][1]-params_range[pp][0])*scaling_factor)
            while params_new[pp]<params_range[pp][0] or params_new[pp]>params_range[pp][1]:
                params_new[pp] = np.random.normal(loc=params_old[pp],scale=(params_range[pp][1]-params_range[pp][0])*scaling_factor)
    return params_new

def get_Eabs_param(Rbc_lab, Eabs_cs, param_type, params):
    from scipy.interpolate import interp1d
    if param_type == 'modifiedLiu2017_fmax' :
        Rbc_max = params[0]
        f_max = params[1]
        Fin = np.array(Rbc_lab/(Rbc_max))
        Fin[Rbc_lab>=Rbc_max] = 1.
        Eabs = f_max*(Eabs_cs*Fin + (1-Fin))
    elif param_type == 'modifiedLiu2017' :
        Rbc_max = params[0]
        Fin = np.array(Rbc_lab/(Rbc_max))
        Fin[Rbc_lab>=Rbc_max] = 1.
        Eabs = (Eabs_cs*Fin + (1-Fin))
    elif param_type == 'Liu2017':
        Rbc_min = params[0]
        Rbc_max = params[1]
        Fin = np.array(Rbc_lab/(Rbc_max - Rbc_min) - Rbc_min/(Rbc_max - Rbc_min))
        Fin[Rbc_lab<=Rbc_min] = 0.
        Fin[Rbc_lab>=Rbc_max] = 1.
        Eabs = Eabs_cs*Fin + (1-Fin)
    elif param_type == 'Liu2017_fmax': 
        Rbc_min = params[0]
        Rbc_max = params[1]
        f_max = params[2]
        Fin = np.array(Rbc_lab/(Rbc_max - Rbc_min) - Rbc_min/(Rbc_max - Rbc_min))
        Fin[Rbc_lab<=Rbc_min] = 0.
        Fin[Rbc_lab>=Rbc_max] = 1.
        Eabs = (f_max*Eabs_cs*Fin + (1-Fin))
    elif param_type == 'f_max':
        f_max = params[0]
        Eabs = f_max*Eabs_cs
    elif param_type == 'effective_coating':
        f_offset = params[0]
        Rbc_effective = Rbc_lab*f_offset
        Eabs = interp1d(np.hstack([0.,Rbc_lab]),np.hstack([1.,Eabs_cs]))(Rbc_effective)
    return Eabs

def gaussian_kernel_scalar(x, x_i, hx):
    import numpy as np
    K = np.exp(-((x-x_i)/hx)**2/2);
    if abs(x-x_i) > hx*10.:
        K = 0.0
    return K

def gaussian_kernel(x, x_i, hx):
    import numpy as np
    K = np.exp(-((x-x_i)/hx)**2/2);
    too_big = abs(x-x_i) > hx*100.
    K[too_big] = 0.0
    return K
