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

import PyMieScatt 
from matplotlib.colors import ListedColormap

# =============================================================================
#  Define parameter ranges and paramterization form
# =============================================================================
Rbc_min_range = [0,9]
Rbc_max_range = [0,20]
fmax_range = [0,1]
sigE_range = [0.001,1]
Rbc_effective_range = [0,10]
f_offset_range = [0.,1.]

param_type = 'modifiedLiu2017' 
params_range = [Rbc_max_range,sigE_range]

wvls = [405e-9, 532e-9, 632e-9]

####################################################################################
#
#  Read in BC4 data, fit parameterization, plot
#
####################################################################################

dat = np.loadtxt('data/measurements/fig1_BC4data.csv',delimiter=',',skiprows=1)
idx_notisnan, = np.where(~np.isnan(dat[:,7]));
Rbc_vol = dat[:,0]
idx_sorted = np.argsort(Rbc_vol)
core_dia =dat[:,5]
m_core = np.zeros(dat[:,0].shape)/0. + 0j
m_shell = np.zeros(dat[:,0].shape)/0. + 0j

Eabs_lab = np.zeros([dat.shape[0],len(wvls)])/0.

Eabs_cs = np.zeros([dat.shape[0],len(wvls)])/0.
Eabs_cs_2fg = np.zeros([dat.shape[0],len(wvls)])/0.
Eabs_cs_5fg = np.zeros([dat.shape[0],len(wvls)])/0.
Eabs_cs_10fg = np.zeros([dat.shape[0],len(wvls)])/0.
Eabs_cs_20fg = np.zeros([dat.shape[0],len(wvls)])/0.

Eabs_param = np.zeros([dat.shape[0],len(wvls)])/0.
Eabs_param_2fg = np.zeros([dat.shape[0],len(wvls)])/0.
Eabs_param_5fg = np.zeros([dat.shape[0],len(wvls)])/0.
Eabs_param_10fg = np.zeros([dat.shape[0],len(wvls)])/0.
Eabs_param_20fg = np.zeros([dat.shape[0],len(wvls)])/0.

ww = 0
N = 500
N_burn = 2000


for wvl in wvls:
    if wvl == 532e-9:
        Eabs_lab[:,ww] = dat[:,2]
        m_core[idx_notisnan] =1.88 + 0.8j
    elif wvl == 632e-9:
        Eabs_lab[:,ww] = dat[:,4]
        m_core[idx_notisnan] =  1.95 + 0.79j
    elif wvl == 405e-9:
        Eabs_lab[:,ww] = dat[:,3]
        m_core[idx_notisnan] = 1.88 + 0.8j
        
    m_soa = 1.45 + 1e-4j
    m_inorg = 1.5
    m_shell[:] = m_soa
    m_shell[np.where(dat[:,7]==4)] = m_inorg
    m_shell[np.where(dat[:,7]==5)] = 0.5*m_inorg + 0.5*m_soa

    
    cols = np.zeros(len(dat[:,0]))/0.
    cols[np.where(dat[:,7]==1)] = 1
    cols[np.where(dat[:,7]==4)] = 2
    cols[np.where(dat[:,7]==5)] = 3
    
    for ii in range(dat.shape[0]):
        if ~np.isnan(dat[ii,7] + Rbc_vol[ii] + core_dia[ii]):
            # This one is used to make the parameterization
            output = PyMieScatt.MieQ(m_core[ii], wvl*1e9, core_dia[ii], asCrossSection=True)
            abs_crossect_bc = output[2]
            dry_dia = (core_dia[ii]**3*(1.+Rbc_vol[ii]))**(1./3.)
            output = PyMieScatt.MieQCoreShell(m_core[ii], m_shell[ii], wvl*1e9, core_dia[ii], dry_dia,asCrossSection=True)
            Eabs_cs[ii,ww] = output[2]/abs_crossect_bc
            
            # These are used to show the parameterization
            
            # 1-fg BC core
            Dcore_2fg = (2e-18/1800*6/np.pi)**(1/3)*1e9;
            output = PyMieScatt.MieQ(m_core[ii], wvl*1e9, Dcore_2fg, asCrossSection=True)
            abs_crossect_bc_2fg = output[2]            
            output = PyMieScatt.MieQCoreShell(m_core[ii], m_soa, wvl*1e9, Dcore_2fg, (Dcore_2fg**3*(Rbc_vol[ii]+1))**(1/3),asCrossSection=True)
            Eabs_cs_2fg[ii,ww] = output[2]/abs_crossect_bc_2fg

            # 5-fg BC core
            Dcore_5fg = (5e-18/1800*6/np.pi)**(1/3)*1e9;
            output = PyMieScatt.MieQ(m_core[ii], wvl*1e9, Dcore_5fg, asCrossSection=True)
            abs_crossect_bc_5fg = output[2]            
            output = PyMieScatt.MieQCoreShell(m_core[ii], m_soa, wvl*1e9, Dcore_5fg, (Dcore_5fg**3*(Rbc_vol[ii]+1))**(1/3),asCrossSection=True)
            Eabs_cs_5fg[ii,ww] = output[2]/abs_crossect_bc_5fg

            # 10-fg BC core
            Dcore_10fg = (10e-18/1800*6/np.pi)**(1/3)*1e9;
            output = PyMieScatt.MieQ(m_core[ii], wvl*1e9, Dcore_10fg, asCrossSection=True)
            abs_crossect_bc_10fg = output[2]            
            output = PyMieScatt.MieQCoreShell(m_core[ii], m_soa, wvl*1e9, Dcore_10fg, (Dcore_10fg**3*(Rbc_vol[ii]+1))**(1/3),asCrossSection=True)
            Eabs_cs_10fg[ii,ww] = output[2]/abs_crossect_bc_10fg            


            # 20-fg BC core
            Dcore_20fg = (20e-18/1800*6/np.pi)**(1/3)*1e9;
            output = PyMieScatt.MieQ(m_core[ii], wvl*1e9, Dcore_20fg, asCrossSection=True)
            abs_crossect_bc_20fg = output[2]            
            output = PyMieScatt.MieQCoreShell(m_core[ii], m_soa, wvl*1e9, Dcore_20fg, (Dcore_20fg**3*(Rbc_vol[ii]+1))**(1/3),asCrossSection=True)
            Eabs_cs_20fg[ii,ww] = output[2]/abs_crossect_bc_20fg                        
    ww = ww + 1
    
all_Rbc_vol = Rbc_vol.reshape([-1,1])
for ww in range(1,len(wvls)):
    all_Rbc_vol = np.hstack([all_Rbc_vol,Rbc_vol.reshape([-1,1])])


params = get_Eabs_model_params_redo(all_Rbc_vol.ravel(),
                                    Eabs_cs.ravel(), Eabs_lab.ravel(), param_type, params_range, N, N_burn)

wvl = 532e-9
ww = 0


Eabs_param[:,ww] = get_Eabs_param_redo(Rbc_vol, Eabs_cs[:,ww], param_type, np.mean(params,axis=0))
Eabs_param_2fg[:,ww] = get_Eabs_param_redo(Rbc_vol, Eabs_cs_2fg[:,ww], param_type, np.mean(params,axis=0))
Eabs_param_5fg[:,ww] = get_Eabs_param_redo(Rbc_vol, Eabs_cs_5fg[:,ww], param_type, np.mean(params,axis=0))
Eabs_param_10fg[:,ww] = get_Eabs_param_redo(Rbc_vol, Eabs_cs_10fg[:,ww], param_type, np.mean(params,axis=0))    
Eabs_param_20fg[:,ww] = get_Eabs_param_redo(Rbc_vol, Eabs_cs_20fg[:,ww], param_type, np.mean(params,axis=0))        

fig = plt.figure()
ax = fig.subplots(1)

hln_cs_2fg, = ax.plot(Rbc_vol[idx_sorted],Eabs_cs_2fg[idx_sorted,ww],color='k',linestyle=':')    
hln_cs_5fg, = ax.plot(Rbc_vol[idx_sorted],Eabs_cs_5fg[idx_sorted,ww],color='k',linestyle='-.')    
hln_cs_10fg, = ax.plot(Rbc_vol[idx_sorted],Eabs_cs_10fg[idx_sorted,ww],color='k',linestyle='--')        
hln_cs_20fg, = ax.plot(Rbc_vol[idx_sorted],Eabs_cs_20fg[idx_sorted,ww],color='k',linestyle='-')    

these, = np.where((cols==1)|(cols==2)|(cols==3))

N = 4
vals = np.ones((len(cols), 4))
col0 = np.linspace(230/256, 1, 4)
col1 = np.linspace(184/256, 1, 4)
col2 = np.linspace(0/256, 1, 4)    

col_idx = 0
these, = np.where((cols==1))
vals[these, 0] = col0[col_idx]
vals[these, 1] = col1[col_idx]
vals[these, 2] = col2[col_idx]

col_idx = 1
these, = np.where((cols==2))
vals[these, 0] = col0[col_idx]
vals[these, 1] = col1[col_idx]
vals[these, 2] = col2[col_idx]

col_idx = 2
these, = np.where((cols==3))
vals[these, 0] = col0[col_idx]
vals[these, 1] = col1[col_idx]
vals[these, 2] = col2[col_idx]

vals_leg = np.ones((N, 4))
vals_leg[:, 0] = col0
vals_leg[:, 1] = col1
vals_leg[:, 2] = col2

these, = np.where((cols==1)|(cols==2)|(cols==3))
hsc_lab = ax.scatter(Rbc_vol[these],Eabs_lab[these,ww],s=10*np.pi/6*(core_dia[these]/1e9)**3*1800*1e18,c=cols[these],edgecolor='k',cmap=cm.Reds,zorder=10);
herr_lab = ax.errorbar(Rbc_vol[these],Eabs_lab[these,ww],yerr=0.08,ecolor='k',fmt='none',zorder=0)#cols[these],cmap=cm.Reds);

####################################################################################
#
#  Read in and plot Fontana data --> wavelength = 532 nm
#
####################################################################################
    
observational_dat = np.genfromtxt('FontanaData-090817.txt',delimiter='\t',skip_header=1);

from datetime import datetime
july4_2015 = (datetime(2015, 7, 4, 0, 0)-datetime(1904, 1, 1, 0, 0)).total_seconds()
july7_2015 = (datetime(2015, 7, 7, 0, 0)-datetime(1904, 1, 1, 0, 0)).total_seconds()

these, = np.nonzero((sum(np.vstack([observational_dat[:,1]<july4_2015, observational_dat[:,2]>july7_2015]),0)>0) & 
                    (observational_dat[:,15]>0)&~np.isnan(observational_dat[:,6]))

BC_observed = observational_dat[these,7]
Rbc_observed = observational_dat[these,15]

MAC_observed = observational_dat[these,4]
Eabs_observed = observational_dat[these,6]

Rbc_grid_coarse = np.linspace(0,12,21)
Rbc_mids_coarse = 0.5*(Rbc_grid_coarse[range(len(Rbc_grid_coarse)-1)] + Rbc_grid_coarse[range(1,len(Rbc_grid_coarse))])

Eabs_dat = list(np.zeros(len(Rbc_grid_coarse)-1))
for rr in range(1,len(Rbc_grid_coarse)):
    idx, = np.nonzero((Rbc_observed>Rbc_grid_coarse[rr-1]) & (Rbc_observed<=Rbc_grid_coarse[rr]))
    Eabs_dat[rr-1] = Eabs_observed[idx]
    
medianprops = dict(linestyle='-', linewidth=1.5, color='k')
meanprops = dict(markerfacecolor='k',markeredgecolor='k')    

ax.plot([0,13],np.ones(2),color='k',linewidth=1)

h_obs = ax.boxplot(Eabs_dat,positions=Rbc_mids_coarse,
         showfliers=False,whis=[5,95],widths=0.3,#showmeans=True,
         medianprops=medianprops,meanprops=meanprops,patch_artist=True);
for patch in h_obs['boxes']:
    patch.set_facecolor('white')


####################################################################################
#
#  Add legend, add labels, format figure
#
####################################################################################


fontProperties = {'family':'sans-serif','fontname':['Helvetica'],
    'weight' : 'normal', 'size' : 10}
hsc_lab1 = ax.scatter(-10,-10,s=10*2,c='w',edgecolor='k')
hsc_lab1 = [hsc_lab1,ax.scatter(-10,-10,s=10*5,c='w',edgecolor='k')]
hsc_lab1 = [hsc_lab1[0],hsc_lab1[1],ax.scatter(-10,-10,s=10*10,c='w',edgecolor='k')]
hsc_lab1 = [hsc_lab1[0],hsc_lab1[1],hsc_lab1[2],ax.scatter(-10,-10,s=10*20,c='w',edgecolor='k')]

types = np.array([0.,1.,2.])
type_labs = [r'$\alpha$-Pinene SOA', 'H$_2$SO$_4$', r'H$_2$SO$_4$ + $\alpha$-Pinene SOA']
cols = cm.Reds(types/2)

hsc_lab2 = ax.scatter(-10,-10,s=10*5,c=cols[0,:],edgecolor='k')
hsc_lab2 = [hsc_lab2,ax.scatter(-10,-10,s=10*5,c=cols[1,:],edgecolor='k')]
hsc_lab2 = [hsc_lab2[0],hsc_lab2[1],ax.scatter(-10,-10,s=10*5,c=cols[2,:],edgecolor='k')]

xticks = np.linspace(0,12,7)
ax.set_xticks(xticks)

xtick_labels = [ "%d" % int(float(l)) for l in xticks]
ax.set_xticklabels(xtick_labels)

ax.set_ylim([0.8,2.0])
ax.set_xlim([0,12.5])

ax.set_xlabel('$R_{\mathrm{BC}}$')
ax.set_ylabel('absorption enhancement, $E_{\mathrm{abs}}$')


hleg1 = ax.legend(hsc_lab2,type_labs,title='coating type',loc=(1.0,1.0),bbox_to_anchor=(0.5,0.6),framealpha=1.)
hleg2 = ax.legend(hsc_lab1,['2 fg', '5 fg','10 fg','20 fg'],title='mass of BC core',loc=(1.0,1.0),bbox_to_anchor=(0.5,0.237),framealpha=1.)

ax.add_artist(hleg2)
hleg3 = ax.legend([hln_cs_2fg,hln_cs_5fg,hln_cs_10fg,hln_cs_20fg],['2 fg','5 fg','10 fg', '20 fg'],title='core-shell model',loc=(1.0,1.0),bbox_to_anchor=(0.5,-0.125),framealpha=1.)
ax.add_artist(hleg1)
ax.add_artist(hleg2)
