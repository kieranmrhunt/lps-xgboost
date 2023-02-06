import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import linregress
import shap
from statsmodels.nonparametric.smoothers_lowess import lowess as  sm_lowess
import cmath
from matplotlib.colors import LinearSegmentedColormap
import cartopy
import cartopy.feature as cfeature


df = pd.read_csv("../temp_data/filtered_lps_with_enviromental.csv")
df['dvo850_dt'] = df['dvo850_dt']*4

#df = df[df['reached_peak']==0]
#print(df.columns)
'''
['x', 'y', 'frame', 'year', 'month', 'day', 'hour', 'radius', 'max_vort',
       'mean_vort', 'mean_vort_850', 'mean_vort_700', 'mean_vort_500',
       'max_trunc_vort', 'track_id', 'mean_swv', 'mean_u200', 'mean_u850',
       'mean_skt', 'mean_land_frac', 'mcz_tcwv', 'vortex_depth', 'over_land',
       'dvo850_dt', 'acc_land_time', 'total_land_time', 'qshear_850',
       'ushear_850', 'mean_cape', 'mcz_cape', 'mean_dthetae_dp_900_750',
       'mean_dthetae_dp_750_500', 'mean_land_skt', 'mean_sst',
       'mean_swvl1_grad', 'mean_swvl2_grad', 'mean_swvl1', 'mean_swvl2',
       'olr_90', 'olr_75', 'olr_50', 'qshear_850_background',
       'ushear_850_background', 'mean_q_850', 'orography_height',
       'peak_vorticity', 'reached_peak', 'mean_prcp_400', 'mean_prcp_800',
       'max_prcp_400', 'max_prcp_800', 'mean_vimfd_400', 'mean_v200',
       'mean_v500', 'mean_v850', 'mean_u500', 'zonal_speed', 'merid_speed']
'''

feature_names = np.array(['x', 'y', 'hour', 'month',
       'mean_vort_850', 'mean_vort_500', 'vortex_depth',
       'mean_u200', 'mean_u500', 'mean_u850',
       'mean_v200', 'mean_v500', 'mean_v850',
       'mean_skt', 'mean_land_frac', 'mcz_tcwv',
       'mean_cape', 'mcz_cape', 'mean_dthetae_dp_900_750',
       'mean_dthetae_dp_750_500', 'mean_land_skt', 'mean_sst',
       'mean_swvl1_grad', 'mean_swvl2_grad', 'mean_swvl1', 'mean_swvl2',
       'olr_90', 'olr_50', 'qshear_850_background',
       'ushear_850_background', 'mean_q_850', 'orography_height',])



lons = df.x.values
lats = df.y.values
vds = df.vortex_depth.values
u500s = df.mean_u500.values 

plt.figure(figsize=(10,5))
ax1 = plt.subplot(1,2,1, projection=cartopy.crs.PlateCarree())


r=1.5
gridx = np.arange(70,95,r)
gridy = np.arange(5,30,r)

grid_u_hi = np.zeros((len(gridy),len(gridx)))*np.nan
grid_v_hi = np.zeros((len(gridy),len(gridx)))*np.nan

grid_u_lo = np.zeros((len(gridy),len(gridx)))*np.nan
grid_v_lo = np.zeros((len(gridy),len(gridx)))*np.nan

ihi = u500s>np.percentile(u500s,75)
ilo = u500s<np.percentile(u500s,25)

s=200

for j, y in enumerate(gridy):
	for i, x in enumerate(gridx):
		it = (lats<=y+r)&(lats>y-r)&(lons>x-r)&(lons<=x+r)
		
		it1 = it&ihi
		it2 = it&ilo
		
		if np.sum(it1)>=5:
			grid_u_hi[j,i]=np.mean(df.zonal_speed.values[it1])
			grid_v_hi[j,i]=np.mean(df.merid_speed.values[it1])

		if np.sum(it2)>=5:
			grid_u_lo[j,i]=np.mean(df.zonal_speed.values[it2])		
			grid_v_lo[j,i]=np.mean(df.merid_speed.values[it2])	
		
		
q1 = ax1.quiver(gridx,gridy, grid_u_lo, grid_v_lo, color='k', scale=s)
q2 = ax1.quiver(gridx,gridy, grid_u_hi, grid_v_hi, color='r', scale=s)

qk1 = ax1.quiverkey(q1, 0.225, -0.1, 10, label="<25th %ile (10 km hr$^{-1}$)", labelpos='S',)
qk2 = ax1.quiverkey(q2, 0.775, -0.1, 10, label=">75th %ile (10 km hr$^{-1}$)", labelpos='S',)

ax2 = plt.subplot(1,2,2, projection=cartopy.crs.PlateCarree())

grid_u_hi = np.zeros((len(gridy),len(gridx)))*np.nan
grid_v_hi = np.zeros((len(gridy),len(gridx)))*np.nan

grid_u_lo = np.zeros((len(gridy),len(gridx)))*np.nan
grid_v_lo = np.zeros((len(gridy),len(gridx)))*np.nan

ihi = vds>np.percentile(vds,75)
ilo = vds<np.percentile(vds,25)


for j, y in enumerate(gridy):
	for i, x in enumerate(gridx):
		it = (lats<=y+r)&(lats>y-r)&(lons>x-r)&(lons<=x+r)
		
		it1 = it&ihi
		it2 = it&ilo
		
		if np.sum(it1)>=5:
			grid_u_hi[j,i]=np.mean(df.zonal_speed.values[it1])
			grid_v_hi[j,i]=np.mean(df.merid_speed.values[it1])

		if np.sum(it2)>=5:
			grid_u_lo[j,i]=np.mean(df.zonal_speed.values[it2])		
			grid_v_lo[j,i]=np.mean(df.merid_speed.values[it2])	



q1 = ax2.quiver(gridx,gridy, grid_u_lo, grid_v_lo, color='tab:blue', scale=s)
q2 = ax2.quiver(gridx,gridy, grid_u_hi, grid_v_hi, color='tab:orange', scale=s)

qk1 = ax1.quiverkey(q1, 0.225, -0.1, 10, label="<25th %ile (10 km hr$^{-1}$)", labelpos='S',)
qk2 = ax1.quiverkey(q2, 0.775, -0.1, 10, label=">75th %ile (10 km hr$^{-1}$)", labelpos='S',)

for ax in [ax1, ax2]:
	ax.add_feature(cfeature.COASTLINE)
	ax.set_xlim([65,95])
	ax.set_ylim([5, 35])
	ax.add_feature(cfeature.BORDERS, linestyle=':',edgecolor='grey')
	gl = ax.gridlines(draw_labels=True, alpha=0)
	gl.top_labels=False
	
	
	if ax==ax1:
		gl.right_labels=False
	
	if ax==ax2:
		gl.left_labels=False


ax1.set_title("(a) 500 hPa zonal wind")
ax2.set_title("(b) vortex depth")


plt.show()
