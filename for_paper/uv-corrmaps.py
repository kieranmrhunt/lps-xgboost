import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import cmath
from matplotlib.colors import LinearSegmentedColormap
import cartopy
import cartopy.feature as cfeature


df = pd.read_csv("../temp_data/filtered_lps_with_enviromental.csv")
df['dvo850_dt'] = df['dvo850_dt']*4


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


params = {'objective':'reg:squarederror','colsample_bytree': 0.3,'learning_rate': 0.25,
          'max_depth': 6, 'alpha': 10, 'gamma' : 5}
n_estimators = 100

X = df[feature_names]

yu = df[['zonal_speed']]
yv = df[['merid_speed']]


cmap = LinearSegmentedColormap.from_list('cmap', ['lightgray','lightgray','gray','yellow','orange','red'])

lons = df.x.values
lats = df.y.values

plt.figure(figsize=(10,5))
ax1 = plt.subplot(1,2,1, projection=cartopy.crs.PlateCarree())
fname = 'mean_u500'

lpsu = df.zonal_speed.values
u500 = df.mean_u500.values

r=1
gridx = np.arange(65,95,r)
gridy = np.arange(5,35,r)

grid_corru = np.zeros((len(gridy),len(gridx)))*np.nan

for j, y in enumerate(gridy):
	for i, x in enumerate(gridx):
		it = (lats<=y+r)&(lats>y-r)&(lons>x-r)&(lons<=x+r)

		if np.sum(it)<5:
			grid_corru[j,i]=np.nan
		else:
			R,p = pearsonr(lpsu[it], u500[it])
			grid_corru[j,i] = R**2


cs1 = ax1.pcolormesh(gridx,gridy,grid_corru, vmin=0,vmax=0.5,  cmap=cmap)



ax2 = plt.subplot(1,2,2, projection=cartopy.crs.PlateCarree())
fname = 'mean_v500'

lpsv = df.merid_speed.values
v500 = df.mean_v500.values

grid_corrv = np.zeros((len(gridy),len(gridx)))*np.nan

for j, y in enumerate(gridy):
	for i, x in enumerate(gridx):
		it = (lats<=y+r)&(lats>y-r)&(lons>x-r)&(lons<=x+r)
		
		if np.sum(it)<5:
			grid_corrv[j,i]=np.nan
		else:
			R,p = pearsonr(lpsv[it], v500[it])
			grid_corrv[j,i] = R**2

cs2 = ax2.pcolormesh(gridx,gridy,grid_corrv, vmin=0,vmax=0.5,cmap=cmap)


for ax in [ax1, ax2]:
	ax.add_feature(cfeature.COASTLINE)
	ax.set_xlim([60,100])
	ax.set_ylim([5, 40])
	ax.add_feature(cfeature.BORDERS, linestyle=':',edgecolor='grey')
	gl = ax.gridlines(draw_labels=True, alpha=0)
	gl.top_labels=False
	
	
	if ax==ax1:
		gl.right_labels=False
	
	if ax==ax2:
		gl.left_labels=False




plt.show()
