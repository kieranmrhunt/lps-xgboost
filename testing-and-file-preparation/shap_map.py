import cartopy
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

df = pd.read_csv("temp_data/filtered_lps_with_enviromental.csv")
feature_names = np.array(['x','y','month','mean_vort_850','mean_vort_500',
             'mean_swv', 'mean_u200', 'mean_u850', 'mean_skt',
			 'mean_land_frac', 'mcz_tcwv', 'vortex_depth', 'over_land',
             'dvo850_dt','acc_land_time'])
N=3


shap_values = np.load("temp_data/shap_values_6.100.npy")

lons = df.x.values
lats = df.y.values
swv = df.mean_swv.values

swv_shap = shap_values[:,N]

r=1
gridx = np.arange(65,95,r)
gridy = np.arange(5,35,r)

grid_shap_mag = np.zeros((len(gridy),len(gridx)))*np.nan

for j, y in enumerate(gridy):
	for i, x in enumerate(gridx):
		it = (lats<=y+r)&(lats>y-r)&(lons>x-r)&(lons<=x+r)&(swv>0)
		
		if len(it)<5:
			grid_shap_mag[j,i]=np.nan
		else:
			mag = np.mean(np.abs((swv_shap[it])))
			grid_shap_mag[j,i]=mag


		
fig = plt.figure()		
ax  = fig.add_subplot(1,1,1, projection=cartopy.crs.PlateCarree())

cmap = LinearSegmentedColormap.from_list('cmap', ['lightgray','lightgray','gray','yellow','orange','red'])

cs = ax.pcolormesh(gridx,gridy,grid_shap_mag, vmin=0, vmax=5, cmap=cmap)
ax.coastlines()
ax.set_xticks(np.arange(gridx.min(),gridx.max(),5))
ax.set_yticks(np.arange(gridy.min(),gridy.max(),5))
ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)


cb = plt.colorbar(cs, extend='max')
cb.set_label("Coupling coefficient")
plt.title(feature_names[N])
plt.show()
