import pandas as pd
from netCDF4 import Dataset
from datetime import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import fiona
import cartopy.io.shapereader as shpreader
import shapely.geometry as sgeom
import shapely.vectorized as svec
import cartopy
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import seaborn as sns


geoms = fiona.open(shpreader.natural_earth(resolution='50m',category='physical',name='land'))
land_geom = sgeom.MultiPolygon([sgeom.shape(geom['geometry'])for geom in geoms])

def haversine(lon1_, lat1_, lon2_, lat2_):
	lon1=lon1_*np.pi/180.
	lat1=lat1_*np.pi/180.
	lon2=lon2_*np.pi/180.
	lat2=lat2_*np.pi/180.
	dlon = lon2 - lon1 
	dlat = lat2 - lat1 
	a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
	c = 2 * np.arcsin(np.sqrt(a)) 
	r = 6371.
	return c * r 


radius= 400 #km

find = lambda x, arr: np.argmin(np.abs(x-arr))

df = pd.read_csv("temp_data/filtered_lps.csv")
#df = pd.read_csv("/home/users/kieran/work/tracking/era5-lps/tracks/lps_era5_extension.csv")

df = df.sort_values(by=['track_id','frame'])

dvo850_dt = []
land_frac = []
hours = []

for _,track in df.groupby('track_id'):
	v = track.max_trunc_vort.values
	G = np.gradient(v)
	dvo850_dt.extend(G)
	hours.append(len(track))

'''
plt.hist(hours,bins=np.arange(-0.5,120.5,1))
plt.xlabel("Duration (hours)")
plt.ylabel("Count")
plt.show()
'''

dvo850_dt = np.array(dvo850_dt)
print(np.nanstd(dvo850_dt))


'''
gridx, gridy = np.meshgrid(np.arange(0,150,0.5),np.arange(-10,50,0.5))
is_land = svec.contains(land_geom,gridx,gridy).astype(int)

for _, row in df.iterrows():
	H = haversine(row.x, row.y, gridx, gridy)<=radius
	land_frac.append(np.mean(is_land[H]))

land_frac = np.array(land_frac)


df['dvo850_dt'] = dvo850_dt
df['land_frac'] = land_frac
#df.to_csv("temp_data/filtered_lps.csv", index=False)
'''


#df = pd.read_csv("temp_data/filtered_lps.csv")
loc = np.array(["",]*len(df))

loc[df['land_frac']>=0.75] = 'land'
loc[(df['land_frac']<0.75)&(df['land_frac']>0.25)] = 'coast'
loc[df['land_frac']<=0.25] = 'sea'



ax = sns.boxplot(x='hour', y='dvo850_dt', data=df, hue=loc)

plt.show()





