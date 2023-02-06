import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import fiona
import cartopy.io.shapereader as shpreader
import shapely.geometry as sgeom
import shapely.vectorized as svec

geoms = fiona.open(shpreader.natural_earth(resolution='50m',category='physical',name='land'))
land_geom = sgeom.MultiPolygon([sgeom.shape(geom['geometry'])for geom in geoms])

#to add: centre_over_land [0,1]; vortex depth (vo500*vo700/vo_850**2); dvo_dt (change in vo_850 between successive 6hrs)

df = pd.read_csv("temp_data/filtered_lps_with_enviromental.csv")
#df = pd.read_csv("/home/users/kieran/work/tracking/era5-lps/tracks/lps_era5_extension.csv")
df.sort_values(by=['track_id','frame'],inplace=True)
#df.drop(columns=['Unnamed: 0'], inplace=True)

'''
df['vortex_depth'] = df['mean_vort_500']*df['mean_vort_700']/(df['mean_vort_850'])**2
df['over_land'] = svec.contains(land_geom,df.x,df.y).astype(int)
dvo850_dt = []
total_land_time = []
acc_land_time = []
'''

zonal_speed = []
merid_speed = []

for _,track in df.groupby('track_id'):
	x = track.x.values
	y = track.y.values
	
	dlon = np.gradient(x)
	dlat = np.gradient(y)
	
	dx = dlon*111*np.cos(y*np.pi/180)
	dy = dlat*111
	
	zonal_speed.extend(dx/6)
	merid_speed.extend(dy/6)
	
	
	'''
	v = track.max_trunc_vort.values
	G = np.gradient(v)
	dvo850_dt.extend(G)
	
	L = track.over_land.values
	acc = np.cumsum(L)*6
	fin = np.ones_like(acc)*acc[-1]
	acc_land_time.extend(acc)
	total_land_time.extend(fin)
	'''
	

#df['dvo850_dt'] = dvo850_dt
#df['acc_land_time'] = acc_land_time
#df['total_land_time'] = total_land_time

df['zonal_speed'] = zonal_speed
df['merid_speed'] = merid_speed


print(df)
df.to_csv("temp_data/filtered_lps_with_enviromental.csv", index=False)
