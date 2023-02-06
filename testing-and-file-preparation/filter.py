import matplotlib.pyplot as plt
import cartopy
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter as gf
import fiona
import cartopy.io.shapereader as shpreader
import shapely.geometry as sgeom
import shapely.vectorized as svec
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

geoms = fiona.open(shpreader.natural_earth(resolution='50m',category='physical',name='land'))
land_geom = sgeom.MultiPolygon([sgeom.shape(geom['geometry'])for geom in geoms])



df = pd.read_csv("/home/users/kieran/work/tracking/era5-lps/tracks/lps_era5_extension.csv")
df = df[df.month.isin([6,7,8,9])]

df_genesis = df.groupby("track_id").first()
df_genesis = df_genesis[(~svec.contains(land_geom,df_genesis.x,df_genesis.y))
						& (df_genesis.x>78) & (df_genesis.x<95)
						& (df_genesis.y>10)].reset_index()



df_lysis = df.groupby("track_id").last()
df_lysis = df_lysis[(svec.contains(land_geom,df_lysis.x,df_lysis.y))
						& (df_lysis.x<90) & (df_lysis.x>60)
						& (df_lysis.y>10)].reset_index()


df = df[df.track_id.isin(df_genesis.track_id) & df.track_id.isin(df_lysis.track_id)]
print(df)


ax = plt.subplot(111,projection = cartopy.crs.PlateCarree())

i=0
for _, t in df.groupby('track_id'):
	ax.plot(t.x, t.y, 'k-', lw=0.5, alpha=0.5)
	ax.plot(t.x.values[0], t.y.values[0], 'ro', ms=2, zorder=5)
	i+=1
print(i)

ax.coastlines()
ax.add_feature(cartopy.feature.BORDERS)
ax.set_xticks(np.arange(30,120,5))
ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
ax.set_yticks(np.arange(0,45,5))
ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)

ax.set_xlim([65, 95])
ax.set_ylim([5, 32])

#df.to_csv("temp_data/filtered_lps.csv", index=False)

plt.show()

