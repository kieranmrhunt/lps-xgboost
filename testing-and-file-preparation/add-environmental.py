import pandas as pd
from netCDF4 import Dataset
from datetime import datetime as dt
import matplotlib.pyplot as plt
import numpy as np


find = lambda x, arr: np.argmin(np.abs(x-arr))

df = pd.read_csv("temp_data/filtered_lps.csv")
df = df[df.hour.isin([0,6,12,18])]
df = df[df.year<2019]

print(df)

radius = 5 #(*0.7*111km)

#model level corresponding to 200mb over ocean is 74
#model level corresponding to 850mb over ocean is 114

#to extract: soil moisture (SWVL1[0]), 200mb u (U[6]), 850mb u (U[22])
# skt(SKT[0]), fraction of system at radius over land
#use SSTK[0] to get land fraction and for mask
#mean tcwv over the central monsoon trough (18.5-27N; 75-85E)

swv, u200, u850, skt, landfrac, tcwv_mcz = [], [], [], [], [], []

for _, row in df.iterrows():
	date = dt(int(row.year), int(row.month), int(row.day), int(row.hour))
	ggasfile = Dataset(date.strftime("/badc/ecmwf-era-interim/data/gg/as/%Y/%m/%d/ggas%Y%m%d%H00.nc"))
	ggapfile = Dataset(date.strftime("/badc/ecmwf-era-interim/data/gg/ap/%Y/%m/%d/ggap%Y%m%d%H00.nc"))

	lons = ggasfile.variables['longitude'][:]
	lats = ggasfile.variables['latitude'][:]
	
	ix, iy = find(lons, row.x), find(lats, row.y)
	
	sst = ggasfile.variables['SSTK'][0,0,iy-radius:iy+radius,ix-radius:ix+radius]
	sst[sst.mask]=0
	sst[sst>0]=1
	
	landfrac.append(np.mean(1-sst))
	
	u200_file = ggapfile.variables['U'][0,22,iy-radius:iy+radius,ix-radius:ix+radius]
	u850_file = ggapfile.variables['U'][0,6,iy-radius:iy+radius,ix-radius:ix+radius]
	
	u200.append(np.mean(u200_file))
	u850.append(np.mean(u850_file))
	
	skt_file = ggasfile.variables['SKT'][0,0,iy-radius:iy+radius,ix-radius:ix+radius]
	skt.append(np.mean(skt_file))
	
	swv_file = ggasfile.variables['SWVL1'][0,0,iy-radius:iy+radius,ix-radius:ix+radius]
	swv_file[sst==1]=np.nan
	swv_mean = np.nanmean(swv_file)
	swv.append(swv_mean if not np.isnan(swv_mean) else 0)
	
	ix1, ix2 = find(lons, 75), find(lons, 85)
	iy1, iy2 = find(lats, 27), find(lats, 18.5)
	tcwv_file = ggasfile.variables['TCWV'][0,0,iy1:iy2,ix1:ix2]
	tcwv_mcz.append(np.mean(tcwv_file))

	print(date, swv[-1], u200[-1], u850[-1], skt[-1], landfrac[-1], tcwv_mcz[-1])


df['mean_swv'] = swv
df['mean_u200'] = u200
df['mean_u850'] = u850
df['mean_skt'] = skt
df['mean_land_frac'] = landfrac
df['mcz_tcwv'] = tcwv_mcz

	
df.to_csv("temp_data/filtered_lps_with_enviromental.csv")

	





