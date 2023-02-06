import pandas as pd
from netCDF4 import Dataset, MFDataset
from datetime import datetime as dt, timedelta as td
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
import metpy.calc as mpcalc
from metpy.units import units

find = lambda x, arr: np.argmin(np.abs(x-arr))


df = pd.read_csv("temp_data/filtered_lps_with_enviromental.csv")

mean_prcp_400 = []
mean_prcp_800 = []
max_prcp_400 = []
max_prcp_800 = []

mean_vimfd_400 = []
mean_vimfd_800 = []

radius1 = 4
radius2 = 8

for _, row in df.iterrows():
	date = dt(int(row.year), int(row.month), int(row.day), int(row.hour))
		
	prcp_file = Dataset(date.strftime("/home/users/kieran/incompass/users/kieran/era5/hourly_precip_SA/%Y%m.nc"))
	it = int(24*(row.day-1)+row.hour)
	lons = prcp_file.variables['longitude'][:]
	lats = prcp_file.variables['latitude'][:]
	
	ixa1, ixa2 = find(lons, row.x-radius1), find(lons, row.x+radius1)
	iya1, iya2 = find(lats, row.y+radius1), find(lats, row.y-radius1)
	
	ixb1, ixb2 = find(lons, row.x-radius2), find(lons, row.x+radius2)
	iyb1, iyb2 = find(lats, row.y+radius2), find(lats, row.y-radius2)
	
	
	
	prcp400 = prcp_file.variables['mtpr'][it:it+6,iya1:iya2,ixa1:ixa2].mean(axis=0)
	prcp800 = prcp_file.variables['mtpr'][it:it+6,iyb1:iyb2,ixb1:ixb2].mean(axis=0)
	vimfd400 = prcp_file.variables['mvimd'][it,iya1:iya2,ixa1:ixa2]
	
	mean_prcp_400.append(np.mean(prcp400)*3600)
	mean_prcp_800.append(np.mean(prcp800)*3600)
	max_prcp_400.append(np.max(prcp400)*3600)
	max_prcp_800.append(np.max(prcp800)*3600)
	mean_vimfd_400.append(np.mean(vimfd400)*1e5)
	
	print(date, np.mean(prcp400)*3600)


	
df['mean_prcp_400'] = mean_prcp_400
df['mean_prcp_800'] = mean_prcp_800
df['max_prcp_400'] = max_prcp_400
df['max_prcp_800'] = max_prcp_800
df['mean_vimfd_400'] = mean_vimfd_400


df.to_csv("temp_data/filtered_lps_with_enviromental.csv", index=False)
	
