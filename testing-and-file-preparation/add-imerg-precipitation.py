import pandas as pd
from netCDF4 import Dataset, MFDataset
from datetime import datetime as dt, timedelta as td
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
import glob

find = lambda x, arr: np.argmin(np.abs(x-arr))


df = pd.read_csv("temp_data/filtered_lps_with_enviromental.csv")

mean_prcp_400 = []


radius1 = 4
stem = '/home/users/rz908899/cluster/mr806421/datasets/'

for _, row in df.iterrows():
	date = dt(int(row.year), int(row.month), int(row.day), int(row.hour))
	if date.year<2000: 
		mean_prcp_400.append(np.nan)
		continue
		
	pslice = []
	dlist = [date+td(minutes=30*i) for i in range(12)]
	try:
		flist = [glob.glob(d.strftime(stem+"IMERG/%Y/3B-HHR.MS.MRG.3IMERG.%Y%m%d-S%H%M%S*"))[0] for d in dlist]
	except: 
		mean_prcp_400.append(np.nan)
		continue
	
	for f in flist:
		prcp_file = Dataset(f)
		lons = prcp_file.variables['lon'][:]
		lats = prcp_file.variables['lat'][:]
		
		ixa1, ixa2 = find(lons, row.x-radius1), find(lons, row.x+radius1)
		iya1, iya2 = find(lats, row.y-radius1), find(lats, row.y+radius1)
			
		prcp400 = prcp_file.variables['precipitationCal'][0,ixa1:ixa2,iya1:iya2]
		pslice.append(prcp400)
	
	mean_prcp_400.append(np.nanmean(pslice))
	
	
	print(date, np.nanmean(prcp400))


	
df['mean_prcp_imerg'] = mean_prcp_400


df.to_csv("temp_data/filtered_lps_with_enviromental.csv", index=False)
	
