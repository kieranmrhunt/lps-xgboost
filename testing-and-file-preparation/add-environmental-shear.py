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


#to add: mean magnitude of sm gradient L1/L2, swvl2, CAPE, OLR percentile, dtheta/dz for different layers,
#q/u shear more reflective of background state; fix swvl1; separate land temp and SST

radius = 4
box_height = 17
box_width = 10


mean_cape, mcz_cape, mean_dthetae_dp_900_750, mean_dthetae_dp_750_500 = [], [], [], []  
mean_land_skt, mean_sst = [], []
mean_swvl1_grad, mean_swvl2_grad, mean_swvl1, mean_swvl2 = [], [], [], []
olr_90, olr_75, olr_50 = [], [], []
qshear_850_background, u_shear_850_background = [], []
qshear_850, ushear_850 = [], []
mean_q_850 = []
orography_height = []

for _, row in df.iterrows():
	date = dt(int(row.year), int(row.month), int(row.day), int(row.hour))
	'''
	cape_file = Dataset(date.strftime("/badc/ecmwf-era5/data/oper/an_sfc/%Y/%m/%d/ecmwf-era5_oper_an_sfc_%Y%m%d%H00.cape.nc"))
	lons = cape_file.variables['longitude'][:]
	lats = cape_file.variables['latitude'][:]
	ixb1, ixb2 = find(lons, 75), find(lons, 85)
	iyb1, iyb2 = find(lats, 27), find(lats, 18.5)
	cape_box = cape_file.variables['cape'][0,iyb1:iyb2,ixb1:ixb2]
	mcz_cape.append(cape_box.mean())
	ix1, ix2 = find(lons, row.x-radius), find(lons, row.x+radius)
	iy1, iy2 = find(lats, row.y+radius), find(lats, row.y-radius)
	cape_lps = cape_file.variables['cape'][0,iy1:iy2,ix1:ix2]
	mean_cape.append(cape_lps.mean())
	
	oro_file = Dataset("/badc/ecmwf-era5/data/invariants/ecmwf-era5_oper_an_sfc_200001010000.z.inv.nc")
	ix, iy = find(lons, row.x), find(lats, row.y)
	z = oro_file.variables['z'][0,iy,ix]/9.8
	orography_height.append(z)
	'''
	
	olr_file = Dataset(date.strftime("/home/users/kieran/incompass/users/kieran/era5/6hrly_olr_SA/%Y%m.nc"))
	it = int(4*(row.day-1)+row.hour/6)
	lons = olr_file.variables['longitude'][:]
	lats = olr_file.variables['latitude'][:]
	ix1, ix2 = find(lons, row.x-radius), find(lons, row.x+radius)
	iy1, iy2 = find(lats, row.y+radius), find(lats, row.y-radius)
	olr = olr_file.variables['mtnlwrf'][it,iy1:iy2,ix1:ix2]
	olr_50.append(np.percentile(olr, 50))
	olr_75.append(np.percentile(olr, 25))
	olr_90.append(np.percentile(olr, 10))
	print(date)
	
	'''
	iz1, iz2, iz3 = 4, 10, 15
	iz = np.array([iz1, iz2, iz3])
	ggapfile = Dataset(date.strftime("/badc/ecmwf-era-interim/data/gg/ap/%Y/%m/%d/ggap%Y%m%d%H00.nc"))
	lons = ggapfile.variables['longitude'][:]
	lats = ggapfile.variables['latitude'][:]
	ix1, ix2 = find(lons, row.x-radius), find(lons, row.x+radius)
	iy1, iy2 = find(lats, row.y+radius), find(lats, row.y-radius)
	
	
	q = np.array(ggapfile.variables['Q'][0, iz, iy1:iy2, ix1:ix2])*units("kg kg^-1")
	t = np.array(ggapfile.variables['T'][0, iz, iy1:iy2, ix1:ix2])*units.kelvin
	p = np.array([900,750,500])[:,None,None]*units("hPa")
	dewpt = mpcalc.dewpoint_from_specific_humidity(p, t, q)
	theta_e = mpcalc.equivalent_potential_temperature(p, t, dewpt)
	theta_e = np.mean(theta_e, axis=(-1,-2)).magnitude
	mean_dthetae_dp_900_750.append((theta_e[0]-theta_e[1])/150)
	mean_dthetae_dp_750_500.append((theta_e[1]-theta_e[2])/150)	
	q850 = ggapfile.variables['Q'][0, 6, iy1:iy2, ix1:ix2]
	mean_q_850.append(np.mean(q850))
	
	ggasfile = Dataset(date.strftime("/badc/ecmwf-era-interim/data/gg/as/%Y/%m/%d/ggas%Y%m%d%H00.nc"))
	sst = ggasfile.variables['SSTK'][0,0,iy1:iy2, ix1:ix2]
	sst[sst.mask]=np.nan
	mean_sst.append(np.nanmean(sst))
	skt = ggasfile.variables['SKT'][0,0,iy1:iy2, ix1:ix2]
	skt[~np.isnan(sst)]=np.nan
	mean_land_skt.append(np.nanmean(skt))
	
	swv1 = ggasfile.variables['SWVL1'][0,0,iy1:iy2, ix1:ix2]
	swv2 = ggasfile.variables['SWVL2'][0,0,iy1:iy2, ix1:ix2]
	swv1[~np.isnan(sst)]=np.nan
	swv2[~np.isnan(sst)]=np.nan
	dx, dy = mpcalc.lat_lon_grid_deltas(lons[ix1:ix2], lats[iy1:iy2])
	
	swv1_grad = np.hypot(*mpcalc.gradient(swv1, deltas=[dy,dx])).magnitude
	swv2_grad = np.hypot(*mpcalc.gradient(swv2, deltas=[dy,dx])).magnitude
	mean_swvl1_grad.append(np.nanmean(swv1_grad))
	mean_swvl2_grad.append(np.nanmean(swv2_grad))
	mean_swvl1.append(np.nanmean(swv1))
	mean_swvl2.append(np.nanmean(swv2))
	
	
	
	previous_ten_days = [date-td(days=i) for i in range(-10,1)]
	bg_file = MFDataset([d.strftime("/badc/ecmwf-era-interim/data/gg/ap/%Y/%m/%d/ggap%Y%m%d%H00.nc") for d in previous_ten_days])
	
	iy1, iy2 = find(lats, 27), find(lats, 10)
	ix1, ix2 = find(lons, row.x-box_width/2), find(lons, row.x+box_width/2)
	u = bg_file.variables['U'][:,6,iy1:iy2, ix1:ix2]
	q = bg_file.variables['Q'][:,6,iy1:iy2, ix1:ix2]
	uy = u.mean(axis=-1) 
	qy = q.mean(axis=-1)
	uslope = uy[-1,0]-uy[-1,-1]
	qslope = qy[-1,0]-qy[-1,-1]
	ubgslope = np.mean(uy[:,0]-uy[:,-1])
	qbgslope = np.mean(qy[:,0]-qy[:,-1])
	qshear_850_background.append(qbgslope)
	u_shear_850_background.append(ubgslope)
	qshear_850.append(qslope)
	ushear_850.append(uslope)
	
	print(date, row.x,row.y, (theta_e[0]-theta_e[1])/150, ubgslope, np.nanmean(sst), np.nanmean(swv1_grad))
	'''
	
'''
df['mean_cape'] = mean_cape
df['mcz_cape'] = mcz_cape
df['mean_dthetae_dp_900_750'] = mean_dthetae_dp_900_750
df['mean_dthetae_dp_750_500'] = mean_dthetae_dp_750_500
df['mean_land_skt'] = mean_land_skt
df['mean_sst'] = mean_sst
df['mean_swvl1_grad'] = mean_swvl1_grad
df['mean_swvl2_grad'] = mean_swvl2_grad
df['mean_swvl1'] = mean_swvl1
df['mean_swvl2'] = mean_swvl2
'''
df['olr_90'] = olr_90
df['olr_75'] = olr_75
df['olr_50'] = olr_50

'''
df['qshear_850_background'] = qshear_850_background
df['u_shear_850_background'] = u_shear_850_background
df['qshear_850'] = qshear_850
df['ushear_850'] = ushear_850
df['mean_q_850'] = mean_q_850
df['orography_height'] = orography_height
'''

df.to_csv("temp_data/filtered_lps_with_enviromental.csv", index=False)



'''
#to add: meridional gradient in q850, meridional gradient in u850

box_height = 17
box_width = 10

qshear_850, ushear_850 = [], []

for _, row in df.iterrows():
	
	date = dt(int(row.year), int(row.month), int(row.day), int(row.hour))
	ggapfile = Dataset(date.strftime("/badc/ecmwf-era-interim/data/gg/ap/%Y/%m/%d/ggap%Y%m%d%H00.nc"))
	lons = ggapfile.variables['longitude'][:]
	lats = ggapfile.variables['latitude'][:]

	iy1, iy2 = find(lats, 27), find(lats, 10)
	ix1, ix2 = find(lons, row.x-box_width/2), find(lons, row.x+box_width/2)
	
	u = ggapfile.variables['U'][0,6,iy1:iy2, ix1:ix2]
	q = ggapfile.variables['Q'][0,6,iy1:iy2, ix1:ix2]
	
	uy = u.mean(axis=-1) 
	qy = q.mean(axis=-1)
	
	qslope = linregress(lats[iy1:iy2], qy)[0]
	uslope = linregress(lats[iy1:iy2], uy)[0]
	
	qshear_850.append(qslope)
	ushear_850.append(uslope)
	
	print(date, uslope, qslope)
	

df['qshear_850'] = qshear_850
df['ushear_850'] = ushear_850

df.to_csv("temp_data/filtered_lps_with_enviromental.csv", index=False)
'''

