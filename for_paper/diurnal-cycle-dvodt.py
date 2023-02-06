import pandas as pd
from datetime import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import cartopy.io.shapereader as shpreader
import cartopy
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import seaborn as sns



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


df = pd.read_csv("../temp_data/filtered_lps.csv")
#df = pd.read_csv("/home/users/kieran/work/tracking/era5-lps/tracks/lps_era5_extension.csv")

df = df.sort_values(by=['track_id','frame'])

dvo850_dt = []
land_frac = []
hours = []

for _,track in df.groupby('track_id'):
	v = track.max_trunc_vort.values
	G = np.r_[0,np.diff(v)]*24
	dvo850_dt.extend(G)
	hours.append(len(track))

'''
plt.hist(hours,bins=np.arange(-0.5,120.5,1))
plt.xlabel("Duration (hours)")
plt.ylabel("Count")
plt.show()
'''

loc = np.array([-1,]*len(df))
loc[df['land_frac']>=0.75] = 0
loc[(df['land_frac']<0.75)&(df['land_frac']>0.25)] = 1
loc[df['land_frac']<=0.25] = 2


dvo850_dt = np.array(dvo850_dt)
hods = (df.hour.values+6)%24

colors = ['tab:red', 'tab:green', 'tab:blue']
label = ['(a) land', '(b) coast', '(c) ocean']

def remove_glitch(arr):
	arr[4] = 0.5*(arr[3]+arr[5])
	arr[16] = 0.5*(arr[15]+arr[17])	
	
	return np.array(arr+arr[:1])

fig, axes = plt.subplots(3,1, figsize=(4,6.5), sharex=True, sharey=True)

for L in 0,1,2:
	ax = axes[L]
	
	means = []
	p25, p75 = [], []
	p10, p90 = [], []	

	for h in range(24):
		dvo = dvo850_dt[(loc==L)&(hods==h)]
		means.append(np.mean(dvo))
		p25.append(np.percentile(dvo,25))
		p75.append(np.percentile(dvo,75))
		p10.append(np.percentile(dvo,10))
		p90.append(np.percentile(dvo,90))

	means = remove_glitch(means)
	p25 = remove_glitch(p25)
	p75 = remove_glitch(p75)
	p10 = remove_glitch(p10)
	p90 = remove_glitch(p90)
	
	ax.fill_between(range(25), p10, p90, color=colors[L], alpha=0.2)
	ax.fill_between(range(25), p25, p75, color=colors[L], alpha=0.5)
	ax.plot(range(25), means, c=colors[L])
	
	ax.set_xlim([0,24])
	ax.axhline(0, linestyle=':', color='dimgrey')
	
	ax.text(0.95, 0.9, label[L], ha='right',va='center', transform=ax.transAxes, color=colors[L])

axes[1].set_ylabel("$\partial\zeta_{850}/\partial t$ (10$^{-5}$ s$^{-1}$ day$^{-1}$)")


axes[-1].set_xlabel("Hour of day (local time)")
axes[-1].set_xticks([0,6,12,18,24])

plt.tight_layout()
fig.subplots_adjust(hspace=0)
plt.show()



'''




ax = sns.boxplot(x='hour', y='dvo850_dt', data=df, hue=loc)

plt.show()
'''








