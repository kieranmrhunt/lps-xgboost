import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import linregress
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import interp1d
from matplotlib import patheffects as pe

feature_names = np.array(['x', 'y', 'hour', 'month',
       'mean_vort_850', 'mean_vort_500', 'vortex_depth',
       'mean_u200', 'mean_u500', 'mean_u850',
       'mean_v200', 'mean_v500', 'mean_v850',
       'mean_skt', 'mean_land_frac', 'mcz_tcwv',
       'mean_cape', 'mcz_cape', 'mean_dthetae_dp_900_750',
       'mean_dthetae_dp_750_500', 'mean_land_skt', 'mean_sst',
       'mean_swvl1_grad', 'mean_swvl2_grad', 'mean_swvl1', 'mean_swvl2',
       'olr_90', 'olr_50', 'qshear_850_background',  'dvo850_dt',
       'ushear_850_background', 'mean_q_850', 'orography_height',])


df = pd.read_csv("../temp_data/filtered_lps_with_enviromental.csv")
df['dvo850_dt'] = df['dvo850_dt']*4

dvo_dt = df.dvo850_dt.values
vo500 = df.mean_vort_500.values
pr_mean = df.mean_prcp_400.values
pr_max =  df.max_prcp_400.values

dv_life = []
vo_life = []
peaks = []

for _,track in df.groupby('track_id'):
	old_t = np.linspace(0,1,len(track),endpoint=True)
	new_t = np.linspace(0,1,51,endpoint=True)
	
	d = interp1d(old_t, track.dvo850_dt)(new_t)
	v = interp1d(old_t, track.mean_vort_500)(new_t)
	
	dv_life.append(d)
	vo_life.append(v)
	peaks.append(np.max(track.mean_vort_850))

dv_life = np.array(dv_life)
vo_life = np.array(vo_life)
peaks = np.array(peaks)
it = peaks>np.percentile(peaks, 90)

cmap = LinearSegmentedColormap.from_list('cmap', ['lightgray','lightgray','gray','yellow','orange','red'])

fig, ax = plt.subplots(1,1, figsize=(11,5))
sc = ax.scatter(vo500, dvo_dt, c=pr_mean, cmap=cmap, s=pr_max, vmin=0, vmax=2.5)
handles, labels = sc.legend_elements("sizes", num=4)
L = plt.legend(handles, labels, ncol=2, loc='lower right', title='Max precip (mm hr$^{-1}$)')
ax.add_artist(L)

l1 = ax.plot(np.mean(vo_life, axis=0), np.mean(dv_life, axis=0), path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()], color='lightskyblue', label='All LPS mean')
l2 = ax.plot(np.mean(vo_life[it], axis=0), np.mean(dv_life[it], axis=0), path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()], color='tab:blue', label='90th %ile LPS mean')
plt.legend(loc='lower left')

ax.axhline(0, linestyle=':', color='k')
ax.axvline(np.median(vo500), linestyle=':', color='k')

ax.set_ylabel("$\partial\zeta_{850}/\partial t$ (10$^{-5}$ s$^{-1}$ day$^{-1}$)")
ax.set_xlabel("500 hPa vorticity (10$^{-5}$ s$^{-1}$)")



cb = plt.colorbar(sc, extend='max')
cb.set_label("Mean precipitation within 400 km (mm hr$^{-1}$)")
plt.show()



