import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import linregress
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter

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
pv = np.load("../temp_data/composite-lps-vertical-PV.npy")*1e7

shap_values = np.load("../temp_data/shap_values_zonal_speed.npy")

span = 10
resolution = 0.5

xgrid = np.linspace(-span,span,int(2*span/resolution))
zgrid = np.linspace(100,1025,25)


fig, axes = plt.subplots(1,3, figsize=(10,6), sharey=True, sharex=True)
ax1, ax2, ax3 = axes

vds = df.vortex_depth.values

pv_hi = np.mean(pv[vds>np.percentile(vds,75)], axis=0)
pv_lo = np.mean(pv[vds<np.percentile(vds,25)], axis=0)

cs1 =ax1.contourf(xgrid, zgrid, np.mean(pv, axis=0), levels=np.arange(0,10,1), extend='max')
ax1.contour(xgrid, zgrid, np.mean(pv, axis=0), levels=np.arange(0,15,1), colors='k', linewidths=0.5)

cs2 =ax2.contourf(xgrid, zgrid, pv_hi-pv_lo, levels=np.arange(-3,3.25,0.25), extend='both', cmap=plt.cm.RdBu)
ax2.contour(xgrid, zgrid, pv_hi-pv_lo, levels=np.arange(-4,4.25,0.25), colors='k', linewidths=0.5)

ix1 = np.where(feature_names=='vortex_depth')[0][0]
ix2 = np.where(feature_names=='mean_u850')[0][0]
s1 = shap_values[:,ix1]
s2 = shap_values[:,ix2]

rarr1 = np.empty((len(zgrid), len(xgrid)))*np.nan
parr1 = np.empty((len(zgrid), len(xgrid)))*np.nan
rarr2 = np.empty((len(zgrid), len(xgrid)))*np.nan
parr2 = np.empty((len(zgrid), len(xgrid)))*np.nan

for i in range(len(xgrid)):
	for j in range(len(zgrid)):
		try:
			S,I,R,P,E = linregress(s1, pv[:,j,i])
			rarr1[j,i] = R
			parr1[j,i] = P
			S,I,R,P,E = linregress(s2, pv[:,j,i])
			rarr2[j,i] = R
			parr2[j,i] = P
		except:
			rarr1[j,i] = 0
			parr1[j,i] = 1
			rarr2[j,i] = 0
			parr2[j,i] = 1

cs = ax3.contour(xgrid, zgrid, rarr1, levels=np.arange(-1,1,0.1), extend='both', colors='k')
cs3 = ax3.contourf(xgrid, zgrid, rarr2, levels=np.arange(-0.3,0.35,0.05), extend='both', cmap=plt.cm.RdBu_r)



ax1.set_ylim([200,975])
ax1.invert_yaxis()
ax1.set_ylabel("Pressure (hPa)")

for ax in axes:
	ax.set_xlabel("Relative longitude")
	ax.xaxis.set_major_formatter(FormatStrFormatter(u'%d\u00B0'))

ax1.set_title("(a) all LPSs")
ax2.set_title("(b) top quartile minus bottom\nquartile vortex depth")
ax3.set_title("(c) Shapley value correlations")

fig.subplots_adjust(wspace=.125, bottom=0.25)
x11,x12 =  ax1.get_position().x0, ax1.get_position().x1 
x21,x22 =  ax2.get_position().x0, ax2.get_position().x1
x31,x32 =  ax3.get_position().x0, ax3.get_position().x1

cax1 = fig.add_axes([x11, 0.1, x12-x11, 0.05])
cb1 = fig.colorbar(cs1, cax=cax1, orientation='horizontal')
cb1.set_label("Potential vorticity (10$^{-7}$ m$^2$ s$^{-1}$ K kg$^{-1}$)")

cax2 = fig.add_axes([x21, 0.1, x22-x21, 0.05])
cb2 = fig.colorbar(cs2, cax=cax2, orientation='horizontal', ticks=(-3,-2,-1,0,1,2,3))
cb2.set_label("PV difference (10$^{-7}$ m$^2$ s$^{-1}$ K kg$^{-1}$)")

cax3 = fig.add_axes([x31, 0.1, x32-x31, 0.05])
cb3 = fig.colorbar(cs3, cax=cax3, orientation='horizontal',)# ticks=(-2,-1,0,1,2))
cb3.set_label("Correlation coefficient")

ax3.clabel(cs, cs.levels, inline=True, fontsize=10, manual=True, inline_spacing=1)

plt.show()




