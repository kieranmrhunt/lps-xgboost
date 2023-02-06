import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import linregress
import shap
from statsmodels.nonparametric.smoothers_lowess import lowess as  sm_lowess
import cmath
from matplotlib.colors import LinearSegmentedColormap
import cartopy
import cartopy.feature as cfeature
from statsmodels.nonparametric.smoothers_lowess import lowess as  sm_lowess

df = pd.read_csv("../temp_data/filtered_lps_with_enviromental.csv")

#df = df[df['reached_peak']==0]
#print(df.columns)
'''
['x', 'y', 'frame', 'year', 'month', 'day', 'hour', 'radius', 'max_vort',
       'mean_vort', 'mean_vort_850', 'mean_vort_700', 'mean_vort_500',
       'max_trunc_vort', 'track_id', 'mean_swv', 'mean_u200', 'mean_u850',
       'mean_skt', 'mean_land_frac', 'mcz_tcwv', 'vortex_depth', 'over_land',
       'dvo850_dt', 'acc_land_time', 'total_land_time', 'qshear_850',
       'ushear_850', 'mean_cape', 'mcz_cape', 'mean_dthetae_dp_900_750',
       'mean_dthetae_dp_750_500', 'mean_land_skt', 'mean_sst',
       'mean_swvl1_grad', 'mean_swvl2_grad', 'mean_swvl1', 'mean_swvl2',
       'olr_90', 'olr_75', 'olr_50', 'qshear_850_background',
       'ushear_850_background', 'mean_q_850', 'orography_height',
       'peak_vorticity', 'reached_peak', 'mean_prcp_400', 'mean_prcp_800',
       'max_prcp_400', 'max_prcp_800', 'mean_vimfd_400', 'mean_v200',
       'mean_v500', 'mean_v850', 'mean_u500', 'zonal_speed', 'merid_speed']
'''

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


X = df[feature_names]


pvar = 'peak_vorticity'
y = df[[pvar]]

data_dmatrix = xgb.DMatrix(data=X,label=y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


params = {'objective':'reg:squarederror','colsample_bytree': 0.3,'learning_rate': 0.25,
          'max_depth': 6, 'alpha': 10, 'gamma' : 0}
n_estimators = 1000



xg_reg = xgb.XGBRegressor(n_estimators = n_estimators, 
                          **params)
xg_reg.fit(X_train,y_train, early_stopping_rounds=20, eval_set = [(X_test, y_test)])
preds = xg_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
sn = rmse/np.std(y[pvar].values)
print("RMSE: %f" % (rmse))

print(linregress(np.array(y_test[pvar].values),np.array(preds)))



explainer = shap.TreeExplainer(xg_reg)
shap_values = explainer.shap_values(X)

cmap = LinearSegmentedColormap.from_list('cmap', ['lightgray','lightgray','gray','yellow','orange','red'])

lons = df.x.values
lats = df.y.values

df_r = df[df.reached_peak==True].groupby('track_id').first()
sm_y, sm_x = sm_lowess(df_r.x, df_r.y,  frac=1./2., it=5, return_sorted = True).T


plt.figure(figsize=(10,5))
ax1 = plt.subplot(1,2,1, projection=cartopy.crs.PlateCarree())

ix = np.where(feature_names=='mean_vort_500')[0][0]
shap = shap_values[:,ix]

r=1
gridx = np.arange(65,95,r)
gridy = np.arange(5,35,r)

grid_shap_mag = np.zeros((len(gridy),len(gridx)))*np.nan

for j, y in enumerate(gridy):
	for i, x in enumerate(gridx):
		it = (lats<=y+r)&(lats>y-r)&(lons>x-r)&(lons<=x+r)
		
		if np.sum(it)<5:
			grid_shap_mag[j,i]=np.nan
		else:
			grid_shap_mag[j,i]=np.mean(np.abs(shap[it]))


cs1 = ax1.pcolormesh(gridx,gridy,grid_shap_mag, vmin=0, vmax=.333, cmap=cmap)



ax2 = plt.subplot(1,2,2, projection=cartopy.crs.PlateCarree())
fname = 'ushear_850_background'
ix = np.where(feature_names==fname)[0][0]
shap = shap_values[:,ix]

r=1
gridx = np.arange(65,95,r)
gridy = np.arange(5,35,r)

grid_shap_mag = np.zeros((len(gridy),len(gridx)))*np.nan

for j, y in enumerate(gridy):
	for i, x in enumerate(gridx):
		it = (lats<=y+r)&(lats>y-r)&(lons>x-r)&(lons<=x+r)
		
		if np.sum(it)<5:
			grid_shap_mag[j,i]=np.nan
		else:
			grid_shap_mag[j,i]=np.mean(np.abs(shap[it]))


cs2 = ax2.pcolormesh(gridx,gridy,grid_shap_mag, vmin=0, vmax=.333, cmap=cmap)

ax1.plot(sm_x, sm_y, 'k:',)
ax2.plot(sm_x, sm_y, 'k:',)

for ax in [ax1, ax2]:
	ax.add_feature(cfeature.COASTLINE)
	ax.set_xlim([60,100])
	ax.set_ylim([5, 40])
	ax.add_feature(cfeature.BORDERS, linestyle=':',edgecolor='grey')
	gl = ax.gridlines(draw_labels=True, alpha=0)
	gl.top_labels=False
	
	
	if ax==ax1:
		gl.right_labels=False
	
	if ax==ax2:
		gl.left_labels=False

ax1.set_title("(a) 500 hPa relative vorticity")
ax2.set_title("(b) meridional shear of\nbackground 850 hPa zonal wind")

fig = plt.gcf()
fig.subplots_adjust(top=1, wspace=.1, bottom=0.2)
x11,x12 =  ax1.get_position().x0, ax1.get_position().x1 
x21,x22 =  ax2.get_position().x0, ax2.get_position().x1
x1 = 0.5*(x11+x12)
x2 = 0.5*(x21+x22)

cax1 = fig.add_axes([x1, 0.15, x2-x1, 0.05])
cb1 = fig.colorbar(cs1, cax=cax1, orientation='horizontal', extend='max')
cb1.set_label("Mean absolute Shapley value (10$^{-5}$ s$^{-1}$)")


plt.show()

