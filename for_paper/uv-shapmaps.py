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


df = pd.read_csv("../temp_data/filtered_lps_with_enviromental.csv")
df['dvo850_dt'] = df['dvo850_dt']*4

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
       'olr_90', 'olr_50', 'qshear_850_background',
       'ushear_850_background', 'mean_q_850', 'orography_height',])


params = {'objective':'reg:squarederror','colsample_bytree': 0.3,'learning_rate': 0.25,
          'max_depth': 6, 'alpha': 10, 'gamma' : 5}
n_estimators = 100

X = df[feature_names]

yu = df[['zonal_speed']]
yv = df[['merid_speed']]


data_dmatrixu = xgb.DMatrix(data=X,label=yu)
X_train, X_test, y_trainu, y_testu = train_test_split(X, yu, test_size=0.2, random_state=123)
xg_regu = xgb.XGBRegressor(n_estimators = n_estimators, **params)
xg_regu.fit(X_train,y_trainu, early_stopping_rounds=20, eval_set = [(X_test, y_testu)])
predsu = xg_regu.predict(X_test)
rmseu = np.sqrt(mean_squared_error(y_testu, predsu))
print(linregress(np.array(y_testu['zonal_speed'].values),np.array(predsu)))


data_dmatrixv = xgb.DMatrix(data=X,label=yv)
X_train, X_test, y_trainv, y_testv = train_test_split(X, yv, test_size=0.2, random_state=123)
xg_regv = xgb.XGBRegressor(n_estimators = n_estimators, **params)
xg_regv.fit(X_train,y_trainv, early_stopping_rounds=20, eval_set = [(X_test, y_testv)])
predsv = xg_regv.predict(X_test)
rmsev = np.sqrt(mean_squared_error(y_testv, predsv))
print(linregress(np.array(y_testv['merid_speed'].values),np.array(predsv)))




explaineru = shap.TreeExplainer(xg_regu)
shap_valuesu = explaineru.shap_values(X)

explainerv = shap.TreeExplainer(xg_regv)
shap_valuesv = explainerv.shap_values(X)

cmap = LinearSegmentedColormap.from_list('cmap', ['lightgray','lightgray','gray','yellow','orange','red'])

lons = df.x.values
lats = df.y.values

plt.figure(figsize=(10,5))
ax1 = plt.subplot(1,2,1, projection=cartopy.crs.PlateCarree())

ix = np.where(feature_names=='mean_u500')[0][0]
iy = np.where(feature_names=='mean_u850')[0][0]

lpsu = df.zonal_speed.values
shap5 = shap_valuesu[:,ix]
shap8 = shap_valuesu[:,iy]

r=1
gridx = np.arange(65,95,r)
gridy = np.arange(5,35,r)

grid_shap_magu = np.zeros((len(gridy),len(gridx)))*np.nan
grid_u = np.zeros((len(gridy),len(gridx)))*np.nan

for j, y in enumerate(gridy):
	for i, x in enumerate(gridx):
		it = (lats<=y+r)&(lats>y-r)&(lons>x-r)&(lons<=x+r)
		
		if np.sum(it)<5:
			grid_shap_magu[j,i]=np.nan
			grid_u[j,i]=np.nan
		else:
			mag = np.mean(np.abs((shap5[it]+shap8[it])))
			grid_shap_magu[j,i]=mag
			grid_u[j,i] = np.mean(np.abs((lpsu[it])))

cs1 = ax1.pcolormesh(gridx,gridy,grid_shap_magu/grid_u, vmin=0, vmax=0.5, cmap=cmap)



ax2 = plt.subplot(1,2,2, projection=cartopy.crs.PlateCarree())
fname = 'mean_v500'
ix = np.where(feature_names=='mean_v500')[0][0]
iy = np.where(feature_names=='mean_v850')[0][0]

lpsv = df.merid_speed.values
shap5 = shap_valuesv[:,ix]
shap8 = shap_valuesv[:,iy]

grid_shap_magv = np.zeros((len(gridy),len(gridx)))*np.nan
grid_v = np.zeros((len(gridy),len(gridx)))*np.nan

for j, y in enumerate(gridy):
	for i, x in enumerate(gridx):
		it = (lats<=y+r)&(lats>y-r)&(lons>x-r)&(lons<=x+r)
		
		if np.sum(it)<5:
			grid_shap_magv[j,i]=np.nan
			grid_v[j,i]=np.nan
			
		else:
			mag = np.mean(np.abs((shap5[it]+shap8[it])))
			grid_shap_magv[j,i]=mag
			grid_v[j,i] = np.mean(np.abs((lpsv[it])))

cs2 = ax2.pcolormesh(gridx,gridy,grid_shap_magv/grid_v, vmin=0, vmax=0.5, cmap=cmap)


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




plt.show()
