import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import linregress
import shap
from matplotlib.colors import LinearSegmentedColormap

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
       #'mean_v200', 'mean_v500', 'mean_v850',
       'mean_skt', 'mean_land_frac', 'mcz_tcwv',
       'mean_cape', 'mcz_cape', 'mean_dthetae_dp_900_750',
       'mean_dthetae_dp_750_500', 'mean_land_skt', 'mean_sst',
       'mean_swvl1_grad', 'mean_swvl2_grad', 'mean_swvl1', 'mean_swvl2',
       'olr_90', 'olr_50', 'qshear_850_background',  'dvo850_dt',
       'ushear_850_background', 'mean_q_850', 'orography_height',])


X = df[feature_names]


pvar = 'total_land_time'
y = df[[pvar]]

data_dmatrix = xgb.DMatrix(data=X,label=y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


params = {'objective':'reg:squarederror','colsample_bytree': 0.3,'learning_rate': 0.25,
          'max_depth': 6, 'alpha': 10, 'gamma' : 5}
n_estimators = 1000



xg_reg = xgb.XGBRegressor(n_estimators = n_estimators, 
                          **params)
xg_reg.fit(X_train,y_train, early_stopping_rounds=10, eval_set = [(X_test, y_test)])
preds = xg_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
sn = rmse/np.std(y[pvar].values)
print("RMSE: %f" % (rmse))

print(linregress(np.array(y_test[pvar].values),np.array(preds)))





cmap = LinearSegmentedColormap.from_list('cmap',['mediumblue', 'tab:blue', 'tab:grey','gold','tab:orange','tab:red',][::-1])



explainer = shap.TreeExplainer(xg_reg)
shap_values = explainer.shap_values(X)
print(shap_values.shape)

fig = plt.figure(figsize=(10,5))

ax1 = plt.subplot(1,2,1)

fname = 'mcz_tcwv'
iname = 'mean_u200'
ix = np.where(feature_names==fname)[0][0]
sc = ax1.scatter(X[fname].values, shap_values[:,ix], c=X[iname].values, cmap=plt.cm.coolwarm, s=5, vmin=-25, vmax=0)
ax1.set_ylabel("Shapley value (hr)")
ax1.set_xlabel("mcz_tcwv (mm)")
ax1.set_title("(a) TCWV over the monsoon core zone")

ax2 = plt.subplot(1,2,2)

fname = 'mean_vort_850'
iname = 'mean_u200'
ix = np.where(feature_names==fname)[0][0]
sc = ax2.scatter(X[fname].values, shap_values[:,ix], c=X[iname].values, cmap=plt.cm.coolwarm, s=5, vmin=-25, vmax=0)
ax2.set_ylabel("Shapley value (hr)")
ax2.set_xlabel("mean_vort_850 (10$^{-5}$ s$^{-1}$)")
ax2.set_title("(b) Mean 850 hPa relative vorticity")

fig.subplots_adjust(bottom=0.27,)# wspace=.25)
x0,x1,x2,x3 = ax1.get_position().x0, ax1.get_position().x1, ax2.get_position().x0, ax2.get_position().x1
cax1 = fig.add_axes([0.67*(x1-x0)+x0, 0.1, (0.33*(x3-x2)+x2)-(0.67*(x1-x0)+x0), 0.05])
cb1 = fig.colorbar(sc, cax=cax1, orientation='horizontal', extend='both')
cb1.set_label("200 hPa zonal wind (m s$^{-1}$)")



plt.show()
