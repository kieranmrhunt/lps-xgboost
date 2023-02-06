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

def mean(angles, deg=True):
    a = np.deg2rad(angles) if deg else np.array(angles)
    angles_complex = np.frompyfunc(cmath.exp, 1, 1)(a * 1j)
    mean = cmath.phase(angles_complex.sum()) % (2 * np.pi)
    return round(np.rad2deg(mean) if deg else mean, 7)

def corrcoef(x, y, deg=True):
    convert = np.pi / 180.0 if deg else 1
    sx = np.frompyfunc(np.sin, 1, 1)((x - mean(x, deg)) * convert)
    sy = np.frompyfunc(np.sin, 1, 1)((y - mean(y, deg)) * convert)
    r = (sx * sy).sum() / np.sqrt((sx ** 2).sum() * (sy ** 2).sum())

    return round(r, 7)

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

feature_names = np.array(['y', 'hour', 'month',
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
n_estimators = 1000

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
shap_valuesu = explaineru(X_test)

explainerv = shap.TreeExplainer(xg_regv)
shap_valuesv = explainerv(X_test)


ax1 = plt.subplot(1,2,1)
shap.summary_plot(shap_valuesu, X_test, plot_type='layered_violin', color='coolwarm',show=False, max_display=12)
plt.gcf().axes[-1].set_aspect(100)
plt.gcf().axes[-1].set_box_aspect(100)

s,i,r,p,e = linregress(y_testu.squeeze(),predsu.squeeze())
ax1.set_title("(a) zonal speed (r={:1.3f})".format(r))
ax1.tick_params(labelsize=10)
ax1.set_xlabel("Shapley value (km hr$^{-1}$)", fontsize=10)

cb_ax = plt.gcf().axes[1]
cb_ax.tick_params(labelsize=10)
cb_ax.set_ylabel("Predictor value", fontsize=10)




ax2 = plt.subplot(1,2,2)
shap.summary_plot(shap_valuesv, X_test, plot_type='layered_violin', color='coolwarm',show=False, max_display=12)
plt.gcf().axes[-1].set_aspect(100)
plt.gcf().axes[-1].set_box_aspect(100)
plt.gcf().set_size_inches((10,5))

s,i,r,p,e = linregress(y_testv.squeeze(),predsv.squeeze())
ax2.set_title("(b) meridional speed (r={:1.3f})".format(r))
ax2.tick_params(labelsize=10)
ax2.set_xlabel("Shapley value (km hr$^{-1}$)", fontsize=10)

cb_ax = plt.gcf().axes[-1]
cb_ax.tick_params(labelsize=10)
cb_ax.set_ylabel("Predictor value", fontsize=10)
'''



fig = plt.figure(figsize=(10,5))
y_test = np.hypot(y_testu, y_testv)
preds = np.hypot(predsu, predsv)

ax3 = plt.subplot(1,2,1)
ax3.plot(preds.squeeze(), y_test.squeeze(), marker='o', lw=0, color='orange', alpha=0.3)
sm_x, sm_y = sm_lowess(y_test.squeeze(), preds.squeeze(),  frac=1./2., it=5, return_sorted = True).T
s,i,r,p,e = linregress(y_test.squeeze(),preds.squeeze())
ax3.text(0.05, 0.95, 'r={:1.3f}'.format(r), ha='left',va='center', transform=ax3.transAxes)

ax3.plot(sm_x, sm_y, 'k--')
ax3.set_xlabel("Predicted propagation speed (km hr$^{-1}$)")
ax3.set_ylabel("Actual propagation speed (km hr$^{-1}$)")
ax3.set_title("(a) LPS propagation speed")
ymin, ymax = ax3.get_ylim()
xmin, xmax = ax3.get_xlim()
ax3.plot([-1000,1000],[-1000,1000], ls='--', color='grey')
ax3.set_xlim([xmin, xmax])
ax3.set_ylim([ymin, ymax])


ax4 = plt.subplot(1,2,2)
act_headings = (270-(np.arctan2(y_testv.squeeze(),y_testu.squeeze())*180/np.pi))%360
pred_headings = (270-(np.arctan2(predsv.squeeze(),predsu.squeeze())*180/np.pi))%360
it = y_test.squeeze().values>5

diffs = np.abs((act_headings[it]-pred_headings[it]+180)%360-180)

cmap = LinearSegmentedColormap.from_list('cmap',['orange','tab:red','purple'])
sc = ax4.scatter(pred_headings[it], act_headings[it], marker='o', c=diffs, cmap = cmap, alpha=0.3, vmin=0)
ax4.set_title("(b) LPS propagation heading")
ax4.set_xlabel("Predicted heading")
ax4.set_ylabel("Actual heading")

r = corrcoef(act_headings, pred_headings, deg=True)
ax4.text(0.05, 0.95, 'r$_c$={:1.3f}'.format(r), ha='left',va='center', transform=ax4.transAxes)

ymin, ymax = ax4.get_ylim()
xmin, xmax = ax4.get_xlim()
ax4.plot([-1000,1000],[-1000,1000], ls='--', color='grey')
ax4.set_xlim([xmin, xmax])
ax4.set_ylim([ymin, ymax])

ax4.set_xticks(np.arange(0,400,45), labels = ['S','SW','W','NW','N','NE','E','SE','S'])
ax4.set_yticks(np.arange(0,400,45), labels = ['S','SW','W','NW','N','NE','E','SE','S'])

fig.subplots_adjust(right=0.875)
y0, y1 = ax4.get_position().y0, ax4.get_position().y1
cax1 = fig.add_axes([.885, y0, 0.025, (y1-y0)])
cb1 = fig.colorbar(sc, cax=cax1, orientation='vertical')
cb1.set_label("Absolute heading error (°)")
'''

plt.show()





