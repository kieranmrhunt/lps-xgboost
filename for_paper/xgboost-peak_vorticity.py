import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import linregress
import shap
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
shap_values = explainer(X_test)

'''
clustering = shap.utils.hclust(X_test, y_test)
shap.plots.bar(shap_values, clustering=clustering, clustering_cutoff=0.5, max_display=50)
plt.show()
'''


ax1 = plt.subplot(1,2,2)
shap.summary_plot(shap_values, X_test, plot_type='layered_violin', color='coolwarm',show=False, max_display=12)
plt.gcf().axes[-1].set_aspect(100)
plt.gcf().axes[-1].set_box_aspect(100)
plt.gcf().set_size_inches((10,5))

ax1.set_title("(b) Shapley value distributions")
ax1.tick_params(labelsize=10)
ax1.set_xlabel("Shapley value (10$^{-5}$ s$^{-1}$)", fontsize=10)

cb_ax = plt.gcf().axes[1]
cb_ax.tick_params(labelsize=10)
cb_ax.set_ylabel("Predictor value", fontsize=10)


ax2 = plt.subplot(1,2,1)
ax2.plot(preds.squeeze(), y_test.squeeze(), marker='o', lw=0, color='orange', alpha=0.3)
#sm_x, sm_y = sm_lowess(y_test.squeeze(), preds.squeeze(),  frac=1./2., it=5, return_sorted = True).T

sm_x = preds.squeeze()[np.argsort(preds.squeeze())]
sm_y = np.poly1d(np.polyfit(preds.squeeze(), y_test.squeeze(), 3))(sm_x)
ax2.plot(sm_x, sm_y, 'k--', label='cubic best fit')

s,i,r,p,e = linregress(y_test.squeeze(),preds.squeeze())
ax2.text(0.05, 0.95, 'r={:1.3f}'.format(r), ha='left',va='center', transform=ax2.transAxes)

ax2.set_xlabel("Predicted peak 850 hPa vorticity (10$^{-5}$ s$^{-1}$)")
ax2.set_ylabel("Actual peak 850 hPa vorticity (10$^{-5}$ s$^{-1}$)")
ax2.set_title("(a) model verificaton")
ymin, ymax = ax2.get_ylim()
xmin, xmax = ax2.get_xlim()
ax2.plot([-1000,1000],[-1000,1000], ls='--', color='grey', label='1:1')
ax2.set_xlim([xmin, xmax])
ax2.set_ylim([ymin, ymax])
plt.legend(loc='lower right')

plt.tight_layout()
plt.show()

