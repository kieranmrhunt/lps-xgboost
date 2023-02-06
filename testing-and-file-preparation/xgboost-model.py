import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import linregress

df = pd.read_csv("temp_data/filtered_lps_with_enviromental.csv")

df = df[df['reached_peak']==0]
#print(df.columns)
#['x', 'y', 'frame', 'year', 'month', 'day', 'hour', 'radius', 'max_vort',
#       'mean_vort', 'mean_vort_850', 'mean_vort_700', 'mean_vort_500',
#       'max_trunc_vort', 'track_id', 'mean_swv', 'mean_u200', 'mean_u850',
#       'mean_skt', 'mean_land_frac', 'mcz_tcwv', 'vortex_depth', 'over_land',
#       'dvo850_dt', 'acc_land_time', 'total_land_time', 'qshear_850',
#       'ushear_850', 'mean_cape', 'mcz_cape', 'mean_dthetae_dp_900_750',
#       'mean_dthetae_dp_750_500', 'mean_land_skt', 'mean_sst',
#       'mean_swvl1_grad', 'mean_swvl2_grad', 'mean_swvl1', 'mean_swvl2',
 #      'olr_90', 'olr_75', 'olr_50', 'qshear_850_background',
#       'u_shear_850_background', 'mean_q_850', 'orography_height',
#       'peak_vorticity', 'reached_peak', 'mean_prcp_400', 'mean_prcp_800',
#       'max_prcp_400', 'max_prcp_800', 'mean_vimfd_400']


feature_names = np.array(['x', 'y', 'hour', 'max_vort',
       'mean_vort_850', 'mean_vort_700', 'mean_vort_500',
       'mean_swv', 'mean_u200', 'mean_u850',
       'mean_skt', 'mean_land_frac', 'mcz_tcwv',
       'mean_cape', 'mcz_cape', 'mean_dthetae_dp_900_750',
       'mean_dthetae_dp_750_500', 'mean_land_skt', 'mean_sst',
       'mean_swvl1_grad', 'mean_swvl2_grad', 'mean_swvl1', 'mean_swvl2',
       'olr_90', 'olr_50', 'qshear_850_background',  'dvo850_dt',
       'ushear_850_background', 'mean_q_850', 'orography_height',])


#feature_names = np.array(['qshear_850_background', 'u_shear_850_background', 'x'])


X = df[feature_names]


pvar = 'peak_vorticity'
y = df[[pvar]]

data_dmatrix = xgb.DMatrix(data=X,label=y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


params = {'objective':'reg:squarederror','colsample_bytree': 0.3,'learning_rate': 0.25,
          'max_depth': 6, 'alpha': 10, 'gamma' : 5}
n_estimators = 120





'''
cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
                    num_boost_round=n_estimators,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)

print(cv_results.tail())
'''


xg_reg = xgb.XGBRegressor(n_estimators = n_estimators, 
                          **params)
xg_reg.fit(X_train,y_train)
preds = xg_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
sn = rmse/np.std(y[pvar].values)
print("RMSE: %f" % (rmse))
print(sn)


print(linregress(np.array(y_test[pvar].values),np.array(preds)))

'''
remaining_actual = y_test.total_land_time-X_test.acc_land_time
remaining_predicted = preds-X_test.acc_land_time
it = X_test.over_land.values==1
plt.plot(remaining_actual[it], remaining_predicted[it], 'kx')
plt.xlabel("Actual time remaining")
plt.ylabel("Predicted time remaining")
plt.title("All preds over land")
plt.show()
'''
#xg_reg = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=n_estimators)

'''
xgb.plot_tree(xg_reg,num_trees=0)
plt.show()
'''


'''
xgb.plot_importance(xg_reg, importance_type='total_gain', show_values=False, xlabel="Total gain")
plt.title("Feature importance ({:s})\n s/n measure: {:1.2f}".format(pvar, sn))
plt.tight_layout()
plt.show()
'''



#from sklearn.inspection import permutation_importance

'''
perm_importance = permutation_importance(xg_reg, X_test, y_test)
sorted_idx = perm_importance.importances_mean.argsort()
plt.barh(feature_names[sorted_idx],perm_importance.importances_mean[sorted_idx])
plt.xlabel("Normalized permutation importance")
plt.title("Feature importance ({:s})\n s/n measure: {:1.2f}".format(pvar, sn))
plt.tight_layout()
plt.show()
'''






import shap


explainer = shap.TreeExplainer(xg_reg)
shap_values = explainer.shap_values(X_test)
#np.save("temp_data/shap_values_6.1000_no_acc.npy", shap_values)

#plt.pcolormesh(shap_values)


shap.summary_plot(shap_values, X_test, plot_type='layered_violin', color='coolwarm',show=False,)
plt.gcf().axes[-1].set_aspect(100)
plt.gcf().axes[-1].set_box_aspect(100)


#shap.dependence_plot("qshear_850_background", shap_values, X_test, interaction_index="mean_cape", cmap=plt.cm.coolwarm, show=False)
#shap.summary_plot(shap_values, X_test, plot_type='bar', show=False)

#print(shap_values.shape)




plt.tight_layout()
plt.show()

