import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from dtreeviz.trees import *
from scipy.stats import linregress

df = pd.read_csv("../temp_data/filtered_lps_with_enviromental.csv")


feature_names = np.array(['x', 'y', 'hour', 'month',
       'mean_vort_850', 'mean_vort_500', 'vortex_depth',
       'mean_u200', 'mean_u850', 'mean_u500',
       'mean_v200', 'mean_v500', 'mean_v850',
       'mean_skt', 'mean_land_frac', 'mcz_tcwv',
       'mean_cape', 'mcz_cape', 'mean_dthetae_dp_900_750',
       'mean_dthetae_dp_750_500', 'mean_land_skt', 'mean_sst',
       'mean_swvl1_grad', 'mean_swvl2_grad', 'mean_swvl1', 'mean_swvl2',
       'olr_90', 'olr_50', 'qshear_850_background',  'dvo850_dt',
       'ushear_850_background', 'mean_q_850', 'orography_height',])


X = df[feature_names]
pvar = 'mean_prcp_400'
y = df[[pvar]]

data_dmatrix = xgb.DMatrix(data=X,label=y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
params = {'objective':'reg:squarederror','colsample_bytree': 0.3,'learning_rate': 0.25,
          'max_depth': 3, 'alpha': 10, 'gamma' : 5}
n_estimators = 5



xg_reg = xgb.XGBRegressor(n_estimators = n_estimators, 
                          **params)
xg_reg.fit(X_train,y_train, early_stopping_rounds=20, eval_set = [(X_test, y_test)])
preds = xg_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
sn = rmse/np.std(y[pvar].values)
print("RMSE: %f" % (rmse))

print(linregress(np.array(y_test[pvar].values),np.array(preds)))

'''
#feature_names = X.keys()
viz = dtreeviz(xg_reg, X_train, y_train.squeeze(), target_name=pvar, feature_names=list(feature_names), tree_index=0)

viz.view()
'''


xgb.plot_tree(xg_reg, num_trees = 0)
plt.show()





