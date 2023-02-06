from bayes_opt import BayesianOptimization
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np


df = pd.read_csv("temp_data/filtered_lps_with_enviromental.csv")

print(df)
print(df.columns)

X = df[['x', 'y', 'hour', 'max_vort',
       'mean_vort_850', 'mean_vort_700', 'mean_vort_500',
       'mean_swv', 'mean_u200', 'mean_u850',
       'mean_skt', 'mean_land_frac', 'mcz_tcwv',
       'acc_land_time', 'mean_cape', 'mcz_cape', 'mean_dthetae_dp_900_750',
       'mean_dthetae_dp_750_500', 'mean_land_skt', 'mean_sst',
       'mean_swvl1_grad', 'mean_swvl2_grad', 'mean_swvl1', 'mean_swvl2',
       'olr_90', 'olr_50', 'qshear_850_background',  'dvo850_dt',
       'ushear_850_background', 'mean_q_850', 'orography_height']]

pvar = 'total_land_time'
y = df[[pvar]]

data_dmatrix = xgb.DMatrix(data=X,label=y)


def bo_tune_xgb(gamma,alpha,eta, learning_rate):
	params = {'max_depth': 6,
        	  'gamma': gamma,
			  'alpha':alpha,
        	  'n_estimators': 120,
        	  'learning_rate':learning_rate,
        	  'subsample': 0.8,
        	  'eta': eta,
        	  'eval_metric': 'rmse','verbosity':0}
	#Cross validating with the specified parameters in 5 folds and 70 iterations
	cv_result = xgb.cv(params, data_dmatrix, num_boost_round=70, nfold=5)
	#Return the negative RMSE
	return -1.0 * cv_result['test-rmse-mean'].iloc[-1]


#Invoking the Bayesian Optimizer with the specified parameters to tune
xgb_bo = BayesianOptimization(bo_tune_xgb, {
                                             'gamma': (0, 5),
											 'alpha': (0,20),
											 'eta': (0,10),
                                             'learning_rate':(0,1),
                                            })

#performing Bayesian optimization for 5 iterations with 8 steps of random exploration with an #acquisition function of expected improvement
L = xgb_bo.maximize(n_iter=250, init_points=10, acq='ei')
print(L)
