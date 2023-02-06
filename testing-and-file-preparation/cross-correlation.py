import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

df = pd.read_csv("temp_data/filtered_lps_with_enviromental.csv")

print(df)
print(df.columns)


feature_names = np.array(['x', 'y', 'acc_land_time', 'orography_height',
       'mean_land_frac', 'mean_skt', 
       'mean_swvl1',  'mean_swvl1_grad',
	   
	   
	   'mean_u200', 'mean_u850',
       'mcz_tcwv', 'mean_q_850', 
       'mean_cape', 'mcz_cape', 'mean_dthetae_dp_900_750',
       'mean_dthetae_dp_750_500',
       
       'olr_90', 'olr_50', 'mean_prcp_400',
	   'qshear_850_background',
       'ushear_850_background', 
	   
	   'mean_vort_850', 'dvo850_dt', 'vortex_depth', 'peak_vorticity',
        ])


X = df[feature_names]
pvar = 'total_land_time'
y = df[[pvar]]

cmap = LinearSegmentedColormap.from_list('cmap', ['blue','tab:blue','cyan','w','w','orange','tab:red','brown'][::-1], N=20)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.001, random_state=123)


correlations = X_train.corr()

fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(correlations, vmax=0.75, center=0, fmt='.2f', cmap=cmap, square=True,
            linewidths=0.5, annot=True,cbar_kws={"shrink":.7, "label":"Correlation coefficient", "extend":"both"},annot_kws={"size":6.5},
			vmin=-0.75,
			)


plt.tight_layout()
plt.show()
