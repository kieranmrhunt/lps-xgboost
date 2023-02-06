import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

df = pd.read_csv("../temp_data/filtered_lps_with_enviromental.csv")


print(df)
print(df.columns)


feature_names = np.array(['x', 'y', 'orography_height', 'total_land_time',
       'mean_land_frac', 'mean_skt', 
       'mean_swvl1',  'mean_swvl1_grad',
	   
	   
	   'mean_u200', 'mean_u500', 'mean_u850', 'zonal_speed',
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

cmap = LinearSegmentedColormap.from_list('cmap', ['blue','tab:blue','cyan','w','w','orange','tab:red','brown'], N=20)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.001, random_state=123)


correlations = X_train.corr()

fig, ax = plt.subplots(figsize=(9,9))
hm = sns.heatmap(correlations, vmax=0.75, center=0, fmt='.2f', cmap=cmap, square=True,
            linewidths=0.5, linecolor='grey', clip_on=False, annot=True,annot_kws={"size":6.25},
			vmin=-0.75,  cbar=False
			)

#fig.colorbar(ax.collections[0], ax=ax, location="right", use_gridspec=False, label='Correlation coefficient', extend='both', shrink=0.7)


for label in ax.get_yticklabels():
    if label.get_text() in ['zonal_speed', 'peak_vorticity','mean_prcp_400','total_land_time','dvo850_dt']:
        label.set_weight("bold")

for label in ax.get_xticklabels():
    if label.get_text() in ['zonal_speed', 'peak_vorticity','mean_prcp_400','total_land_time','dvo850_dt']:
        label.set_weight("bold")



plt.tight_layout()
plt.show()
