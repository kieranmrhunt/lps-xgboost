import pandas as pd
from datetime import datetime as dt, timedelta as td
import numpy as np
import matplotlib.pyplot as plt

find = lambda x, arr: np.argmin(np.abs(x-arr))

df_un = pd.read_csv("temp_data/filtered_lps.csv")
df = pd.read_csv("temp_data/filtered_lps_with_enviromental.csv")


peak_vorticity = []
reached_peak = []

for _, track in df_un.groupby('track_id'):
	L = len(track)
	peak_tvorticity = np.max(track['max_trunc_vort'])
	it = np.argmax(track['max_trunc_vort'])

	peak = np.ones(L) * peak_tvorticity
	reached = np.zeros(L) 
	reached[it:] = 1
	
	peak_vorticity.extend(peak)
	reached_peak.extend(reached)
	
df_un['peak_vorticity'] = peak_vorticity
df_un['reached_peak'] = reached_peak

df_un = df_un[df_un.hour.isin([0,6,12,18])]
df_un = df_un[df_un.year<2019]

df['peak_vorticity'] = df_un['peak_vorticity'].values
df['reached_peak'] = df_un['reached_peak'].values

df.to_csv("temp_data/filtered_lps_with_enviromental.csv")


plt.plot(df.peak_vorticity.values)
plt.show()
