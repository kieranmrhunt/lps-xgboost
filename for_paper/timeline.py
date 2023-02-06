import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess as  sm_lowess
from matplotlib.ticker import StrMethodFormatter
from matplotlib.gridspec import GridSpec

def lowess(x, y, f=1./3.):
    """
    Basic LOWESS smoother with uncertainty. 
    Note:
        - Not robust (so no iteration) and
             only normally distributed errors. 
        - No higher order polynomials d=1 
            so linear smoother.
    """
    # get some paras
    xwidth = f*(x.max()-x.min()) # effective width after reduction factor
    N = len(x) # number of obs
    # Don't assume the data is sorted
    order = np.argsort(x)
    # storage
    y_sm = np.zeros_like(y)
    y_stderr = np.zeros_like(y)
    # define the weigthing function -- clipping too!
    tricube = lambda d : np.clip((1- np.abs(d)**3)**3, 0, 1)
    # run the regression for each observation i
    for i in range(N):
        dist = np.abs((x[order][i]-x[order]))/xwidth
        w = tricube(dist)
        # form linear system with the weights
        A = np.stack([w, x[order]*w]).T
        b = w * y[order]
        ATA = A.T.dot(A)
        ATb = A.T.dot(b)
        # solve the syste
        sol = np.linalg.solve(ATA, ATb)
        # predict for the observation only
        yest = A[i].dot(sol)# equiv of A.dot(yest) just for k
        place = order[i]
        y_sm[place]=yest 
        sigma2 = (np.sum((A.dot(sol) -y [order])**2)/N )
        # Calculate the standard error
        y_stderr[place] = np.sqrt(sigma2 * 
                                A[i].dot(np.linalg.inv(ATA)
                                                    ).dot(A[i]))
    return y_sm, y_stderr



df = pd.read_csv("../temp_data/filtered_lps_with_enviromental.csv")
prcp = np.load("../temp_data/composite_prcp_8x0.25.npy")*3600

lifetime = []

for _,track in df.groupby('track_id'):
	lifetime.extend(np.linspace(0,1,len(track)))

lifetime = np.array(lifetime)

fig = plt.figure(figsize=(8,6))

gs1 = GridSpec(5, 3, left=0.15, right=0.85, wspace=0.5, hspace=0.5)
ax1 = fig.add_subplot(gs1[:-2,0])
ax2 = fig.add_subplot(gs1[:-2,1])
ax3 = fig.add_subplot(gs1[:-2,2])


ax1.plot(lifetime, df.mean_vort_850, marker ='.', mec='tab:gray', lw=0, markersize=0.25)
#sm_x, sm_y = sm_lowess(df.mean_vort_850, lifetime,  frac=1./5., it=5, return_sorted = True).T
sm_y, std_y = lowess(lifetime, df.mean_vort_850, f=1./5.)
order=np.argsort(lifetime)
ax1.plot(lifetime[order], sm_y[order], 'k--')
ax1.fill_between(lifetime[order], sm_y[order] - 2.58*std_y[order],
                 sm_y[order] + 2.58*std_y[order], alpha=0.5, color='tab:red')
ax1.set_ylabel("850 hPa vorticity (10$^{-5}$ s$^{-1}$)")


ax2.plot(lifetime, df.dvo850_dt*4, marker ='.', mec='tab:gray', lw=0, markersize=0.25)
sm_y, std_y = lowess(lifetime, df.dvo850_dt*4, f=1./5.)
order=np.argsort(lifetime)
ax2.plot(lifetime[order], sm_y[order], 'k--')
ax2.fill_between(lifetime[order], sm_y[order] - 2.58*std_y[order],
                 sm_y[order] + 2.58*std_y[order], alpha=0.5, color='tab:blue')
ax2.set_ylabel("$\partial\zeta_{850}/\partial t$ (10$^{-5}$ s$^{-1}$ day$^{-1}$)")
ax2.set_xlabel("Standardised LPS age")

ax3.plot(lifetime, df.mean_land_frac, marker ='.', mec='tab:gray', lw=0, markersize=0.25)
sm_y, std_y = lowess(lifetime, df.mean_land_frac, f=1./5.)
order=np.argsort(lifetime)
ax3.plot(lifetime[order], sm_y[order], 'k--')
ax3.fill_between(lifetime[order], sm_y[order] - 2.58*std_y[order],
                 sm_y[order] + 2.58*std_y[order], alpha=0.5, color='tab:orange')
ax3.set_ylim([0,1])
ax3.set_ylabel("Land fraction")

for ax in [ax1,ax2,ax3]:
	ax.set_xlim([0,1])

gs2 = GridSpec(3, 5, left=0.15, right=0.85, wspace=0.0)
axes = [fig.add_subplot(gs2[-1,i], aspect='equal') for i in range(5)]
xgrid = np.arange(-8,8,0.25)

for n, ax in enumerate(axes):
	ip = (lifetime>=(n/5)) & (lifetime<=((n+1)/5))
	cs=ax.contourf(xgrid, xgrid, np.nanmean(prcp[ip], axis=0), levels = np.arange(0,2.75,0.25), cmap = plt.cm.terrain_r)
	ax.contour(xgrid, xgrid, np.nanmean(prcp[ip], axis=0), levels = np.arange(0,2.75,0.25), colors='k', linewidths=0.5)
	ax.yaxis.set_major_formatter(StrMethodFormatter(u"{x:0.0f}°"))
	ax.xaxis.set_major_formatter(StrMethodFormatter(u"{x:0.0f}°"))
	ax.set_title("{}$\leq L<${}".format(n/5, (n+1)/5), fontsize='medium')
	
	if n>0: ax.set_yticks([])
	if n==0: ax.set_ylabel("Relative latitude")
	if n==2: ax.set_xlabel("Relative longitude")
	
	
	#plt.colorbar(cs,ax=ax)

ax = axes[-1]
y0, y1 = ax.get_position().y0, ax.get_position().y1
cax1 = fig.add_axes([.885, y0, 0.025, (y1-y0)])
cb1 = fig.colorbar(cs, cax=cax1, orientation='vertical')
cb1.set_label("Mean precip (mm hr$^{-1}$)")

plt.show()






