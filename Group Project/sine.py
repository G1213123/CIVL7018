import numpy as np
from scipy.optimize import leastsq
import pylab as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared
import pandas as pd
import os
import sys
sys.path.insert(0, '.')
here = os.path.dirname(os.path.abspath(__file__))

from utils import *


# 加载数据集
data_path = os.path.join(here, "PEMS03_num31.npz")
data = np.load(data_path)['data']

# 数据预处理
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data.reshape(-1, 1)).reshape(-1)
t= np.arange(0, data.size, 1)

fits = {
    k: sm.tsa.UnobservedComponents(
        data_normalized,
        level=True, trend=True,
        freq_seasonal=[{'period':250,'harmonics':k}]
        ).fit()
    for k in range(9, 10)
}

fig, ax = plt.subplots(1, 1, figsize=(7,7), sharex=True, sharey=True, dpi=120)
#axs = axs.ravel()
fit = next(iter(fits.items()))[1]
k = next(iter(fits.items()))[0]
ax.plot(data_normalized, 'k')
ax.plot(fit.predict(), 'g', linewidth=0.5)
lag = np.linspace(len(data_normalized), len(data_normalized)+500, num=500)
fc = fit.get_forecast(500)
ax.plot(lag,fc.predicted_mean)
ci = ciclean(pd.DataFrame(fc.conf_int()))
ax.fill_between(lag, ci.lower, ci.upper, alpha=.2)
ci = ciclean(pd.DataFrame(fc.conf_int(alpha=.2)))
ax.fill_between(lag,  ci.lower, ci.upper, alpha=.2, color='C0')
ax.grid()
#xdate(ax, '%Y', '4ys')
ax.text(.1, .95, f'$k={k}$\nAICc = {fit.aicc:.2f}', va='top', transform=ax.transAxes)
fig.text(0, .5, 'Turnover', rotation=90, va='center')
plt.tight_layout()
plt.show()