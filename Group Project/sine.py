import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
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


# 使用滑动窗口创建训练集、验证集和测试集
train_size = int(len(data_normalized) * 0.7)
valid_size = int(len(data_normalized) * 0.2)
test_size = len(data_normalized) - train_size - valid_size

train_data = data_normalized[:train_size]
valid_data = data_normalized[train_size:train_size+valid_size]
test_data = data_normalized[train_size+valid_size:]


fits = {
    k: sm.tsa.UnobservedComponents(
        train_data,
        level=True, trend=True,
        freq_seasonal=[{'period':288,'harmonics':k}]
        ).fit()
    for k in range(2,3)
}

fig, ax = plt.subplots(1, 1, figsize=(7,7), sharex=True, sharey=True, dpi=120)
#axs = axs.ravel()
fit = next(iter(fits.items()))[1]
k = next(iter(fits.items()))[0]
ax.plot(test_data, 'k')
#ax.plot(fit.predict(), 'g', linewidth=0.5)
lag = np.linspace(0, len(test_data), num=len(test_data))
fc = fit.get_forecast(len(valid_data)+len(test_data))
ax.plot(lag,fc.predicted_mean[:len(test_data)])
ci = ciclean(pd.DataFrame(fc.conf_int()))
#ax.fill_between(lag, ci.lower, ci.upper, alpha=.2)
ci = ciclean(pd.DataFrame(fc.conf_int(alpha=.2)))
#ax.fill_between(lag,  ci.lower, ci.upper, alpha=.2, color='C0')
ax.grid()
#xdate(ax, '%Y', '4ys')


# calc MSE
mse = np.mean((fc.predicted_mean[len(valid_data):] - test_data) ** 2)
print(mse)

ax.text(.1, .95, f'$k={k}$\nMSE = {mse}', va='top', transform=ax.transAxes)
fig.text(0, .5, 'Turnover', rotation=90, va='center')
plt.tight_layout()
plt.show()
