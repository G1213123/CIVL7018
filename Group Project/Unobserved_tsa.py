import numpy as np
from scipy.optimize import leastsq
import pylab as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import os
import sys
sys.path.insert(0, '.')
here = os.path.dirname(os.path.abspath(__file__))

from utils import *
from timeit import default_timer as timer

# 加载数据集
data_path = os.path.join(here, "PEMS03_num31.npz")
data = np.load(data_path)['data']

# 数据预处理
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data.reshape(-1, 1))
imp = SimpleImputer(missing_values=0, strategy='mean')
imp.fit(data_normalized)
data_normalized = imp.transform(data_normalized).reshape(-1)

# 使用滑动窗口创建训练集、验证集和测试集
train_size = int(len(data_normalized) * 0.7)
valid_size = int(len(data_normalized) * 0.2)
test_size = len(data_normalized) - train_size - valid_size

train_data = data_normalized[:train_size]
valid_data = data_normalized[train_size:train_size+valid_size]
test_data = data_normalized[train_size+valid_size:]

periodic = [12, 288]

fits = [{
        k: sm.tsa.UnobservedComponents(
            train_data,
            level=True, trend=True,
            #cycle=True,
            freq_seasonal=[{'period':q,'harmonics':[1,k][p==q] } for q in periodic]
            ).fit()
        for k in range(1, 10)
    }
    for p in periodic
]

def mse_test_data (data):
    return ((data-test_data)**2).mean(axis=None)

def find_max_k(models):
    min_mse = 99
    out = 1
    ms=[]
    for k, fit in models.items():
        m = mse_test_data(fit.get_forecast(valid_size+test_size).predicted_mean[valid_size:])
        print(m)
        ms.append(m)
        if  m < min_mse:
            out = k 
            min_mse = m
    return out

ks = [find_max_k(f) for f in fits]

start = timer()
best_fit = sm.tsa.UnobservedComponents(
        train_data,
        level=True, trend=True,
        #cycle=True,
        freq_seasonal=[{'period':p,'harmonics':k} for p,k in zip(periodic, ks)]
        ).fit()
end = timer()
print('runtime is: ', end -start)

mse = ((best_fit.get_forecast(test_size).predicted_mean - test_data)**2).mean(axis=None) 



def show2plot():
    fig, ax = plt.subplots(1, 2, figsize=(7,7), sharex=True, sharey=True, dpi=120)
    #axs = axs.ravel()
    ax1, ax2 = ax

    # predict
    ax1.plot(train_data, 'k')
    ax1.plot(best_fit.predict())

    # forecast
    ax2.plot(test_data, 'k')
    fc = best_fit.get_forecast(test_size)
    ax2.plot(fc.predicted_mean)
    """
    ci = ciclean(pd.DataFrame(fc.conf_int()))
    ax2.fill_between(0, ci.lower, ci.upper, alpha=.2)
    ci = ciclean(pd.DataFrame(fc.conf_int(alpha=.2)))
    ax2.fill_between(0,  ci.lower, ci.upper, alpha=.2, color='C0')"""
    ax2.grid()
    #xdate(ax, '%Y', '4ys')
    ax2.text(.1, .95, f'$k1={k1}$\n$k2={k2}$\nMSE = {mse}', va='top', transform=ax2.transAxes)
    fig.text(0, .5, 'Turnover', rotation=90, va='center')
    plt.tight_layout()

def show1plot():
    fig, ax = plt.subplots(1, 1, figsize=(7,7), sharex=True, sharey=True, dpi=120)

    # forecast
    ax.plot(test_data, 'orange', label='True Values')
    fc = best_fit.get_forecast(test_size)
    ax.plot(fc.predicted_mean, label='Predictions')

    ax.legend()
    plt.xlabel("Time")
    plt.ylabel("Traffic Flow")
    
    ks_text = '\n'.join([f'$k{i}={k}$' for i, k in enumerate(ks)])
    ax.text(.1, .95, f'{ks_text}\nMSE = {mse}', va='top', transform=ax.transAxes)
    plt.tight_layout()

show1plot()
best_fit.plot_components()
print(best_fit.summary())

plt.show()