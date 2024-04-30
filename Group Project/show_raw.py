import numpy as np
from scipy.optimize import leastsq
import pylab as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared
from sklearn.impute import SimpleImputer
import pandas as pd
import os
import sys
sys.path.insert(0, '.')
here = os.path.dirname(os.path.abspath(__file__))

from utils import *
from timeit import default_timer as timer

# 加载数据集
data_path = os.path.join(here, "PEMS03.npz")
data = np.load(data_path)['data']

# 数据预处理
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data.reshape(-1, 1))
imp = SimpleImputer(missing_values=0, strategy='mean')
imp.fit(data_normalized)
data_normalized = imp.transform(data_normalized).reshape(-1)

plt.plot(data_normalized)
plt.show()