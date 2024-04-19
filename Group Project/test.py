
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
# 加载数据集
data_path = r"C:\Users\1213123\Documents\Scripts\CIVL7018\Group Project\PEMS03_num31.npz"
data = np.load(data_path)['data']

# 绘制测试集预测结果对比图
plt.figure(figsize=(14, 7))
plt.plot(data, label='True Values')
plt.title('CNN: True Values vs Predictions')
plt.xlabel('Time')
plt.ylabel('Traffic Flow')
plt.legend()
plt.show()