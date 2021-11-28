# from tensorflow.python.client import device_lib
#
# print(device_lib.list_local_devices())

# import tensorflow as tf
# print(tf.__version__)
# print(tf.test.is_gpu_available())
# from tensorflow.python.client import device_lib
#
#
# print(device_lib.list_local_devices())
# import tensorflow as tf
#
# print('GPU', tf.test.is_gpu_available())

# tf.compat.v1.disable_eager_execution()
#
# with tf.device('/cpu:0'):
#     a = tf.constant([1.0, 2.0, 3.0], shape=[3], name='a')
#     b = tf.constant([1.0, 2.0, 3.0], shape=[3], name='b')
# with tf.device('/gpu:1'):
#     c = a + b
#
# # 注意：allow_soft_placement=True表明：计算设备可自行选择，如果没有这个参数，会报错。
# # 因为不是所有的操作都可以被放在GPU上，如果强行将无法放在GPU上的操作指定到GPU上，将会报错。
# sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True))
# # sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
# sess.run(tf.compat.v1.global_variables_initializer())
# print(sess.run(c))
# tensor=tf.constant([[1, 2, 3], [4, 5, 6]], shape=[2, 3])


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dropout




from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pylab import mpl
import os
import pandas as pd
import tensorflow as tf
"""#二项分布
n2 = 5    # 做某件事情的次数
p2 = 0.5  # 做某件事情成功的概率（抛硬币正面朝上的概率）
X2 = np.arange(0,n2+1,1) # 做某件事成功的次数（抛硬币正面朝上的次数）
X2
np.array([0, 1, 2, 3, 4, 5])
pList2 = stats.binom.pmf(X2,n2,p2)
pList2
np.array([0.03125, 0.15625, 0.3125 , 0.3125 , 0.15625, 0.03125])
plt.plot(X2,pList2,marker='o',linestyle='None')

plt.vlines(X2,0,pList2)
plt.xlabel('随机变量：抛硬币正面朝上的次数')
plt.ylabel('概率')
plt.title('二项分布：n=%i,p=%.2f' % (n2,p2))
plt.show()"""

def data_split(path, n_predictions, n_next):#定义函数
    path_list = os.listdir(path)#确定路径
    # path_list.sort(key=lambda x: float(x.split('-')[0]))#排序在-之前的小数 从小到大排序
    print(path_list)#打印文件
    X, Y = [], []
    data = pd.read_excel(os.path.join(path, path_list[0]), index_col=0)#将文件以路径加名字的方式规定
    #将第一列作为索引列
    data.loc[data['power'] < 1, 'power'] = 0# 将小于1的置零
    data = pd.DataFrame(data, columns=['ws10', 'ws30', 'wd10', 'wd30', 'Temp_Air','Humidity','Pressure',
                                       'power'])

    # 选择提取的行啊列啊
    for i in range(data.shape[0]- n_predictions - n_next + 2 ):#第一维的空间行长度-一组的长度-间隔长度""""""
        a = data.iloc[i:(i + n_predictions), :-1]#指定行列，不包括倒数第一行，进行切片""""""
        X.append(a)
        # prediction  33的话 是8h一个大组
        b = data.iloc[(i+n_predictions-1):(i + n_predictions),-1:]#一个冒号对应一个[]
        Y.append(b)
    X = np.array(X, dtype='float64')
    Y = np.array(Y, dtype='float64')
    return X, Y
path= "E:\py\AI\winddata"
# np.set_printoptions(suppress=True)
x,y=data_split(path,12,1)

x=x.reshape(-1, 7)
y=y.reshape(-1, 1)
# print(x,y)
print(x.shape[0],y.shape[0])

ss = MinMaxScaler(feature_range=(0, 1))
x=ss.fit_transform(x)
y=ss.fit_transform(y)
x=np.reshape(x,(-1,12,7))#修改这个改变精确率
y=np.reshape(y,(-1,1))#修改这个改变精确率

a=-360
x_train=x[:a]
y_train=y[:a]
x_test=x[a:]
y_test=y[a:]

print(len(x_train),len(y_train),len(x_test),len(y_test))


model = Sequential()
model.add(LSTM(32, input_shape=(x_train.shape[1], x_train.shape[2]),return_sequences=True))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(32, return_sequences=False))
model.add(Dropout(0.1))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# 拟合神经网络
history = model.fit(x_train, y_train, epochs=50, batch_size=64, validation_data=(x_test, y_test), validation_freq=1)
# 画出学习过程
# p1 = pyplot.plot(history.history['loss'], color='blue', label='train')
# p2 = pyplot.plot(history.history['val_loss'], color='yellow', label='test')
# 保存model

# pyplot.legend(["train", "test"])


loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(loss, color='red', label='loss')
plt.plot(val_loss, color='blue', label='val_loss')
plt.legend()
pyplot.show()
# predicted_stock_price = model.predict(x_test)
# # 对预测数据还原---从（0，1）反归一化到原始范围
# predicted_stock_price = ss.inverse_transform(predicted_stock_price)
# y_test1=np.reshape(y_test,(-1,1))
# real_stock_price = ss.inverse_transform(y_test1)
# plt.plot(real_stock_price, color='red', label='WP Reality')
# plt.plot(predicted_stock_price, color='blue', label='Predicted WP')
# plt.title('WP Prediction')
# plt.xlabel('Time')
# plt.ylabel('WP Reality')
# plt.legend()
# plt.show()