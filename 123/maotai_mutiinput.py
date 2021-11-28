import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, LSTM
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
q=33
def data_split(path, n_predictions, n_next):  # 定义函数
    path_list = os.listdir(path)  # 确定路径
    # path_list.sort(key=lambda x: float(x.split('-')[0]))#排序在-之前的小数 从小到大排序
    print(path_list)  # 打印文件
    X, Y = [], []
    data = pd.read_excel(os.path.join(path, path_list[0]), index_col=0)  # 将文件以路径加名字的方式规定
    # 将第一列作为索引列
    data.loc[data['power'] < 1, 'power'] = 0  # 将小于1的置零
    data = pd.DataFrame(data, columns=['ws10', 'ws30', 'wd10', 'wd30', 'Temp_Air','Humidity','Pressure',
                                       'power'])

    # 选择提取的行啊列啊
    for i in range(data.shape[0] - n_predictions - n_next + 2):  # 第一维的空间行长度-一组的长度-间隔长度""""""
        a = data.iloc[i:(i + n_predictions), :-1]  # 指定行列，不包括倒数第一行，进行切片""""""
        X.append(a)
        # prediction  33的话 是8h一个大组
        b = data.iloc[(i+n_predictions-1):(i + n_predictions+n_next-1), -1:]  # 一个冒号对应一个[]
        Y.append(b)
    X = np.array(X, dtype='float64')
    Y = np.array(Y, dtype='float64')
    return X, Y

b=3
path = "E:\py\AI\winddata"
# np.set_printoptions(suppress=True)
x, y = data_split(path, q, b)

x = x.reshape(-1,7)
y = y.reshape(-1,b)

sc = MinMaxScaler(feature_range=(0, 1))
x= sc.fit_transform(x)  # 求得训练集的最大值，最小值这些训练集固有的属性，并在训练集上进行归一化
y = sc.fit_transform(y)
x = np.reshape(x,(-1,q,7))
y = np.reshape(y,(-1,b))
print("x===",x)
print("y===",y)


a=-360
x_train=x[:a]
y_train=y[:a]
x_test=x[a:]
y_test=y[a:]





print(len(x_train),len(y_train),len(x_test),len(y_test))
# np.random.seed(7)
# np.random.shuffle(x_train)
# np.random.seed(7)
# np.random.shuffle(y_train)
# tf.random.set_seed(7)

# x_train = x_train.reshape(len(x_train),q,5)
# y_train = y_train.reshape(len(y_train),q,1)
# x_test = x_test.reshape(len(x_test),q,5)
# y_test = y_test.reshape(len(y_test),q,1)



'''x_train, y_train = np.array(x_train), np.array(y_train)'''
# x_test, y_test = np.array(x_test), np.array(y_test)
#
#
# x_test = np.reshape(x_test, (x_test.shape[0], q, 7))


# 测试集变array并reshape为符合RNN输入要求：[送入样本数， 循环核时间展开步数， 每个时间步输入特征个数]

model = tf.keras.Sequential([
    LSTM(32,input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True),
    LSTM(64),
    Dropout(0.1),
    Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='mean_squared_error')

# checkpoint_save_path = "E:\py\AI\winddata\LSTM_wind.ckpt"
#
# if os.path.exists(checkpoint_save_path + '.index'):
#     print('-------------load the model-----------------')
#     model.load_weights(checkpoint_save_path)
#
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
#                                                  save_weights_only=True,
#                                                  save_best_only=True,
#                                                  monitor='val_loss')


history = model.fit(x_train, y_train, batch_size=64, epochs=5, validation_data=[x_test, y_test] ,validation_freq=1)
# history = model.fit(x_train, y_train, batch_size=64, epochs=5, verbose=2)
#


model.summary()

loss = history.history['loss']
val_loss = history.history['val_loss']


plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()


# y_test=np.reshape(y_test,(-1,1))
# 测试集输入模型进行预测
predicted_stock_price = model.predict(x_test)
# 对预测数据还原---从（0，1）反归一化到原始范围
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
# 对真实数据还原---从（0，1）反归一化到原始范围
real_stock_price = sc.inverse_transform(y_test[q+b:])
# 画出真实数据和预测数据的对比曲线
plt.plot(real_stock_price, color='red', label='WP Reality')
plt.plot(predicted_stock_price, color='blue', label='Predicted WP')
plt.title('WP Prediction')
plt.xlabel('Time')
plt.ylabel('WP Reality')
plt.legend()
plt.show()



# mse = mean_squared_error(predicted_stock_price, real_stock_price)
# # calculate RMSE 均方根误差--->sqrt[MSE]    (对均方误差开方)
# rmse = math.sqrt(mean_squared_error(predicted_stock_price, real_stock_price))
# # # calculate MAE 平均绝对误差----->E[|预测值-真实值|](预测值减真实值求绝对值后求均值）
# mae = mean_absolute_error(predicted_stock_price, real_stock_price)
# print('均方误差: %.6f' % mse)
# print('均方根误差: %.6f' % rmse)
# print('平均绝对误差: %.6f' % mae)
