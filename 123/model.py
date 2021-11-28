# e_all=0.3
# m_t=10
# m_t_error=6
# e=round(m_t_error/m_t,4)
# print(e)
# #二项分布
# from scipy.special import comb
# def cal(m_t,m_t_error):
#     p=(comb(m_t,m_t_error))*(e_all**m_t_error)*((1-e_all)**(m_t-m_t_error))
#     p=round(p,4)
#     return p
# #计算所有的
# def cal_ps(m_t):
#     m_t_errors=list(range(m_t+1))
#     ps=[]
#     for i in range(len(m_t_errors)):
#         m_t_error=m_t_errors[i]
#         p=cal(m_t,m_t_error)
#         ps.append(p)
#     return m_t_errors,ps
# m_t_errors,ps=cal_ps(m_t)
# print(m_t_errors,"\n",ps)
#
# import matplotlib.pyplot as plt
# def plot_scatter(x,y):
#     plt.scatter(x,y,s=10,c='b',alpha=0.7)
#     plt.show()
#     return
# plot_scatter(m_t_errors,ps)
import numpy as np
import matplotlib.pyplot as plt
# x=2*np.random.rand(100,1)
# y=4+2*x+np.random.rand(100,1)
#
# x_b=np.c_[x,np.ones((100,1)),]
# best=np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)
# print(best)
# y0=best[0]*x+best[1]
# plt.scatter(x,y)
# plt.plot(x,y0)
# plt.show()


# def gini_index_single(a,b):
#     single_gini=1-((a/(a+b))**2)-((b/(a+b))**2)
#     return round(single_gini,2)
# # print(gini_index_single(139,39))
#
# def gini_index(a,b,c,d):
#     zuo=gini_index_single(a,b)
#     you=gini_index_single(c,d)
#     gini_index=zuo*((a+b)/(a+b+c+d))+you*((c+d)/(a+b+c+d))
#     return round(gini_index,2)
# print(gini_index(37,127,100,33))

# import tensorflow as tf
# import numpy as np
#
# data = np.loadtxt('data.txt', delimiter='“')
# X_train = data[:, 0]
# y_train = data[:, 1]
import tensorflow as tf
# import numpy as np
#
# data = np.loadtxt("data.txt",dtype=np.int32)
# print(data)

# import seaborn as sns
# from scipy import stats
# import matplotlib.pyplot as plt
# mvnorm = stats.multivariate_normal(mean=[0, 0], cov=[[1., 0.5], [0.5, 1.]])
# # Generate random samples from multivariate normal with correlation .5
# x = mvnorm.rvs(100000)
# h = sns.jointplot(x[:, 0], x[:, 1], kind='kde', stat_func=None);
# h.set_axis_labels('X1', 'X2', fontsize=16);
# norm = stats.norm()
# x_unif = norm.cdf(x)
# h = sns.jointplot(x_unif[:, 0], x_unif[:, 1], kind='hex', stat_func=None)
# h.set_axis_labels('Y1', 'Y2', fontsize=16);
# plt.savefig("multi-unif-x.png") # save fig file



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def data_split(path, n_predictions, n_next):
    path_list = os.listdir(path) #返回指定的文件夹包含的文件或文件夹的名字的列表
    path_list.sort(key=lambda x: float(x.split('-')[0]))#排序 '-'之前的内容
    print(path_list)#输出列表
    X, Y = [], []#建立x y向量or 矩阵
    for filename in path_list:
        data = pd.read_excel(os.path.join(path, filename), index_col=0)
        if len(data) > n_predictions-1:
            data.loc[data['power'] < 1, 'power'] = 0
            data = pd.DataFrame(data, columns=['ws10', 'ws30', 'ws50', 'ws70', 'ws80',
                                               'wd10', 'wd30', 'wd50', 'wd70', 'wd80', 'power'])
            # data = pd.DataFrame(data, columns=['ws10','ws50', 'ws70', 'ws80',
            #                                    'wd10', 'power'])
            for i in range(data.shape[0] - n_predictions - n_next + 2):
                a = data.iloc[i:(i + n_predictions), :-1]
                X.append(a)
                b = data.iloc[(i + n_predictions - 1):(i + n_predictions), -1:]
                Y.append(b)
    X = np.array(X, dtype='float64')
    Y = np.array(Y, dtype='float64')
    return X, Y
