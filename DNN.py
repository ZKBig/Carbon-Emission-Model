# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 09:02:04 2020

@author: Gavin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 从Keras.datasets数据集中导入 Boston_Housing Problem
# from keras.datasets import boston_housing
# 从Keras导入构建神经网络的库
from tensorflow.keras import models
from tensorflow.keras import layers
import tensorflow.keras as keras

# 导入数据
# !数据类型均为 np.ndarray
path = 'CO2_min.csv'
df=pd.read_csv(path)
train_data = df.iloc[0:5000,0:2]
train_targets = df.iloc[0:5000,3]
test_data = df.iloc[5000:6000,0:2]
test_targets = df.iloc[5000:6000,3]

# !Dataframe 转成 ndarray
train_data = np.around(train_data.values, 1)
train_targets = np.around(train_targets.values, 1)
test_data = np.around(test_data.values, 1)
test_targets = np.around(test_targets.values, 1)

'''
(train_data, train_targets),(test_data, test_targets) = \
    boston_housing.load_data()
'''

# 数据标准化
mean = train_data.mean(axis = 0)
train_data -= mean
std = train_data.std(axis = 0)
train_data /= std

# !用于测试数据标准化的均值和标准差都是在训练数据上计算得到的
test_data -= mean
test_data /= std


# 构建神经网络
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                           input_shape = (train_data.shape[1],)))
    
    model.add(layers.Dense(64, activation='relu'))
    # !网络的最后一层只有一个单元,没有激活,是一个线性层
    # !这是标量回归（标量回归是预测单一连续值的回归）的典型设置
    model.add(layers.Dense(1))
    # !编译网络用的是mse损失函数,即均方误差（MSE, mean squared error）
    # !预测值与目标值之差的平方,这是回归问题常用的损失函数
    
    # !平均绝对误差（MAE, mean absolute error）
    # !是预测值与目标值之差的绝对值
    model.compile(optimizer=keras.optimizers.Adam(0.001), loss='mse', metrics=['accuracy'])
    return model

# K折验证
k = 4
num_value_sample = len(train_data) // k
num_epochs = 80
all_scores = []

for i in range(k):
    print("Processing fold #", i)
    # !提取第i折的训练集
    val_data = train_data[i * num_value_sample : (i + 1) * num_value_sample]
    val_targets = train_targets[i * num_value_sample : \
                                (i + 1) * num_value_sample]
        
    partial_train_data = np.concatenate(
        [train_data[:i * num_value_sample],
         train_data[(i + 1) * num_value_sample:]],
        axis = 0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_value_sample],
         train_targets[(i + 1) * num_value_sample:]],
        axis = 0)
    
    # !使用训练集训练
    model = build_model()
    model.fit(partial_train_data, partial_train_targets,
              epochs = num_epochs,
              batch_size = 1,
              verbose=1)

    # !使用验证集验证
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=1)
    all_scores.append(val_mae)
    print("fold #", i, ": ", val_mae)
    
print("MAE: ", np.mean(all_scores))

