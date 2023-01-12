# -*- coding: utf-8 -*-
# @Description:
# @author: victor
# @create time: 2021-05-22-9:15 下午

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
import math
import os

# def smooth_labels(labels, factor=0.1):
#     # smooth the labels
#     labels *= (1 - factor)
#     labels += (factor / len(labels))
#
#     # returned the smoothed labels
#     return labels

emission = pd.read_csv('./processed_data/北京现代.csv')
x_train = emission.iloc[0:1000, 3]
y_train = emission.iloc[0:1000, 4]
x_test = emission.iloc[1000:1189, 3]
y_test = emission.iloc[1000:1189, 4]

# normalization
x_train = np.expand_dims(x_train, axis=1)
# print(x_train)
x_test = np.expand_dims(x_test, axis=1)
x_normalized = MinMaxScaler(feature_range=(0, 1))
x_train = x_normalized.fit_transform(x_train)
x_test = x_normalized.transform(x_test)

#
y_normalized = MinMaxScaler(feature_range=(0,1))
y_train = y_normalized.fit_transform(np.array(y_train).reshape(-1,1))

# shuffle the data set
np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)

# convert the dataset type to numpy.array
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

# y_train = smooth_labels(y_train)

# k = 4
# num_value_sample = len(x_train) // k

# for i in range(k):
#     print("Processing fold #", i)
#     val_data = x_train[i * num_value_sample: (i + 1) * num_value_sample]
#     val_targets = y_train[i * num_value_sample: (i + 1) * num_value_sample]
#
#     partial_train_data = np.concatenate(
#         [x_train[:i * num_value_sample],
#          x_train[(i + 1) * num_value_sample:]],
#         axis=0)
#     partial_train_targets = np.concatenate(
#         [y_train[:i * num_value_sample],
#          y_train[(i + 1) * num_value_sample:]],
#         axis=0)

    # construct the network
model = keras.Sequential([
    layers.Dense(150, activation='tanh'),
    layers.Dropout(0.3),
    layers.Dense(200, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(150, activation='relu'),

        # layers.Dense(120, activation='relu', kernel_regularizer=keras.regularizers.L2()),
        # layers.Dropout(0.2),
        # layers.Dense(120, activation='relu'),
        # layers.Dense(120, activation='relu'),
        #
        # layers.Dense(120, activation='relu'),
        # layers.Dropout(0.2),
        # layers.Dense(120, activation='relu'),
        # layers.Dense(120, activation='relu'),
        #
        # layers.Dense(120, activation='relu', kernel_regularizer=keras.regularizers.L2()),
        # layers.Dropout(0.2),
        # layers.Dense(120, activation='relu'),
        # layers.Dense(120, activation='relu'),
        #
        # layers.Dense(120, activation='relu', kernel_regularizer=keras.regularizers.L2()),
        # layers.Dropout(0.2),
        # layers.Dense(120, activation='relu'),
        # layers.Dense(120, activation='relu'),
        #
        # layers.Dense(120, activation='relu', kernel_regularizer=keras.regularizers.L2()),
        # layers.Dropout(0.2),
        # layers.Dense(120, activation='relu'),
        # layers.Dense(120, activation='relu'),
        #
        # layers.Dense(120, activation='relu', kernel_regularizer=keras.regularizers.L2()),
        # layers.Dropout(0.2),
        # layers.Dense(120, activation='relu'),
        # layers.Dense(120, activation='relu'),
        #
        # layers.Dense(120, activation='relu', kernel_regularizer=keras.regularizers.L2()),
        # layers.Dropout(0.2),
        # layers.Dense(120, activation='relu'),
        # layers.Dense(120, activation='relu'),
        #
        # layers.Dense(120, activation='relu', kernel_regularizer=keras.regularizers.L2()),
        # layers.Dropout(0.2),
        # layers.Dense(120, activation='relu'),
        # layers.Dense(120, activation='relu'),
        #
        # layers.Dense(120, activation='relu', kernel_regularizer=keras.regularizers.L2()),
        # layers.Dropout(0.2),
        # layers.Dense(120, activation='relu'),
        # layers.Dense(120, activation='relu'),
        # #
        # layers.Dense(120, activation='relu', kernel_regularizer=keras.regularizers.L2()),
        # layers.Dropout(0.2),
        # layers.Dense(120, activation='relu'),
        # layers.Dense(120, activation='relu'),
        #
        # layers.Dense(120, activation='relu', kernel_regularizer=keras.regularizers.L2()),
        # layers.Dropout(0.2),
        # layers.Dense(120, activation='relu'),
        # layers.Dense(120, activation='relu'),

    layers.Dense(1)
])

# set the compile method
model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='mean_squared_error',
                   metrics=['accuracy'])

checkpoint_save_path = "./checkpoint/emission/ckpt"

if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------------loading the model--------------------')
    model.load_weights(checkpoint_save_path)

cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                  save_best_only=True,
                                                  save_weights_only=True,
                                                  monitor='val_loss')

# train the network
history = model.fit(x_train, y_train, epochs=2000, batch_size=10,
                    validation_split=0.2, validation_freq=1, callbacks=[cp_callback])

model.summary()

file = open('./weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.show()

predicted_emission = model.predict(x_test)
predicted_emission = y_normalized.inverse_transform(predicted_emission)

x=range(0, len(y_test))
plt.plot(x, y_test, color='red', label="Real CO2 Emission")
plt.plot(x, predicted_emission, color='blue', label='Predicted CO2 emission')
plt.title('Carbon Dioxide Emission Prediction')
plt.xlabel('quantity')
plt.ylabel('CO2 emission')
plt.legend()
plt.show()

mse = mean_squared_error(predicted_emission, y_test)
rmse = math.sqrt(mean_squared_error(predicted_emission, y_test))
mae = mean_absolute_error(predicted_emission, y_test)

print('mse:%.6f' % mse)
print('rmse:%.6f' % rmse)
print('mae:%.6f' % mae)






