# -*- coding: utf-8 -*-
# @Description:
# @author: victor
# @create time: 2021-05-24-10:11 上午
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

emission = pd.read_csv('./processed_data/北京现代.csv')
x_train = emission.iloc[0:1000, 1:4]
y_train = emission.iloc[0:1000, 4]
x_test = emission.iloc[1000:1189, 1:4]
y_test = emission.iloc[1000:1189, 4]

# plt.plot(y_train, color='red', label="Real CO2 Emission")
# plt.show()

# normalization
x_normalized = MinMaxScaler(feature_range=(0,1))
x_train = x_normalized.fit_transform(x_train)
x_test = x_normalized.transform(x_test)


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

# print(x_train)

x_train = np.reshape(x_train, (x_train.shape[0], 1, 3))
print(x_train)
x_test = np.reshape(x_test, (x_test.shape[0], 1, 3))


# construct the network
model=keras.Sequential([
    layers.LSTM(units=128, return_sequences=True),
    # layers.Dropout(0.2),
    layers.LSTM(units=128, return_sequences=True),
    # layers.Dropout(0.3),
    layers.LSTM(units=128, return_sequences=True),
    # layers.Dropout(0.1),
    # layers.LSTM(units=128, return_sequences=True),
    # layers.Dropout(0.2),
    # layers.LSTM(units=128, return_sequences=True),
    # layers.Dropout(0.1),
    # layers.LSTM(units=128, return_sequences=True),
    # layers.Dropout(0.2),
    # layers.LSTM(units=128, return_sequences=True),
    # layers.Dropout(0.1),
    layers.Dense(1)
    ])

# set the compile method
model.compile(optimizer=keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0), loss='mean_squared_error',
                  metrics=['accuracy'])

checkpoint_save_path = "./checkpoint/emission/ckpt"

if os.path.exists(checkpoint_save_path+'.index'):
    print('-------------------loading the model--------------------')
    model.load_weights(checkpoint_save_path)

cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                              save_best_only=True,
                                              save_weights_only=True,
                                              monitor='val_loss')

# train the network
history = model.fit(x_train, y_train, epochs=5000, batch_size=25,
              validation_split=0.2, validation_freq=1, callbacks=[cp_callback])

model.summary()

file = open('./weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name)+'\n')
    file.write(str(v.shape)+'\n')
    file.write(str(v.numpy())+'\n')
file.close()

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.show()

predicted_emission = model.predict(x_test)
# predicted_emission = y_normalized.inverse_transform(predicted_emission)

x=range(0, len(y_test))
plt.plot(x, y_test, color='red', label="Real CO2 Emission")
plt.plot(x, np.squeeze(predicted_emission), color='blue', label='Predicted CO2 emission')
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