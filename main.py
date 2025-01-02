# importing and reading csv file
import pandas as pd

import tensorflow as tf

from sklearn.model_selection import train_test_split

dataset = pd.read_csv('cancer.csv')

# setting up the x and y attributes
# y attribute is whether it is malignant or begnin
# x attribute consists of tumor characteristics

x = dataset.drop(columns = ['diagnosis(1=m, 0=b)'])
y = dataset[['diagnosis(1=m, 0=b)']]

# creating dataset and testing set

# python -m pip install scikit-learn

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(256, input_shape=(x_train.shape[1],), activation='sigmoid'))
model.add(tf.keras.layers.Dense(256, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x=x_train, y=y_train, epochs=1000)

model.evaluate(x_test, y_test)