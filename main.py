import keras
from keras import layers

# This is the size of our encoded representations
encoding_dim = 30

# This is our input image
input_img = keras.Input(shape=(122,))
# "encoded" is the encoded representation of the input
encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = layers.Dense(122, activation='sigmoid')(encoded)

# This model maps an input to its reconstruction
autoencoder = keras.Model(input_img, decoded)

# This model maps an input to its encoded representation
encoder = keras.Model(input_img, encoded)

# This is our encoded (32-dimensional) input
encoded_input = keras.Input(shape=(encoding_dim,))
# Retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# Create the decoder model
decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

import csv
import numpy as np

file = open("Geom(1).csv")
data = np.genfromtxt(file, delimiter=",")

print(data.shape)

print(data)

import matplotlib.pyplot as plt

n = 5  # How many digits we will display
plt.figure(figsize=(20,10))
for i in range(n):
  x_array=[0]
  y_array=[0]
  for f in range(0,122,2):
    x_array.append(float(data[i][f]))
    y_array.append(float(data[i][f+1]))
  ax = plt.subplot(3, n, i + 1)
  ax.set(xlim=(-150.0,150.0), ylim=(-150.0,150.0))
  ax.set_aspect('equal')
  plt.scatter(x_array,y_array)
  plt.xlabel("X axis")
  plt.ylabel("Y axis")
plt.show()

print(data.shape)

x_train = data[:17]
x_test = data[17:]

#x_train = x_train.astype('float32') / 255.
#x_test = x_test.astype('float32') / 255.

print(x_train.shape)
print(x_test.shape)

autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

# Encode and decode some digits
# Note that we take them from the *test* set
encoded_circles = encoder.predict(x_test)
decoded_circles = decoder.predict(encoded_circles)

print(encoded_circles.shape)
print(decoded_circles.shape)

import matplotlib.pyplot as plt

n = 5  # How many digits we will display
plt.figure(figsize=(20,10))
for i in range(n):
  x_array=[0]
  y_array=[0]
  for f in range(0,122,2):
    x_array.append(x_test[i][f])
    y_array.append(x_test[i][f+1])
  ax = plt.subplot(3, n, i + 1)
  ax.set(xlim=(-150.0,150.0), ylim=(-150.0,150.0))
  ax.set_aspect('equal')
  plt.scatter(x_array,y_array)
  plt.xlabel("X axis")
  plt.ylabel("Y axis")

  x_array2=[0]
  y_array2=[0]
  for f in range(0,122,2):
    x_array.append(decoded_imgs[i][f])
    y_array.append(decoded_imgs[i][f+1])
  ax = plt.subplot(3, n, i + 1 + n)
  ax.set(xlim=(-150.0,150.0), ylim=(-150.0,150.0))
  ax.set_aspect('equal')
  plt.scatter(x_array,y_array)
  plt.xlabel("X axis")
  plt.ylabel("Y axis")

plt.show()

import matplotlib.pyplot as plt

n = 5  # How many digits we will display
plt.figure(figsize=(20,10))
for i in range(n):
  x_array=[0]
  y_array=[0]
  for f in range(0,122,2):
    x_array.append(x_test[i][f])
    y_array.append(x_test[i][f+1])
  ax = plt.subplot(3, n, i + 1)
  ax.set(xlim=(-150.0,150.0), ylim=(-150.0,150.0))
  ax.set_aspect('equal')
  plt.scatter(x_array,y_array)
  plt.xlabel("X axis")
  plt.ylabel("Y axis")

  # Display original
  ax = plt.subplot(3, n, i + 1+n)
  plt.imshow(encoded_circles[i].reshape(15, 2))
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  x_array2=[0]
  y_array2=[0]
  for f in range(0,122,2):
    x_array.append(decoded_imgs[i][f])
    y_array.append(decoded_imgs[i][f+1])
  ax = plt.subplot(3, n, i + 1 + 2*n)
  ax.set(xlim=(-150.0,150.0), ylim=(-150.0,150.0))
  ax.set_aspect('equal')
  plt.scatter(x_array,y_array)
  plt.xlabel("X axis")
  plt.ylabel("Y axis")

plt.show()
